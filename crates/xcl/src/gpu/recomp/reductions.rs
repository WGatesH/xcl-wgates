//! Whole-wave reductions of vertex / pixel shaders, recognised as a whole
//! (docs/wave64-on-wave32.md, "Whole-wave reductions").
//!
//! The console's compiler makes a wave reduction – `WaveActiveMin` / `Max` / `BitOr` of a
//! uint, as far as the retail pack shows – into whole-wave code. In a wave64 program:
//!
//! ```text
//! s_or_saveexec_b64 s[S:S+1], -1          EXEC saved, every lane on
//! s_and_b64 s[M:M+1], s[L:L+1], s[S:S+1]  the lanes that take part (L: the live pixels);
//!                                         without it the mask is the saved EXEC (M = S)
//! v_cndmask_b32 vA, id, x, s[M:M+1]       x in those lanes, the op's identity elsewhere
//! op_dpp8 vA, vA, vA                      three times, over 8 lanes: the other lane of the
//!                                         pair, a lane of the other pair, of the other quad
//! op_dpp vA, vA, vA row_mirror            over the row (16 lanes)
//! v_permlanex16_b32 vB, vA, 0, 0          the other row of the 32-lane half
//! op vA, vB, vA                           over the half
//! v_readlane_b32 sH, vA, 32               the upper half's
//! op vA, sH, vA                           lanes 0..31: over the wave
//! v_readlane_b32 sT, vA, 0                the wave's
//! s_mov_b64 exec, s[S:S+1]
//! ```
//!
//! A wave32 program has the same up to `op vA, vB, vA` (the wave, then), with the 32-bit
//! forms (`s_or_saveexec_b32`, `s_and_b32`, a mask in one SGPR), then `v_readlane_b32 sT, vA,
//! 0` and `s_mov_b32 exec_lo, sS`. The op is min_u32 (identity -1), max_u32 (0) or or_b32 (0);
//! `s_waitcnt` / `s_nop` / `v_nop` may stand anywhere between.
//!
//! On the console all lanes of the wave run it, inactive ones included. The PC runs only the
//! lanes it launched, and in a partial wave – a primitive's edge, the last vertices of a draw
//! – the DPP / `v_permlanex16` / `v_readlane` steps read lanes that do not exist: 0 on NVIDIA,
//! so a min gave 0 there. A wave64 program in 32-lane waves never has lanes 32..63.
//!
//! Lanes the PC did not launch hold the identity on the console: the mask is within the saved
//! EXEC, and EXEC outside whole-wave code holds only lanes the program was started with (as the
//! program uses it). So each value the sequence leaves is the op over the mask's lanes of a
//! part of the wave, which wave operations give at any wave size: per 16-lane row k,
//! `WaveActiveMin(row == k ? q : id)` with q the lane's x if it is in the mask, the identity
//! if not; the halves and the wave from the rows. Every register the steps write gets what the
//! console leaves there: vA the wave's result (wave64: in lanes 0..31, the upper half's in
//! 32..63), vB the other row's of its half, sH and sT. A wave operation leaves out helper
//! lanes, which a mask taken from EXEC after `s_wqm` holds: in a pixel shader q is first the op
//! over the lane's quad (quad reads see helper lanes), as in the console's 2 × 2 steps.
//!
//! Only an exactly matched sequence is replaced; the SALU instructions around it stay as they
//! are. The emitter (`Em::reduction`) uses this for VS / PS at any wave size, so it also
//! changes the HLSL GPUs with 64-lane waves get for those programs.

use std::collections::BTreeSet;

use super::emit::ext_encoding;
use crate::gpu::rdna2::{self, Decoded, Variant};

/// The reduction's operation.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Op {
    MinU32,
    MaxU32,
    OrB32,
}

impl Op {
    fn of(name: &str) -> Option<Op> {
        Some(match name {
            "V_MIN_U32" => Op::MinU32,
            "V_MAX_U32" => Op::MaxU32,
            "V_OR_B32" => Op::OrB32,
            _ => return None,
        })
    }

    /// The value that leaves the other operand unchanged.
    pub fn identity(self) -> u32 {
        match self {
            Op::MinU32 => u32::MAX,
            Op::MaxU32 | Op::OrB32 => 0,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Op::MinU32 => "min_u32",
            Op::MaxU32 => "max_u32",
            Op::OrB32 => "or_b32",
        }
    }

    /// The HLSL wave operation.
    pub fn wave(self) -> &'static str {
        match self {
            Op::MinU32 => "WaveActiveMin",
            Op::MaxU32 => "WaveActiveMax",
            Op::OrB32 => "WaveActiveBitOr",
        }
    }

    /// `a op b` in HLSL.
    pub fn hlsl(self, a: &str, b: &str) -> String {
        match self {
            Op::MinU32 => format!("min({a}, {b})"),
            Op::MaxU32 => format!("max({a}, {b})"),
            Op::OrB32 => format!("({a} | {b})"),
        }
    }

    /// The op on two values (tests, census).
    pub fn apply(self, a: u32, b: u32) -> u32 {
        match self {
            Op::MinU32 => a.min(b),
            Op::MaxU32 => a.max(b),
            Op::OrB32 => a | b,
        }
    }
}

/// A recognised reduction: what the emitter needs to replace it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Reduction {
    pub op: Op,
    /// the 64-bit forms (a wave64 program)
    pub wave64: bool,
    /// the `v_cndmask_b32`: the reduction is emitted there
    pub at: u64,
    /// the instructions after it, up to the last `v_readlane_b32`: emitted as nothing
    pub rest: Vec<u64>,
    /// x, the `v_cndmask`'s SRC1 (operand code: 256 + n for vN), and the instruction's literal
    pub x: u64,
    pub literal: Option<u32>,
    /// the mask: its (first) SGPR
    pub mask: u64,
    /// vA (the steps' register), vB (`v_permlanex16`'s)
    pub acc: u64,
    pub other: u64,
    /// sH (`v_readlane` 32, wave64 only), sT (`v_readlane` 0)
    pub half: Option<u64>,
    pub total: u64,
}

/// No effect in the translation (`emit::NOOPS`): may stand anywhere in the sequence.
fn skipped(d: &Decoded) -> bool {
    matches!(d.inst.name, "S_NOP" | "V_NOP") || d.inst.name.starts_with("S_WAITCNT")
}

/// An integer constant source (inline or literal).
fn constant(code: u64, literal: Option<u32>) -> Option<u32> {
    Some(match code {
        128..=192 => (code - 128) as u32,
        193..=208 => (-((code - 192) as i32)) as u32,
        255 => literal?,
        _ => return None,
    })
}

/// A mask register the translation names: an SGPR pair (s0..s105, VCC) for a wave64
/// program, an SGPR (s0..s105, VCC's words) for a wave32 one.
fn mask_register(r: u64, wave64: bool) -> bool {
    if wave64 {
        r <= 104 || r == 106
    } else {
        r <= 107
    }
}

/// A single SGPR the translation names (s0..s105, VCC's words).
fn sgpr(r: u64) -> bool {
    r <= 107
}

/// The next instruction from `insts[*j]` on that does something; `*j` past it.
fn next<'a>(insts: &'a [Decoded], j: &mut usize) -> Result<&'a Decoded, &'static str> {
    while let Some(d) = insts.get(*j) {
        *j += 1;
        if !skipped(d) {
            return Ok(d);
        }
    }
    Err("the program ends")
}

/// The whole-wave section that starts at `insts[i]` as a reduction, or why it is not one
/// this recognises. `targets`: every branch target of the program.
pub fn at(insts: &[Decoded], i: usize, targets: &BTreeSet<u64>) -> Result<Reduction, &'static str> {
    let fv = |d: &Decoded, f: &str| rdna2::field_value(d.enc, &d.words, f).unwrap_or(0);
    let ext = |d: &Decoded, f: &str| ext_encoding(d).and_then(|e| rdna2::field_value(e, &d.words, f)).unwrap_or(0);
    // a VOP3 instruction without source / output modifiers
    let plain3 = |d: &Decoded| d.enc.name == "ENC_VOP3" && d.variant == Variant::Plain && ["ABS", "NEG", "CLAMP", "OMOD"].iter().all(|f| fv(d, f) == 0);
    let d0 = insts.get(i).ok_or("no instruction")?;
    let wave64 = match d0.inst.name {
        "S_OR_SAVEEXEC_B64" if fv(d0, "SSRC0") == 193 => true,
        "S_OR_SAVEEXEC_B32" if fv(d0, "SSRC0") == 193 => false,
        _ => return Err("not s_or_saveexec s, -1"),
    };
    let saved = fv(d0, "SDST");
    if !mask_register(saved, wave64) {
        return Err("EXEC saved to an unusual register");
    }
    let mut j = i + 1;
    let mut step = || next(insts, &mut j);
    let mut d = step()?;
    let and = if wave64 { "S_AND_B64" } else { "S_AND_B32" };
    let mask = if d.inst.name == and {
        if fv(d, "SSRC0") != saved && fv(d, "SSRC1") != saved {
            return Err("s_and without the saved EXEC");
        }
        let m = fv(d, "SDST");
        if !mask_register(m, wave64) {
            return Err("the mask in an unusual register");
        }
        d = step()?;
        m
    } else {
        saved
    };
    // x in the mask's lanes, a constant elsewhere
    if d.inst.name != "V_CNDMASK_B32" || !plain3(d) || fv(d, "SRC2") != mask {
        return Err("no v_cndmask_b32 of the mask");
    }
    let id = constant(fv(d, "SRC0"), d.literal).ok_or("v_cndmask_b32: no constant outside the mask")?;
    let x = fv(d, "SRC1");
    if !matches!(x, 0..=107 | 128..=208 | 240..=248 | 255 | 256..=511) {
        return Err("v_cndmask_b32: an unusual source");
    }
    let (acc, at, literal) = (fv(d, "VDST"), d.pc, d.literal);
    // the row: DPP8 steps, then row_mirror. The op is idempotent, so each lane holds the op
    // over the lanes it has read so far (bit l of `read[lane]`); every lane must end with its
    // whole row – whichever DPP8 selects do it (the pack's read one lane of the other pair,
    // then of the other quad, of lanes that hold the same by then).
    let mut read: [u16; 16] = std::array::from_fn(|l| 1 << l);
    let mut op = None;
    let mut d = step()?;
    while d.variant == Variant::Dpp8 {
        let o = Op::of(d.inst.name).ok_or("the steps are not v_min_u32 / v_max_u32 / v_or_b32")?;
        if *op.get_or_insert(o) != o {
            return Err("steps of different ops");
        }
        if d.enc.name != "ENC_VOP2" || fv(d, "VDST") != acc || ext(d, "VSRC0") != acc || fv(d, "VSRC1") != acc {
            return Err("a DPP8 step of another register");
        }
        // every lane is active: FI does not matter
        let sel: [usize; 8] = std::array::from_fn(|k| ext(d, &format!("LANE_SEL_{k}")) as usize);
        read = std::array::from_fn(|l| read[l] | read[(l & 8) | sel[l & 7]]);
        d = step()?;
    }
    let op = op.ok_or("no DPP8 step after the v_cndmask_b32")?;
    if id != op.identity() {
        return Err("the constant outside the mask is not the op's identity");
    }
    if Op::of(d.inst.name) != Some(op) || d.variant != Variant::Dpp16 || d.enc.name != "ENC_VOP2" {
        return Err("no row_mirror step");
    }
    let mods = ["SRC0_ABS", "SRC0_NEG", "SRC1_ABS", "SRC1_NEG"].iter().any(|f| ext(d, f) != 0);
    if ext(d, "DPP_CTRL") != 0x140 || ext(d, "ROW_MASK") != 0xF || ext(d, "BANK_MASK") != 0xF || mods {
        return Err("a DPP16 step other than row_mirror of all lanes");
    }
    if fv(d, "VDST") != acc || ext(d, "VSRC0") != acc || fv(d, "VSRC1") != acc {
        return Err("the row_mirror step of another register");
    }
    read = std::array::from_fn(|l| read[l] | read[15 - l]);
    if read.iter().any(|&r| r != 0xFFFF) {
        return Err("the DPP8 steps and row_mirror leave out lanes of the row");
    }
    // the other row of the half: every lane of it holds the row's value, so whichever lane
    // the selects pick (and FI / bound_ctrl only matter for inactive lanes: none here)
    let d = step()?;
    if d.inst.name != "V_PERMLANEX16_B32" || !plain3(d) || fv(d, "SRC0") != 256 + acc || fv(d, "SRC1") >= 256 || fv(d, "SRC2") >= 256 {
        return Err("no v_permlanex16_b32 of the row's value");
    }
    let other = fv(d, "VDST");
    if other == acc {
        return Err("v_permlanex16_b32 into the steps' register");
    }
    // the half
    let d = step()?;
    let (s0, s1) = (fv(d, "SRC0"), fv(d, "VSRC1"));
    if Op::of(d.inst.name) != Some(op) || d.enc.name != "ENC_VOP2" || d.variant != Variant::Plain || fv(d, "VDST") != acc || !((s0 == 256 + other && s1 == acc) || (s0 == 256 + acc && s1 == other)) {
        return Err("no op of the two rows");
    }
    // wave64: the upper half's, then the wave's in lanes 0..31
    let mut d = step()?;
    let half = if wave64 {
        if d.inst.name != "V_READLANE_B32" || !plain3(d) || fv(d, "SRC0") != 256 + acc || fv(d, "SRC1") != 128 + 32 || !sgpr(fv(d, "VDST")) {
            return Err("no v_readlane_b32 of lane 32");
        }
        let half = fv(d, "VDST");
        let d = step()?;
        if Op::of(d.inst.name) != Some(op) || d.enc.name != "ENC_VOP2" || d.variant != Variant::Plain || fv(d, "VDST") != acc || fv(d, "SRC0") != half || fv(d, "VSRC1") != acc {
            return Err("no op of the upper half's value");
        }
        Some(half)
    } else {
        None
    };
    if wave64 {
        d = step()?;
    }
    if d.inst.name != "V_READLANE_B32" || !plain3(d) || fv(d, "SRC0") != 256 + acc || fv(d, "SRC1") != 128 || !sgpr(fv(d, "VDST")) {
        return Err("no v_readlane_b32 of lane 0");
    }
    let total = fv(d, "VDST");
    let last = d.pc;
    let d = step()?;
    let restore = if wave64 { "S_MOV_B64" } else { "S_MOV_B32" };
    if d.inst.name != restore || fv(d, "SDST") != 126 || fv(d, "SSRC0") != saved {
        return Err("EXEC not restored after the v_readlane_b32 of lane 0");
    }
    // entered only at the top: no branch lands after the s_or_saveexec
    if targets.range(d0.pc + 1..=d.pc).next().is_some() {
        return Err("a branch into the sequence");
    }
    let rest = insts.iter().map(|d| d.pc).filter(|&pc| pc > at && pc <= last).collect();
    Ok(Reduction { op, wave64, at, rest, x, literal, mask, acc, other, half, total })
}

/// Every branch target of a program.
fn targets(insts: &[Decoded]) -> BTreeSet<u64> {
    insts.iter().filter_map(rdna2::branch_target).collect()
}

/// Every whole-wave section (`s_or_saveexec_b64` / `_b32` with -1) of a program: its pc, and
/// the reduction or why it is not one this recognises (the census).
pub fn survey(insts: &[Decoded]) -> Vec<(u64, Result<Reduction, &'static str>)> {
    let t = targets(insts);
    insts
        .iter()
        .enumerate()
        .filter(|(_, d)| matches!(d.inst.name, "S_OR_SAVEEXEC_B64" | "S_OR_SAVEEXEC_B32") && rdna2::field_value(d.enc, &d.words, "SSRC0") == Some(193))
        .map(|(i, d)| (d.pc, at(insts, i, &t)))
        .collect()
}

/// The reductions of a program (instructions in address order, far jumps resolved) in the
/// forms of its wave size.
pub fn find(insts: &[Decoded], wave64: bool) -> Vec<Reduction> {
    survey(insts).into_iter().filter_map(|(_, r)| r.ok()).filter(|r| r.wave64 == wave64).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::rdna2::encode;

    const V: u64 = 256;
    type F = Vec<(&'static str, u64)>;
    type I = (&'static str, &'static str, F, Option<u32>);

    fn program(lines: &[I]) -> Vec<Decoded> {
        let mut bytes: Vec<u8> = lines.iter().flat_map(|(n, e, f, l)| encode(n, e, f, *l).unwrap_or_else(|x| panic!("{n}: {x}"))).collect();
        bytes.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
        let mut out = Vec::new();
        let mut off = 0usize;
        while off < bytes.len() {
            let d = rdna2::decode(&bytes[off..], off as u64).unwrap();
            off += d.len;
            out.push(d);
        }
        rdna2::resolve_far_jumps(&mut out);
        out
    }

    fn i(n: &'static str, e: &'static str, f: &[(&'static str, u64)]) -> I {
        (n, e, f.to_vec(), None)
    }

    /// A DPP8 step of op on v6: lane l reads lane `sel[l]` of its 8.
    fn dpp8(op: &'static str, sel: [u64; 8]) -> I {
        let mut f = vec![("VDST", 6), ("SRC0", 0xE9), ("VSRC0", 6), ("VSRC1", 6)];
        let names = ["LANE_SEL_0", "LANE_SEL_1", "LANE_SEL_2", "LANE_SEL_3", "LANE_SEL_4", "LANE_SEL_5", "LANE_SEL_6", "LANE_SEL_7"];
        f.extend(names.iter().copied().zip(sel));
        (op, "VOP2_VOP_DPP8", f, None)
    }

    /// lane ^ k
    fn xor(k: u64) -> [u64; 8] {
        std::array::from_fn(|l| l as u64 ^ k)
    }

    /// The pack's DPP8 steps: the other lane of the pair, a lane of the other pair, of the
    /// other quad
    const PACK: [[u64; 8]; 3] = [[1, 0, 3, 2, 5, 4, 7, 6], [2, 2, 0, 0, 6, 6, 4, 4], [4, 4, 4, 4, 0, 0, 0, 0]];

    /// The pack's sequence: op (by name) over v55 in the lanes of s[18:19] & EXEC, identity
    /// operand code `id`, the upper half's into s10 and the wave's into s61.
    fn retail(op: &'static str, id: u64) -> Vec<I> {
        vec![
            i("S_OR_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 30), ("SSRC0", 193)]),
            i("S_AND_B64", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 18), ("SSRC1", 30)]),
            i("V_CNDMASK_B32", "ENC_VOP3", &[("VDST", 6), ("SRC0", id), ("SRC1", V + 55), ("SRC2", 10)]),
            dpp8(op, PACK[0]),
            dpp8(op, PACK[1]),
            dpp8(op, PACK[2]),
            i(op, "VOP2_VOP_DPP16", &[("VDST", 6), ("SRC0", 0xFA), ("VSRC0", 6), ("VSRC1", 6), ("DPP_CTRL", 0x140), ("ROW_MASK", 0xF), ("BANK_MASK", 0xF)]),
            i("V_PERMLANEX16_B32", "ENC_VOP3", &[("VDST", 7), ("SRC0", V + 6), ("SRC1", 128), ("SRC2", 128)]),
            i(op, "ENC_VOP2", &[("VDST", 6), ("SRC0", V + 7), ("VSRC1", 6)]),
            i("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 6), ("SRC1", 128 + 32)]),
            i(op, "ENC_VOP2", &[("VDST", 6), ("SRC0", 10), ("VSRC1", 6)]),
            i("V_READLANE_B32", "ENC_VOP3", &[("VDST", 61), ("SRC0", V + 6), ("SRC1", 128)]),
            i("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 30)]),
        ]
    }

    fn survey_of(lines: &[I]) -> Vec<Result<Reduction, &'static str>> {
        survey(&program(lines)).into_iter().map(|(_, r)| r).collect()
    }

    fn reason(lines: &[I]) -> &'static str {
        let s = survey_of(lines);
        assert_eq!(s.len(), 1);
        s[0].clone().expect_err("recognised")
    }

    #[test]
    fn the_retail_sequences_are_recognised() {
        for (op, id, want) in [("V_MIN_U32", 193, Op::MinU32), ("V_MAX_U32", 128, Op::MaxU32), ("V_OR_B32", 128, Op::OrB32)] {
            let p = program(&retail(op, id));
            let r = find(&p, true);
            assert_eq!(r.len(), 1, "{op}");
            let r = &r[0];
            assert_eq!((r.op, r.x, r.mask, r.acc, r.other, r.half, r.total), (want, V + 55, 10, 6, 7, Some(10), 61));
            // emitted at the v_cndmask, nothing for the steps up to the last v_readlane
            assert_eq!(r.at, p[2].pc);
            assert_eq!(r.rest, p[3..12].iter().map(|d| d.pc).collect::<Vec<_>>());
        }
    }

    #[test]
    fn variants_of_the_pack() {
        // the mask is the saved EXEC itself (16 sites in the pack)
        let mut l = retail("V_OR_B32", 128);
        l.remove(1);
        l[1].2[3] = ("SRC2", 30);
        let r = find(&program(&l), true);
        assert_eq!(r.iter().map(|r| r.mask).collect::<Vec<_>>(), [30]);
        // s_waitcnt / v_nop between the steps; the saved EXEC as s_and_b64's first operand;
        // the halves' op with its operands swapped; both v_readlane into one register
        let mut l = retail("V_MAX_U32", 128);
        l.insert(1, i("V_NOP", "ENC_VOP1", &[]));
        l.insert(3, i("S_WAITCNT", "ENC_SOPP", &[("SIMM16", 0x3F70)]));
        l[2] = i("S_AND_B64", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 30), ("SSRC1", 18)]);
        l[10] = i("V_MAX_U32", "ENC_VOP2", &[("VDST", 6), ("SRC0", V + 6), ("VSRC1", 7)]);
        l[13].2[0] = ("VDST", 10);
        let r = find(&program(&l), true);
        assert_eq!(r.iter().map(|r| (r.op, r.half, r.total)).collect::<Vec<_>>(), [(Op::MaxU32, Some(10), 10)]);
        // other DPP8 steps that take in the 8 lanes: lane ^ 4, ^ 1, ^ 2; ^ 1, ^ 3, ^ 7; four
        // steps, one of them reading the lane itself
        for steps in [vec![xor(4), xor(1), xor(2)], vec![xor(1), xor(3), xor(7)], vec![xor(1), xor(0), xor(2), xor(4)]] {
            let mut l = retail("V_MIN_U32", 193);
            l.splice(3..6, steps.iter().map(|&sel| dpp8("V_MIN_U32", sel)));
            assert_eq!(find(&program(&l), true).len(), 1, "{steps:?}");
        }
    }

    #[test]
    fn anything_else_is_left_alone() {
        let base = retail("V_MIN_U32", 193);
        let with = |k: usize, d: I| {
            let mut l = base.clone();
            l[k] = d;
            l
        };
        let short = "the DPP8 steps and row_mirror leave out lanes of the row";
        // not the identity outside the mask: max's 0 for a min
        assert_eq!(reason(&with(2, i("V_CNDMASK_B32", "ENC_VOP3", &[("VDST", 6), ("SRC0", 128), ("SRC1", V + 55), ("SRC2", 10)]))), "the constant outside the mask is not the op's identity");
        // a mask not within the saved EXEC: lanes the PC did not launch could be in it
        assert_eq!(reason(&with(1, i("S_AND_B64", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 18), ("SSRC1", 20)]))), "s_and without the saved EXEC");
        // the v_cndmask selects by another register than the mask
        assert_eq!(reason(&with(2, i("V_CNDMASK_B32", "ENC_VOP3", &[("VDST", 6), ("SRC0", 193), ("SRC1", V + 55), ("SRC2", 18)]))), "no v_cndmask_b32 of the mask");
        // a step of another op
        assert_eq!(reason(&with(4, dpp8("V_MAX_U32", PACK[1]))), "steps of different ops");
        // steps that leave lanes out: the pack's in another order (they read one lane of a
        // pair or quad, which holds the same as the other only after the steps before), the
        // other lane of the pair twice, a rotation, two steps
        let mut l = base.clone();
        l.swap(3, 5);
        assert_eq!(reason(&l), short);
        assert_eq!(reason(&with(4, dpp8("V_MIN_U32", PACK[0]))), short);
        assert_eq!(reason(&with(4, dpp8("V_MIN_U32", [1, 2, 3, 4, 5, 6, 7, 0]))), short);
        let mut l = base.clone();
        l.remove(5);
        assert_eq!(reason(&l), short);
        // no DPP8 step at all; a shift or a row mask instead of the whole mirror
        let mut l = base.clone();
        l.drain(3..6);
        assert_eq!(reason(&l), "no DPP8 step after the v_cndmask_b32");
        let shr = with(6, i("V_MIN_U32", "VOP2_VOP_DPP16", &[("VDST", 6), ("SRC0", 0xFA), ("VSRC0", 6), ("VSRC1", 6), ("DPP_CTRL", 0x111), ("ROW_MASK", 0xF), ("BANK_MASK", 0xF)]));
        assert_eq!(reason(&shr), "a DPP16 step other than row_mirror of all lanes");
        let rows = with(6, i("V_MIN_U32", "VOP2_VOP_DPP16", &[("VDST", 6), ("SRC0", 0xFA), ("VSRC0", 6), ("VSRC1", 6), ("DPP_CTRL", 0x140), ("ROW_MASK", 0x5), ("BANK_MASK", 0xF)]));
        assert_eq!(reason(&rows), "a DPP16 step other than row_mirror of all lanes");
        // v_permlane16 (same row) instead of v_permlanex16
        assert_eq!(reason(&with(7, i("V_PERMLANE16_B32", "ENC_VOP3", &[("VDST", 7), ("SRC0", V + 6), ("SRC1", 128), ("SRC2", 128)]))), "no v_permlanex16_b32 of the row's value");
        // another lane than 32, then than 0
        assert_eq!(reason(&with(9, i("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 6), ("SRC1", 128 + 16)]))), "no v_readlane_b32 of lane 32");
        assert_eq!(reason(&with(11, i("V_READLANE_B32", "ENC_VOP3", &[("VDST", 61), ("SRC0", V + 6), ("SRC1", 128 + 1)]))), "no v_readlane_b32 of lane 0");
        // the halves' op reads another register than the readlane's
        assert_eq!(reason(&with(10, i("V_MIN_U32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 11), ("VSRC1", 6)]))), "no op of the upper half's value");
        // something else under the whole wave before the restore
        let mut l = base.clone();
        l.insert(12, i("V_MOV_B32", "ENC_VOP1", &[("VDST", 8), ("SRC0", V + 6)]));
        assert_eq!(reason(&l), "EXEC not restored after the v_readlane_b32 of lane 0");
        // a branch into the middle (to the third DPP8 step)
        let mut l = vec![i("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 8)])];
        l.extend(base.iter().cloned());
        assert_eq!(reason(&l), "a branch into the sequence");
        // the branch to the s_or_saveexec itself is fine
        let mut l = vec![i("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 0)])];
        l.extend(base.iter().cloned());
        assert_eq!(find(&program(&l), true).len(), 1);
        // the 64-bit forms in a wave32 program (`find` by the program's wave size)
        assert!(find(&program(&base), false).is_empty());
    }

    /// The pack's wave32 sequence: the 32-bit forms, no v_readlane of lane 32.
    fn retail32(op: &'static str, id: u64) -> Vec<I> {
        let mut l = retail(op, id);
        l[0] = i("S_OR_SAVEEXEC_B32", "ENC_SOP1", &[("SDST", 30), ("SSRC0", 193)]);
        l[1] = i("S_AND_B32", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 18), ("SSRC1", 30)]);
        l.drain(9..11);
        l[10] = i("S_MOV_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 30)]);
        l
    }

    #[test]
    fn the_wave32_sequences_are_recognised() {
        for (op, id, want) in [("V_MIN_U32", 193, Op::MinU32), ("V_MAX_U32", 128, Op::MaxU32), ("V_OR_B32", 128, Op::OrB32)] {
            let p = program(&retail32(op, id));
            assert!(find(&p, true).is_empty(), "{op}: wave32 forms in a wave64 program");
            let r = find(&p, false);
            assert_eq!(r.len(), 1, "{op}");
            let r = &r[0];
            assert_eq!((r.op, r.wave64, r.x, r.mask, r.acc, r.other, r.half, r.total), (want, false, V + 55, 10, 6, 7, None, 61));
            assert_eq!(r.rest, p[3..10].iter().map(|d| d.pc).collect::<Vec<_>>());
        }
        // the mask is the saved EXEC itself
        let mut l = retail32("V_MIN_U32", 193);
        l.remove(1);
        l[1].2[3] = ("SRC2", 30);
        assert_eq!(find(&program(&l), false).iter().map(|r| r.mask).collect::<Vec<_>>(), [30]);
        // a wave64 restore, a v_readlane of lane 32 (the upper half's: no such lane)
        let mut l = retail32("V_MIN_U32", 193);
        l[10] = i("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 30)]);
        assert_eq!(reason(&l), "EXEC not restored after the v_readlane_b32 of lane 0");
        let mut l = retail32("V_MIN_U32", 193);
        l[9].2[2] = ("SRC1", 128 + 32);
        assert_eq!(reason(&l), "no v_readlane_b32 of lane 0");
    }
}
