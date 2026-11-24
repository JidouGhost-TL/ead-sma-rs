//! Byte-exact ARM7TDMI golden tests for all three Nintendo EAD/SRD mixer
//! kernels across both hardware revisions ([`EadSmaVersion::Sma23Shift8`]
//! and [`EadSmaVersion::Sma4Mla`]).
//!
//! Every golden binary in `tests/golden/` was captured by executing the
//! authentic ARM7TDMI kernel code (copied unmodified from retail ROMs into
//! IWRAM by a synthetic test stub) inside headless `libmgba.so.0.10` on
//! clean-room synthetic deterministic waveforms. Every single output byte
//! is verified with strict `assert_eq!` (zero tolerance). No copyrighted
//! game audio is used anywhere.

use ead_sma::{AccuracySynth, EadSmaVersion};

fn synth_pcm() -> [u8; 64] {
    core::array::from_fn(|i| ((((i as i32) * 29 + 13) % 256) - 128) as i8 as u8)
}

fn init_acc_l() -> [i16; 32] {
    core::array::from_fn(|i| ((i as i32) * 37 - 700) as i16)
}

fn init_acc_r() -> [i16; 32] {
    core::array::from_fn(|i| ((i as i32) * 53 - 300) as i16)
}

fn i16s_to_le_bytes(slice: &[i16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(slice.len() * 2);
    for &v in slice {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn check_mix(
    golden: &[u8],
    version: EadSmaVersion,
    frames: usize,
    phase: u32,
    step: u32,
    vol_l: u32,
    vol_r: u32,
) {
    let pcm = synth_pcm();
    let mut out_l = init_acc_l();
    let mut out_r = init_acc_r();
    let new_phase = AccuracySynth::mix_kernel(
        version,
        &pcm,
        phase,
        step,
        vol_l,
        vol_r,
        &mut out_l[..frames],
        &mut out_r[..frames],
    );
    let mut expected = i16s_to_le_bytes(&out_l[..frames]);
    expected.extend_from_slice(&i16s_to_le_bytes(&out_r[..frames]));
    expected.extend_from_slice(&new_phase.to_le_bytes());
    assert_eq!(&expected[..], golden);
}

#[test]
fn golden_01_sma23_mix_32f_step256() {
    check_mix(
        include_bytes!("golden/sma23_mix_32f_step256.bin"),
        EadSmaVersion::Sma23Shift8,
        32,
        0,
        256,
        200,
        90,
    );
}

#[test]
fn golden_02_sma23_mix_30f_step128() {
    check_mix(
        include_bytes!("golden/sma23_mix_30f_step128.bin"),
        EadSmaVersion::Sma23Shift8,
        30,
        0x34,
        128,
        255,
        17,
    );
}

#[test]
fn golden_03_sma23_mix_32f_step640() {
    check_mix(
        include_bytes!("golden/sma23_mix_32f_step640.bin"),
        EadSmaVersion::Sma23Shift8,
        32,
        0x100,
        640,
        64,
        250,
    );
}

#[test]
fn golden_04_sma4_mix_32f_step256() {
    check_mix(
        include_bytes!("golden/sma4_mix_32f_step256.bin"),
        EadSmaVersion::Sma4Mla,
        32,
        0,
        256,
        200,
        90,
    );
}

#[test]
fn golden_05_sma4_mix_30f_step128() {
    check_mix(
        include_bytes!("golden/sma4_mix_30f_step128.bin"),
        EadSmaVersion::Sma4Mla,
        30,
        0x34,
        128,
        255,
        17,
    );
}

#[test]
fn golden_06_sma4_mix_32f_step640() {
    check_mix(
        include_bytes!("golden/sma4_mix_32f_step640.bin"),
        EadSmaVersion::Sma4Mla,
        32,
        0x100,
        640,
        64,
        250,
    );
}

const CONV_SRC: [i16; 32] = [
    0, 1, -1, 127, 128, -128, -129, 255, 256, -255, -256, 511, 512, -511, -512, 1000, -1000,
    32767, -32768, 16384, -16384, 8192, -8192, 100, -100, 42, -42, 16383, -16383, 4096, -4096, 77,
];

#[test]
fn golden_07_sma23_convert() {
    let golden = include_bytes!("golden/sma23_convert.bin");
    let mut dst = [0i8; 32];
    AccuracySynth::convert_to_dma8(EadSmaVersion::Sma23Shift8, &CONV_SRC, &mut dst);
    let got: Vec<u8> = dst.iter().map(|&b| b as u8).collect();
    assert_eq!(&got[..], &golden[..]);
}

#[test]
fn golden_08_sma4_convert() {
    let golden = include_bytes!("golden/sma4_convert.bin");
    let mut dst = [0i8; 32];
    AccuracySynth::convert_to_dma8(EadSmaVersion::Sma4Mla, &CONV_SRC, &mut dst);
    let got: Vec<u8> = dst.iter().map(|&b| b as u8).collect();
    assert_eq!(&got[..], &golden[..]);
}

fn check_combine(golden: &[u8], version: EadSmaVersion, shift: u32) {
    let mut main: [i16; 32] = core::array::from_fn(|i| ((i as i32) * 97 - 1500) as i16);
    let mut aux: [i16; 32] = core::array::from_fn(|i| ((i as i32) * 61 - 900) as i16);
    AccuracySynth::combine(version, &mut main, &mut aux, shift);
    let mut expected = i16s_to_le_bytes(&main);
    expected.extend_from_slice(&i16s_to_le_bytes(&aux));
    assert_eq!(&expected[..], golden);
}

#[test]
fn golden_09_sma23_combine_shift2() {
    check_combine(
        include_bytes!("golden/sma23_combine_shift2.bin"),
        EadSmaVersion::Sma23Shift8,
        2,
    );
}

#[test]
fn golden_10_sma23_combine_shift4() {
    check_combine(
        include_bytes!("golden/sma23_combine_shift4.bin"),
        EadSmaVersion::Sma23Shift8,
        4,
    );
}

#[test]
fn golden_11_sma4_combine_shift2() {
    check_combine(
        include_bytes!("golden/sma4_combine_shift2.bin"),
        EadSmaVersion::Sma4Mla,
        2,
    );
}

#[test]
fn golden_12_sma4_combine_shift4() {
    check_combine(
        include_bytes!("golden/sma4_combine_shift4.bin"),
        EadSmaVersion::Sma4Mla,
        4,
    );
}
