//! # ead-sma-rs
//!
//! Pure Rust, zero-dependency parser, software synthesizer, and byte-exact
//! `libmgba` ARM golden-tested emulator for the **Nintendo EAD / SRD custom
//! DirectSound sound engine** used by the *Super Mario Advance* series on the
//! Game Boy Advance.
//!
//! A full-corpus audit of all 3,377 GBA ROMs shows this engine is **not**
//! Nintendo's MusicPlayer2000 (M4A / Sappy): *Super Mario Advance 2–4* contain
//! no M4A `SoundMainRAM` mixer at all. Instead they share a proprietary
//! ARM7TDMI software mixer (copied from ROM into IWRAM at boot) with three
//! kernels — a nearest-neighbour voice mixer, an attenuating buffer combine,
//! and a saturating `i16 -> i8` DirectSound DMA converter — across **2 distinct
//! revisions** ([`EadSmaVersion::Sma23Shift8`] and [`EadSmaVersion::Sma4Mla`],
//! 35 ROM images in total including regional, revision, Virtual Console, and
//! China-market releases). The same engine also powers the GBA port of
//! *The Legend of Zelda: A Link to the Past & Four Swords* and the
//! *Mario Bros.* classic modes embedded in the *Super Mario Advance* carts.
//! By contrast, *Super Mario Advance* (the first game), *Mario Kart: Super
//! Circuit*, and the *F-Zero* GBA titles use M4A and are out of scope here.
//!
//! ## Features
//! - **Zero external dependencies** (`[dependencies]` is empty).
//! - **`#![no_std]` compatible** with `default = ["std"]`, `std = ["alloc"]`, `alloc = []`.
//! - **`#![deny(unsafe_code)]`**: 100% safe Rust.
//! - **Dual synthesis pipelines**:
//!   - [`AccuracySynth`]: Bit-exact emulation of both ARM7TDMI IWRAM mixer
//!     revisions (Q8 nearest-neighbour phase, per-revision volume scaling,
//!     loop wrap, attenuating combine, and saturating DMA8 conversion).
//!   - [`StudioSynth`]: 48,000 Hz `f32` high-fidelity renderer with a
//!     compile-time Lanczos-3 polyphase LUT (plus general Lanczos-`a`
//!     fallback) and equal-power stereo panning.
//!
//! ## Quickstart
//! ```rust
//! use ead_sma::{AccuracySynth, EadSampleView, EadSmaVersion, EadVoice, DMA_FRAME_SAMPLES};
//!
//! let mut sample_bytes = [0u8; 24];
//! sample_bytes[0..4].copy_from_slice(&8u32.to_le_bytes());
//! sample_bytes[12..16].copy_from_slice(&8u32.to_le_bytes());
//! sample_bytes[16..24].copy_from_slice(&[0, 32, 64, 96, 127, 96, 64, 32]);
//!
//! let sample = EadSampleView::parse(&sample_bytes).unwrap();
//! let mut voice = EadVoice::new(256, 200, 64);
//! let mut out_l = [0i16; DMA_FRAME_SAMPLES];
//! let mut out_r = [0i16; DMA_FRAME_SAMPLES];
//! AccuracySynth::mix_voice(EadSmaVersion::Sma4Mla, &sample, &mut voice, &mut out_l, &mut out_r);
//! assert!(out_l.iter().any(|&s| s != 0));
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unsafe_code)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod error;
pub mod freq;
pub mod sample;
pub mod synth;
pub mod version;
pub mod voice;

pub use error::{Error, Result};
pub use freq::{
    arm_sdiv, arm_udiv, freq_to_step_q8, pan_gains, rate_from_timer_reload, step_q8_to_freq,
    DMA_FRAME_SAMPLES, GBA_CLOCK_HZ, Q8_ONE,
};
#[cfg(feature = "alloc")]
pub use sample::EadSample;
pub use sample::{EadSampleDesc, EadSampleView, EAD_SAMPLE_DESC_SIZE};
pub use synth::{AccuracySynth, StudioSynth};
pub use version::EadSmaVersion;
pub use voice::EadVoice;
