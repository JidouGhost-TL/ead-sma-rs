# ead-sma-rs

Pure Rust, zero-dependency parser, software synthesizer, and byte-exact
`libmgba` ARM golden-tested emulator for the **Nintendo EAD / SRD custom
DirectSound sound engine** used by the *Super Mario Advance* series on the
Game Boy Advance.

> **gbakit callout:** For full-corpus audio scanning, export, and SoundFont
> tooling across every GBA sound engine, see
> [gbakit](https://github.com/JidouGhost-TL/gbakit)
> (`gbakit audio scan`, `gbakit audio export`, `gbakit audio soundfont`).

## Corpus audit (3,377 GBA ROMs)

This engine is **not** Nintendo's MusicPlayer2000 (M4A / Sappy).
*Super Mario Advance 2–4* contain no M4A `SoundMainRAM` mixer; they share a
proprietary ARM7TDMI software mixer copied from ROM into IWRAM at boot,
with three kernels (nearest-neighbour voice mixer, attenuating buffer
combine, saturating `i16 -> i8` DMA converter) in **2 distinct revisions**
covering **35 ROM images**:

| Revision | ROMs | Titles |
|---|---:|---|
| `Sma23Shift8` | 12 | *Super Mario Advance 2* (USA/EU/JP/CN), *Super Mario Advance 3* (USA/EU/JP + Virtual Console, CN), *Mario & Luigi* (CN proto, classic mode) |
| `Sma4Mla` | 23 | *Super Mario Advance 4* (all regions/revisions/VC/Switch Online), *Super Mario Advance 3* classic mode, *Zelda: A Link to the Past & Four Swords* (USA/EU/JP) |

*Super Mario Advance 3* cartridges contain both revisions (main game:
`Sma23Shift8`; *Mario Bros.* classic mode: `Sma4Mla`). The first
*Super Mario Advance*, *Mario Kart: Super Circuit*, and the *F-Zero* GBA
titles use M4A instead and are out of scope for this crate.

### ARM differences between revisions

| Kernel | `Sma23Shift8` | `Sma4Mla` |
|---|---|---|
| Voice mixer | `acc += (s * vol) >> 8` | `acc += s * vol` (`mla`, no shift) |
| DMA converter | saturate raw `i16` to `[-128, 127]` | `(v + (v<0 ? 127 : 0)) >> 7`, then saturate |
| Buffer combine | bias `1 << (shift-1)` for all values | bias `(1<<shift)-1` only for negative values |

Both revisions share Q8 nearest-neighbour phase (`phase >> 8` sample
index, no interpolation), 176-sample DMA frames, planar `i16` left/right
mix buffers, the 16-byte sample descriptor, and ARM division helpers
(divide-by-zero returns 0).

## Features

- **Zero external dependencies**, `#![no_std]` compatible
  (`default = ["std"]`, `std = ["alloc"]`, `alloc = []`), and
  `#![deny(unsafe_code)]` (100% safe Rust).
- `AccuracySynth`: bit-exact emulation of both ARM revisions — voice
  mixing with loop wrap and one-shot termination, buffer combine, and
  saturating DMA8 conversion.
- `StudioSynth`: 48,000 Hz `f32` renderer with a compile-time
  L1-resident Lanczos-3 polyphase LUT (1,024 phases, normalized per
  phase; general Lanczos-`a` fallback) and equal-power stereo panning.
- `EadSmaVersion::detect_from_bytes`: revision detection from ROM bytes
  with zero false positives across the corpus (M4A titles return `None`).

## Golden test methodology

Every revision is verified byte-for-byte (`assert_eq!`, zero tolerance)
against real ARM7TDMI execution inside headless `libmgba.so.0.10`: the
unmodified kernel machine code from representative retail ROMs is driven
with clean-room synthetic waveforms (deterministic ramps and boundary
extrema) by a synthetic test stub, and the exact output bytes plus the
returned phase are captured under `tests/golden/` (12 vectors, 1,352
bytes, SHA-256 manifest in `tests/golden/manifest.json`). No copyrighted
game audio is used or committed.

## Usage

```rust
use ead_sma::{AccuracySynth, EadSampleView, EadSmaVersion, EadVoice};

let version = EadSmaVersion::detect_from_bytes(rom_bytes);
let sample = EadSampleView::parse(sample_bytes).unwrap();
let mut voice = EadVoice::new(256, 200, 64);
let mut out_l = [0i16; 176];
let mut out_r = [0i16; 176];
AccuracySynth::mix_voice(version.unwrap(), &sample, &mut voice, &mut out_l, &mut out_r);
```

## License

Dual-licensed under MIT OR Apache-2.0.
