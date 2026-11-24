//! Nintendo EAD/SRD Super Mario Advance engine revision detection.
//!
//! A full-corpus audit of all 3,377 GBA ROMs identifies **35 ROM images**
//! powered by the EAD/SRD custom DirectSound mixer across **2 distinct
//! ARM7TDMI IWRAM kernel revisions**:
//!
//! 1. [`EadSmaVersion::Sma23Shift8`] (**12 ROMs** — *Super Mario Advance 2*
//!    USA/Europe/Japan/China, *Super Mario Advance 3* USA/Europe/Japan in
//!    retail and Virtual Console forms, the China-market *Yoshi's Island*
//!    release, and a China-market *Mario & Luigi* prototype whose embedded
//!    *Mario Bros.* classic mode uses this driver):
//!    - Voice mixer accumulates `(sample * volume) >> 8` per output frame
//!      (ARM `mul` + `add ..., asr #8`, truncated to `i16` on store).
//!    - DMA converter stores the low byte of each `i16` accumulator,
//!      saturating to `[-128, 127]` only when the value lies outside that
//!      range (`(v & 0xFF80) != 0 && != 0xFF80`).
//!    - Buffer combine attenuates with round-half bias `1 << (shift - 1)`
//!      for **all** source values, negative or not.
//! 2. [`EadSmaVersion::Sma4Mla`] (**23 ROMs** — all *Super Mario Advance 4*
//!    regional/revision/Virtual Console/Switch Online images, the
//!    *Mario Bros.* classic mode inside *Super Mario Advance 3*, and
//!    *The Legend of Zelda: A Link to the Past & Four Swords* USA/Europe/Japan):
//!    - Voice mixer accumulates `sample * volume` with **no shift**
//!      (ARM `mla`), so its voice volumes are pre-scaled by 1/256 relative
//!      to the older revision.
//!    - DMA converter first computes `(v + (v < 0 ? 127 : 0)) >> 7`
//!      (round-toward-zero division by 128), then applies the same
//!      `[-128, 127]` saturation test to the shifted value.
//!    - Buffer combine attenuates with bias `(1 << shift) - 1` applied
//!      **only to negative** source values (ceil-style rounding).
//!
//! Both revisions share the same voice-render wrapper semantics (Q8 phase,
//! `176`-sample DMA frames, planar left/right `i16` mix buffers), the same
//! 16-byte sample descriptor layout, and identical ARM unsigned/signed
//! division helpers. *Super Mario Advance 3* cartridges contain **both**
//! revisions (the main game uses `Sma23Shift8`; the *Mario Bros.* classic
//! mode uses `Sma4Mla`); [`EadSmaVersion::detect_from_bytes`] reports the
//! main-game revision in that case.

/// Distinct Nintendo EAD/SRD ARM7TDMI mixer revisions discovered across the GBA corpus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum EadSmaVersion {
    /// SMA2/SMA3 revision (12 ROMs): mixer accumulates `(sample * volume) >> 8`;
    /// DMA converter saturates the raw `i16` accumulator; combine uses the
    /// unconditional round-half bias `1 << (shift - 1)`.
    #[default]
    Sma23Shift8,
    /// SMA4 revision (23 ROMs): mixer accumulates `sample * volume` via `mla`
    /// with no shift; DMA converter pre-scales by `>> 7` with negative bias
    /// `+127` before saturating; combine biases only negative values by
    /// `(1 << shift) - 1`.
    Sma4Mla,
}

impl EadSmaVersion {
    /// Returns `true` when the voice mixer shifts each product right by 8
    /// before accumulating (`Sma23Shift8`), or `false` when it accumulates
    /// the full product (`Sma4Mla`).
    #[inline]
    #[must_use]
    pub const fn mixer_shifts_product(self) -> bool {
        match self {
            Self::Sma23Shift8 => true,
            Self::Sma4Mla => false,
        }
    }

    /// Number of GBA ROM images in the audited corpus using this revision.
    #[inline]
    #[must_use]
    pub const fn corpus_rom_count(self) -> usize {
        match self {
            Self::Sma23Shift8 => 12,
            Self::Sma4Mla => 23,
        }
    }

    /// Detect the exact Nintendo EAD/SRD mixer revision from a GBA ROM image.
    ///
    /// Matches the three ARM7TDMI IWRAM kernel signatures (DMA converter,
    /// voice mixer, buffer combine) at their revision-specific relative
    /// offsets. The kernels are copied verbatim from ROM into IWRAM by the
    /// retail driver, so the ROM signatures are byte-identical to the
    /// executing code. Returns `None` for ROMs that do not contain this
    /// engine (including all M4A titles such as the first
    /// *Super Mario Advance*, *Mario Kart: Super Circuit*, and *F-Zero*).
    #[must_use]
    pub fn detect_from_bytes(rom: &[u8]) -> Option<Self> {
        // Sma23Shift8 kernel block: [converter][mixer @ +0xC0][combine @ +0x1E0]
        const A_CONV: [u8; 24] = [
            0x30, 0x00, 0x2D, 0xE9, 0x7F, 0x40, 0xA0, 0xE3, 0xFF, 0x5C, 0xA0, 0xE3, 0x80, 0x50,
            0x85, 0xE3, 0xF2, 0x30, 0xD0, 0xE0, 0x05, 0xC0, 0x13, 0xE0,
        ];
        const A_MIX: [u8; 24] = [
            0x0D, 0xC0, 0xA0, 0xE1, 0xF0, 0x03, 0x2D, 0xE9, 0xF0, 0x00, 0x9C, 0xE8, 0x01, 0xC0,
            0x43, 0xE0, 0x07, 0xC0, 0x0C, 0xE2, 0x00, 0x00, 0x5C, 0xE3,
        ];
        const A_COMB: [u8; 24] = [
            0x30, 0x00, 0x2D, 0xE9, 0x01, 0x40, 0xA0, 0xE3, 0x01, 0xC0, 0x43, 0xE2, 0x14, 0x4C,
            0xA0, 0xE1, 0xF0, 0x50, 0xD0, 0xE1, 0xF0, 0xC0, 0xD1, 0xE1,
        ];
        // Sma4Mla kernel block: [converter][mixer @ +0x120][combine @ +0x218]
        const B_CONV: [u8; 24] = [
            0x30, 0x00, 0x2D, 0xE9, 0x7F, 0x40, 0xA0, 0xE3, 0xFF, 0x5C, 0xA0, 0xE3, 0x80, 0x50,
            0x85, 0xE3, 0xF2, 0x30, 0xD0, 0xE0, 0x00, 0x00, 0x53, 0xE3,
        ];
        const B_MIX: [u8; 24] = [
            0x0D, 0xC0, 0xA0, 0xE1, 0xF0, 0x01, 0x2D, 0xE9, 0xF0, 0x00, 0x9C, 0xE8, 0x01, 0xC0,
            0x43, 0xE0, 0x07, 0xC0, 0x0C, 0xE2, 0x00, 0x00, 0x5C, 0xE3,
        ];
        const B_COMB: [u8; 24] = [
            0x30, 0x00, 0x2D, 0xE9, 0x01, 0x40, 0xA0, 0xE3, 0x14, 0x43, 0xA0, 0xE1, 0x01, 0x40,
            0x44, 0xE2, 0xF0, 0x50, 0xD0, 0xE1, 0xF0, 0xC0, 0xD1, 0xE1,
        ];

        if find_block(rom, &A_CONV, &A_MIX, 0xC0, &A_COMB, 0x1E0) {
            return Some(Self::Sma23Shift8);
        }
        if find_block(rom, &B_CONV, &B_MIX, 0x120, &B_COMB, 0x218) {
            return Some(Self::Sma4Mla);
        }
        None
    }
}

/// Returns `true` when `conv`, `mix` (at `mix_delta`), and `comb` (at
/// `comb_delta`) signatures all occur at consistent relative offsets.
fn find_block(rom: &[u8], conv: &[u8; 24], mix: &[u8; 24], mix_delta: usize, comb: &[u8; 24], comb_delta: usize) -> bool {
    let mut start = 0usize;
    while let Some(rel) = find_subslice(&rom[start..], conv) {
        let base = start + rel;
        if base + comb_delta + comb.len() <= rom.len()
            && rom[base + mix_delta..base + mix_delta + mix.len()] == mix[..]
            && rom[base + comb_delta..base + comb_delta + comb.len()] == comb[..]
        {
            return true;
        }
        start = base + 4;
    }
    false
}

/// Find the first occurrence of `needle` in `haystack`.
#[inline]
fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn revision_metadata_is_consistent() {
        assert!(EadSmaVersion::Sma23Shift8.mixer_shifts_product());
        assert!(!EadSmaVersion::Sma4Mla.mixer_shifts_product());
        assert_eq!(EadSmaVersion::Sma23Shift8.corpus_rom_count(), 12);
        assert_eq!(EadSmaVersion::Sma4Mla.corpus_rom_count(), 23);
        assert_eq!(EadSmaVersion::detect_from_bytes(&[]), None);
        assert_eq!(EadSmaVersion::detect_from_bytes(&[0u8; 4096]), None);
    }
}
