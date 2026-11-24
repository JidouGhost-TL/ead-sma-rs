//! Sample descriptor parsing for the Nintendo EAD/SRD Super Mario Advance engine.

use crate::error::{Error, Result};

/// Size in bytes of an EAD/SRD sample descriptor header (`16` bytes).
pub const EAD_SAMPLE_DESC_SIZE: usize = 16;

/// A parsed 16-byte EAD/SRD sample descriptor header.
///
/// Layout (all fields little-endian `u32`), immediately followed by signed
/// 8-bit PCM data:
/// - word 0: total sample length in samples (the one-shot end index used
///   when word 3 is zero),
/// - word 1: rate word consumed by the sequencer when computing the Q8
///   phase step,
/// - word 2: loop start sample index,
/// - word 3: loop end sample index (`0` means one-shot; the sample ends at
///   word 0 instead).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EadSampleDesc {
    /// Total sample length in samples (one-shot end index).
    pub total_len: u32,
    /// Rate word used by the sequencer for Q8 step computation.
    pub rate_word: u32,
    /// Loop start sample index.
    pub loop_start: u32,
    /// Loop end sample index (`0` = one-shot).
    pub loop_end: u32,
}

impl EadSampleDesc {
    /// Parse a descriptor from the first [`EAD_SAMPLE_DESC_SIZE`] bytes of `bytes`.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < EAD_SAMPLE_DESC_SIZE {
            return Err(Error::UnexpectedEof);
        }
        let word = |i: usize| {
            u32::from_le_bytes([
                bytes[i * 4],
                bytes[i * 4 + 1],
                bytes[i * 4 + 2],
                bytes[i * 4 + 3],
            ])
        };
        Ok(Self {
            total_len: word(0),
            rate_word: word(1),
            loop_start: word(2),
            loop_end: word(3),
        })
    }

    /// Returns `true` when this sample loops (`loop_end != 0`).
    #[inline]
    #[must_use]
    pub const fn is_looped(self) -> bool {
        self.loop_end != 0
    }

    /// Effective end sample index: `loop_end` for looped samples, otherwise
    /// `total_len`.
    #[inline]
    #[must_use]
    pub const fn end_index(self) -> u32 {
        if self.loop_end != 0 {
            self.loop_end
        } else {
            self.total_len
        }
    }

    /// Length of the loop in samples (`loop_end - loop_start`), or `0` for
    /// one-shot samples.
    #[inline]
    #[must_use]
    pub const fn loop_len(self) -> u32 {
        if self.loop_end != 0 {
            self.loop_end.saturating_sub(self.loop_start)
        } else {
            0
        }
    }
}

/// Zero-allocation view over an EAD/SRD sample: a parsed descriptor plus the
/// signed 8-bit PCM payload that follows it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EadSampleView<'a> {
    /// Parsed 16-byte descriptor header.
    pub desc: EadSampleDesc,
    /// Signed 8-bit PCM payload bytes (one byte per sample).
    pub pcm_bytes: &'a [u8],
}

impl<'a> EadSampleView<'a> {
    /// Split a buffer holding `[descriptor][PCM...]` into descriptor + PCM view.
    pub fn parse(data: &'a [u8]) -> Result<Self> {
        let desc = EadSampleDesc::parse(data)?;
        Ok(Self {
            desc,
            pcm_bytes: &data[EAD_SAMPLE_DESC_SIZE..],
        })
    }

    /// Construct a view directly from a descriptor and a PCM payload slice.
    #[inline]
    #[must_use]
    pub const fn from_parts(desc: EadSampleDesc, pcm_bytes: &'a [u8]) -> Self {
        Self { desc, pcm_bytes }
    }

    /// Read the signed sample at byte index `index` (`0` when out of bounds).
    #[inline]
    #[must_use]
    pub fn sample_i8(&self, index: usize) -> i8 {
        self.pcm_bytes.get(index).map_or(0, |&b| b as i8)
    }

    /// Returns `true` when the PCM payload is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pcm_bytes.is_empty()
    }
}

/// Owned EAD/SRD sample (requires the `alloc` feature).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EadSample {
    /// Parsed 16-byte descriptor header.
    pub desc: EadSampleDesc,
    /// Owned signed 8-bit PCM payload bytes.
    pub pcm: alloc::vec::Vec<u8>,
}

#[cfg(feature = "alloc")]
impl EadSample {
    /// Parse an owned sample from a `[descriptor][PCM...]` buffer.
    pub fn parse(data: &[u8]) -> Result<Self> {
        let view = EadSampleView::parse(data)?;
        Ok(Self {
            desc: view.desc,
            pcm: view.pcm_bytes.to_vec(),
        })
    }

    /// Borrow this sample as an [`EadSampleView`].
    #[inline]
    #[must_use]
    pub fn as_view(&self) -> EadSampleView<'_> {
        EadSampleView::from_parts(self.desc, &self.pcm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_parsing_works() {
        let mut buf = [0u8; 16 + 4];
        buf[0..4].copy_from_slice(&100u32.to_le_bytes());
        buf[4..8].copy_from_slice(&22_050u32.to_le_bytes());
        buf[8..12].copy_from_slice(&10u32.to_le_bytes());
        buf[12..16].copy_from_slice(&90u32.to_le_bytes());
        let view = EadSampleView::parse(&buf).unwrap();
        assert_eq!(view.desc.total_len, 100);
        assert_eq!(view.desc.rate_word, 22_050);
        assert!(view.desc.is_looped());
        assert_eq!(view.desc.end_index(), 90);
        assert_eq!(view.desc.loop_len(), 80);
        assert_eq!(view.pcm_bytes.len(), 4);
        assert_eq!(EadSampleDesc::parse(&buf[..8]), Err(Error::UnexpectedEof));
    }
}
