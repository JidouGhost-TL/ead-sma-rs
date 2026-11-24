//! Voice state mirroring the retail driver's IWRAM channel layout.

/// Playback state of a single EAD/SRD voice.
///
/// Mirrors the fields of the retail channel struct consumed by the Thumb
/// voice wrapper: the sample descriptor pointer (modelled here by
/// [`crate::EadSampleView`] at call time), the Q8 phase accumulator stored
/// at channel offset `+0x60`, the envelope volume, and the pan byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EadVoice {
    /// Q8 phase accumulator (`phase >> 8` is the current sample index).
    pub phase_q8: u32,
    /// Q8 phase increment applied per output frame.
    pub step_q8: u32,
    /// 16-bit envelope/master volume consumed by [`crate::pan_gains`].
    pub volume: u16,
    /// Pan byte (`0` = hard left, `127` = hard right in normal use).
    pub pan: u8,
    /// Whether the voice is still sounding (`false` after a one-shot ends).
    pub active: bool,
}

impl EadVoice {
    /// Create a new active voice at phase `0` with the given Q8 `step_q8`,
    /// `volume`, and `pan`.
    #[inline]
    #[must_use]
    pub const fn new(step_q8: u32, volume: u16, pan: u8) -> Self {
        Self {
            phase_q8: 0,
            step_q8,
            volume,
            pan,
            active: true,
        }
    }

    /// Create a new active voice starting at an explicit Q8 `phase_q8`.
    #[inline]
    #[must_use]
    pub const fn with_phase(phase_q8: u32, step_q8: u32, volume: u16, pan: u8) -> Self {
        Self {
            phase_q8,
            step_q8,
            volume,
            pan,
            active: true,
        }
    }

    /// Current integer sample index (`phase_q8 >> 8`).
    #[inline]
    #[must_use]
    pub const fn sample_index(&self) -> u32 {
        self.phase_q8 >> 8
    }

    /// Silence this voice (one-shot termination).
    #[inline]
    pub fn stop(&mut self) {
        self.active = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voice_state_works() {
        let mut voice = EadVoice::with_phase(0x1234, 256, 200, 64);
        assert!(voice.active);
        assert_eq!(voice.sample_index(), 0x12);
        voice.stop();
        assert!(!voice.active);
    }
}
