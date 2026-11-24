//! Fixed-point pitch, phase, panning, and division helpers for Nintendo EAD/SRD.

/// GBA ARM7TDMI master clock frequency in Hz (`2^24` = 16,777,216 Hz).
pub const GBA_CLOCK_HZ: u32 = 16_777_216;

/// Number of samples rendered per DirectSound DMA frame (`0xB0` = 176).
///
/// The retail frame renderer always mixes exactly this many frames into each
/// planar `i16` channel buffer (`0x160` halfwords per channel) before the
/// saturating converter packs the result for the DMA FIFO.
pub const DMA_FRAME_SAMPLES: usize = 0xB0;

/// Unit step (`1.0x` playback speed) in the driver's Q8 phase format (`1 << 8`).
///
/// The ARM voice mixer indexes sample data with `phase >> 8`, so the low 8
/// bits form the fractional phase and are discarded (nearest-neighbour —
/// neither revision interpolates).
pub const Q8_ONE: u32 = 1 << 8;

/// Compute the Q8 phase increment for playing a sample recorded at
/// `sample_rate_hz` through a mixer running at `mix_rate_hz`.
#[inline]
#[must_use]
pub const fn freq_to_step_q8(sample_rate_hz: u32, mix_rate_hz: u32) -> u32 {
    if mix_rate_hz == 0 {
        return 0;
    }
    (((sample_rate_hz as u64) << 8) / (mix_rate_hz as u64)) as u32
}

/// Convert a Q8 phase increment back to an effective playback rate in Hz.
#[inline]
#[must_use]
pub fn step_q8_to_freq(step_q8: u32, mix_rate_hz: u32) -> f64 {
    (step_q8 as f64) * (mix_rate_hz as f64) / (Q8_ONE as f64)
}

/// Compute the exact left/right mixer gains `(vol_l, vol_r)` used by the
/// retail Thumb voice wrapper.
///
/// The wrapper computes, with 32-bit wrapping multiplication followed by a
/// `<< 16 >> 23` logical-shift pair (equivalent to masking to 16 bits and
/// shifting right by 7):
/// ```text
/// vol_l = (((127 - pan) * volume) & 0xFFFF) >> 7
/// vol_r = ((pan * volume) & 0xFFFF) >> 7
/// ```
/// where `volume` is the 16-bit envelope/master volume and `pan` is the
/// 8-bit pan byte (`0` = hard left, `127` = hard right in normal use).
#[inline]
#[must_use]
pub const fn pan_gains(volume: u16, pan: u8) -> (u32, u32) {
    let vol = volume as u32;
    let prod_l = ((127i32 - (pan as i32)) as u32).wrapping_mul(vol);
    let prod_r = (pan as u32).wrapping_mul(vol);
    ((prod_l & 0xFFFF) >> 7, (prod_r & 0xFFFF) >> 7)
}

/// Unsigned 32-bit division exactly as implemented by the driver's ARM
/// helper routine: returns `numerator / denominator`, or `0` when
/// `denominator` is zero.
#[inline]
#[must_use]
pub const fn arm_udiv(numerator: u32, denominator: u32) -> u32 {
    match numerator.checked_div(denominator) {
        Some(v) => v,
        None => 0,
    }
}

/// Signed 32-bit division exactly as implemented by the driver's ARM helper
/// routine: returns `numerator / denominator` (truncating toward zero), or
/// `0` when `denominator` is zero. The `i32::MIN / -1` overflow case wraps
/// to `i32::MIN`, matching the helper's shift/subtract sequence.
#[inline]
#[must_use]
pub const fn arm_sdiv(numerator: i32, denominator: i32) -> i32 {
    match numerator.checked_div(denominator) {
        Some(v) => v,
        // Division by zero (ARM helper returns 0) and the i32::MIN / -1
        // overflow (the helper's shift/subtract sequence wraps to i32::MIN).
        None => {
            if denominator == 0 {
                0
            } else {
                i32::MIN
            }
        }
    }
}

/// Compute the exact DirectSound hardware sample rate in Hz for a GBA
/// Timer0 reload value.
#[inline]
#[must_use]
pub const fn rate_from_timer_reload(reload: u16) -> u32 {
    let ticks = 65_536u32 - (reload as u32);
    (GBA_CLOCK_HZ + (ticks / 2)) / ticks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pitch_pan_and_division_helpers_work() {
        assert_eq!(freq_to_step_q8(10_512, 10_512), Q8_ONE);
        assert_eq!(freq_to_step_q8(21_024, 10_512), Q8_ONE * 2);
        assert_eq!(freq_to_step_q8(5_256, 10_512), Q8_ONE / 2);
        assert_eq!(step_q8_to_freq(Q8_ONE, 10_512), 10_512.0);
        assert_eq!(pan_gains(128, 0), (127, 0));
        assert_eq!(pan_gains(128, 127), (0, 127));
        assert_eq!(pan_gains(256, 64), (126, 128));
        assert_eq!(arm_udiv(100, 7), 14);
        assert_eq!(arm_udiv(100, 0), 0);
        assert_eq!(arm_sdiv(-100, 7), -14);
        assert_eq!(arm_sdiv(100, 0), 0);
        assert_eq!(arm_sdiv(i32::MIN, -1), i32::MIN);
        assert_eq!(DMA_FRAME_SAMPLES, 176);
    }
}
