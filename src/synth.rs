//! Dual synthesis pipelines for the Nintendo EAD/SRD Super Mario Advance engine:
//! - [`AccuracySynth`]: Bit-exact emulation of both ARM7TDMI IWRAM kernel revisions.
//! - [`StudioSynth`]: 48,000 Hz `f32` high-fidelity renderer with band-limited
//!   windowed-sinc interpolation.

use crate::freq::{arm_udiv, pan_gains};
use crate::sample::EadSampleView;
use crate::version::EadSmaVersion;
use crate::voice::EadVoice;

/// Bit-exact software emulation of the Nintendo EAD/SRD ARM7TDMI IWRAM mixer.
#[derive(Debug, Default, Clone, Copy)]
pub struct AccuracySynth;

impl AccuracySynth {
    /// Run the ARM voice-mixer kernel for `out_l.len().min(out_r.len())` frames.
    ///
    /// Per frame, with sample index `phase_q8 >> 8` (nearest-neighbour — the
    /// fractional bits are discarded) and signed 8-bit sample `s`:
    /// - [`EadSmaVersion::Sma23Shift8`]: `acc += (s * vol) >> 8` (arithmetic shift),
    /// - [`EadSmaVersion::Sma4Mla`]: `acc += s * vol`,
    ///
    /// where `acc` is the running `i16` output value (each store truncates the
    /// 32-bit result to 16 bits, exactly like the ARM `strh`). Samples read
    /// beyond `pcm` contribute `0`. Returns the phase after the last frame.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn mix_kernel(
        version: EadSmaVersion,
        pcm: &[u8],
        phase_q8: u32,
        step_q8: u32,
        vol_l: u32,
        vol_r: u32,
        out_l: &mut [i16],
        out_r: &mut [i16],
    ) -> u32 {
        let frames = out_l.len().min(out_r.len());
        if frames == 0 {
            return phase_q8;
        }
        let mut phase = phase_q8;
        let vl = vol_l as i32;
        let vr = vol_r as i32;
        let out_l = &mut out_l[..frames];
        let out_r = &mut out_r[..frames];

        // Interior fast path: a 1:1 pitch run wholly inside `pcm` becomes a
        // contiguous, bounds-check-free sweep that LLVM auto-vectorizes.
        let start = (phase >> 8) as usize;
        if step_q8 == crate::freq::Q8_ONE && start <= pcm.len() && frames <= pcm.len() - start {
            let win = &pcm[start..start + frames];
            match version {
                EadSmaVersion::Sma23Shift8 => {
                    // For |vol| <= 255 (all standard GBA gains), `s * vol` fits
                    // in `i16` without overflow (`[-32640, 32385]`), unlocking
                    // 16-bit SIMD (`pmullw` + `psraw` + `paddw`).
                    if (-255..=255).contains(&vl) && (-255..=255).contains(&vr) {
                        let vl16 = vl as i16;
                        let vr16 = vr as i16;
                        for ((l, r), &b) in out_l.iter_mut().zip(out_r.iter_mut()).zip(win) {
                            let s = (b as i8) as i16;
                            *l = l.wrapping_add(s.wrapping_mul(vl16) >> 8);
                            *r = r.wrapping_add(s.wrapping_mul(vr16) >> 8);
                        }
                    } else {
                        for ((l, r), &b) in out_l.iter_mut().zip(out_r.iter_mut()).zip(win) {
                            let s = (b as i8) as i32;
                            *l = (*l as i32).wrapping_add(s.wrapping_mul(vl) >> 8) as i16;
                            *r = (*r as i32).wrapping_add(s.wrapping_mul(vr) >> 8) as i16;
                        }
                    }
                }
                EadSmaVersion::Sma4Mla => {
                    let vl16 = vl as i16;
                    let vr16 = vr as i16;
                    for ((l, r), &b) in out_l.iter_mut().zip(out_r.iter_mut()).zip(win) {
                        let s = (b as i8) as i16;
                        *l = l.wrapping_add(s.wrapping_mul(vl16));
                        *r = r.wrapping_add(s.wrapping_mul(vr16));
                    }
                }
            }
            return phase.wrapping_add(step_q8.wrapping_mul(frames as u32));
        }

        // Resampled interior fast path: when all sample indices stay in bounds
        // and `phase` does not wrap `u32`, hoist bounds checks out of the loop.
        let last_phase_u64 = (phase as u64) + ((frames - 1) as u64).saturating_mul(step_q8 as u64);
        if last_phase_u64 <= u32::MAX as u64 && ((last_phase_u64 >> 8) as usize) < pcm.len() {
            let src = &pcm[..=((last_phase_u64 >> 8) as usize)];
            match version {
                EadSmaVersion::Sma23Shift8 => {
                    if (-255..=255).contains(&vl) && (-255..=255).contains(&vr) {
                        let vl16 = vl as i16;
                        let vr16 = vr as i16;
                        for (l, r) in out_l.iter_mut().zip(out_r.iter_mut()) {
                            let s = (src[(phase >> 8) as usize] as i8) as i16;
                            *l = l.wrapping_add(s.wrapping_mul(vl16) >> 8);
                            *r = r.wrapping_add(s.wrapping_mul(vr16) >> 8);
                            phase = phase.wrapping_add(step_q8);
                        }
                    } else {
                        for (l, r) in out_l.iter_mut().zip(out_r.iter_mut()) {
                            let s = (src[(phase >> 8) as usize] as i8) as i32;
                            *l = (*l as i32).wrapping_add(s.wrapping_mul(vl) >> 8) as i16;
                            *r = (*r as i32).wrapping_add(s.wrapping_mul(vr) >> 8) as i16;
                            phase = phase.wrapping_add(step_q8);
                        }
                    }
                }
                EadSmaVersion::Sma4Mla => {
                    let vl16 = vl as i16;
                    let vr16 = vr as i16;
                    for (l, r) in out_l.iter_mut().zip(out_r.iter_mut()) {
                        let s = (src[(phase >> 8) as usize] as i8) as i16;
                        *l = l.wrapping_add(s.wrapping_mul(vl16));
                        *r = r.wrapping_add(s.wrapping_mul(vr16));
                        phase = phase.wrapping_add(step_q8);
                    }
                }
            }
            return phase;
        }

        match version {
            EadSmaVersion::Sma23Shift8 => {
                for (l, r) in out_l.iter_mut().zip(out_r.iter_mut()) {
                    let s = pcm
                        .get((phase >> 8) as usize)
                        .map_or(0, |&b| b as i8) as i32;
                    *l = (*l as i32).wrapping_add((s.wrapping_mul(vl)) >> 8) as i16;
                    *r = (*r as i32).wrapping_add((s.wrapping_mul(vr)) >> 8) as i16;
                    phase = phase.wrapping_add(step_q8);
                }
            }
            EadSmaVersion::Sma4Mla => {
                let vl16 = vl as i16;
                let vr16 = vr as i16;
                for (l, r) in out_l.iter_mut().zip(out_r.iter_mut()) {
                    let s = pcm
                        .get((phase >> 8) as usize)
                        .map_or(0, |&b| b as i8) as i16;
                    *l = l.wrapping_add(s.wrapping_mul(vl16));
                    *r = r.wrapping_add(s.wrapping_mul(vr16));
                    phase = phase.wrapping_add(step_q8);
                }
            }
        }
        phase
    }

    /// Render one [`EadVoice`] into planar `i16` mix buffers, porting the
    /// retail Thumb voice wrapper: pan gains, Q8 phase stepping, loop wrap
    /// by `loop_len << 8`, and one-shot termination.
    ///
    /// The wrapper renders in segments bounded by the sample end index
    /// (`loop_end` for looped samples, else `total_len`), computing each
    /// segment length with the driver's ARM unsigned-division helper.
    pub fn mix_voice(
        version: EadSmaVersion,
        sample: &EadSampleView<'_>,
        voice: &mut EadVoice,
        out_l: &mut [i16],
        out_r: &mut [i16],
    ) {
        if !voice.active {
            return;
        }
        let frames = out_l.len().min(out_r.len());
        if frames == 0 {
            return;
        }
        let (vol_l, vol_r) = pan_gains(voice.volume, voice.pan);
        let desc = sample.desc;
        let end = desc.end_index();
        if end == 0 {
            if !desc.is_looped() {
                voice.active = false;
            }
            return;
        }
        let looped = desc.is_looped() && desc.loop_len() > 0;
        let end_q8_u64 = (end as u64) << 8;
        let mut phase = voice.phase_q8;
        let mut done = 0usize;

        while done < frames {
            if (phase as u64) >= end_q8_u64 {
                if looped {
                    let loop_q8 = desc.loop_len().wrapping_shl(8);
                    if loop_q8 == 0 {
                        break;
                    }
                    phase = phase.wrapping_sub(loop_q8);
                    if (phase as u64) >= end_q8_u64 {
                        break;
                    }
                } else {
                    voice.active = false;
                    break;
                }
            }
            let remaining = frames - done;
            let projected_q8 = (phase as u64) + (voice.step_q8 as u64) * (remaining as u64);
            if projected_q8 < end_q8_u64 {
                phase = Self::mix_kernel(
                    version,
                    sample.pcm_bytes,
                    phase,
                    voice.step_q8,
                    vol_l,
                    vol_r,
                    &mut out_l[done..frames],
                    &mut out_r[done..frames],
                );
                break;
            }
            // Segment length via the ARM unsigned-division helper semantics.
            let cnt = arm_udiv(
                end.wrapping_shl(8)
                    .wrapping_sub(phase)
                    .wrapping_sub(1)
                    .wrapping_add(voice.step_q8),
                voice.step_q8,
            ) as usize;
            let cnt = cnt.min(remaining);
            if cnt > 0 {
                phase = Self::mix_kernel(
                    version,
                    sample.pcm_bytes,
                    phase,
                    voice.step_q8,
                    vol_l,
                    vol_r,
                    &mut out_l[done..done + cnt],
                    &mut out_r[done..done + cnt],
                );
                done += cnt;
            }
            if looped {
                phase = phase.wrapping_sub(desc.loop_len().wrapping_shl(8));
                if cnt == 0 {
                    break;
                }
            } else {
                voice.active = false;
                break;
            }
        }
        voice.phase_q8 = phase;
    }

    /// Convert planar `i16` mix data to DirectSound DMA bytes with the ARM
    /// saturating converter kernel.
    ///
    /// - [`EadSmaVersion::Sma23Shift8`]: stores the low byte of each `i16`,
    ///   except values outside `[-128, 127]` saturate to `127`/`-128`.
    /// - [`EadSmaVersion::Sma4Mla`]: first computes
    ///   `(v + (v < 0 ? 127 : 0)) >> 7`, then applies the same saturation
    ///   test to the shifted value.
    pub fn convert_to_dma8(version: EadSmaVersion, src: &[i16], dst: &mut [i8]) {
        let n = src.len().min(dst.len());
        let src = &src[..n];
        let dst = &mut dst[..n];
        match version {
            EadSmaVersion::Sma23Shift8 => {
                // Mathematically identical to the ARM `(v & 0xFF80)` saturation
                // check for all 65,536 `i16` inputs, and auto-vectorizes to
                // `pminsw`/`pmaxsw`/`packsswb`.
                for (d, &v) in dst.iter_mut().zip(src) {
                    *d = v.clamp(-128, 127) as i8;
                }
            }
            EadSmaVersion::Sma4Mla => {
                // `(v + (v < 0 ? 127 : 0)) >> 7` equals `v / 128` (truncating
                // toward zero) for every `i16`, staying in 16-bit SIMD lanes.
                for (d, &v) in dst.iter_mut().zip(src) {
                    *d = (v / 128).clamp(-128, 127) as i8;
                }
            }
        }
    }

    /// Run the ARM buffer-combine kernel: `main[i] += aux[i]` (16-bit
    /// wrapping store), then `aux[i]` is attenuated in place by `shift`.
    ///
    /// - [`EadSmaVersion::Sma23Shift8`]: `aux = (aux + (1 << (shift-1))) >> shift`
    ///   for all values (bias `0` when `shift == 0`).
    /// - [`EadSmaVersion::Sma4Mla`]: `aux = (aux + (aux < 0 ? (1<<shift)-1 : 0)) >> shift`.
    ///
    /// Shifts of 32 or more follow ARM register-shift semantics (the shifted
    /// value becomes the sign extension `0`/`-1`).
    pub fn combine(version: EadSmaVersion, main: &mut [i16], aux: &mut [i16], shift: u32) {
        let n = main.len().min(aux.len());
        let main = &mut main[..n];
        let aux = &mut aux[..n];
        if shift >= 32 {
            for (m, a) in main.iter_mut().zip(aux.iter_mut()) {
                let av = *a;
                *m = m.wrapping_add(av);
                *a = av >> 15;
            }
            return;
        }
        if shift == 0 {
            for (m, &a) in main.iter_mut().zip(aux.iter()) {
                *m = m.wrapping_add(a);
            }
            return;
        }
        match version {
            EadSmaVersion::Sma23Shift8 => {
                if shift < 16 {
                    // For shift in 1..=15, `(v + (1 << (shift - 1))) >> shift` equals
                    // `(v >> shift) + ((v >> (shift - 1)) & 1)` for all 65,536 `i16`s,
                    // staying entirely in 16-bit SIMD lanes without widen/narrow.
                    let s1 = shift - 1;
                    for (m, a) in main.iter_mut().zip(aux.iter_mut()) {
                        let av = *a;
                        *m = m.wrapping_add(av);
                        *a = (av >> shift).wrapping_add((av >> s1) & 1);
                    }
                } else {
                    let bias: i32 = 1i32 << (shift - 1);
                    for (m, a) in main.iter_mut().zip(aux.iter_mut()) {
                        let av = *a;
                        *m = m.wrapping_add(av);
                        *a = ((av as i32).wrapping_add(bias) >> shift) as i16;
                    }
                }
            }
            EadSmaVersion::Sma4Mla => {
                if shift < 16 {
                    // For shift in 1..=15 and `av < 0`, `av + ((1 << shift) - 1)` is
                    // in `[-32767, 32766]`, never overflowing `i16`.
                    let mask16 = ((1u16 << shift) - 1) as i16;
                    for (m, a) in main.iter_mut().zip(aux.iter_mut()) {
                        let av = *a;
                        *m = m.wrapping_add(av);
                        *a = av.wrapping_add((av >> 15) & mask16) >> shift;
                    }
                } else {
                    let mask: i32 = ((1u32 << shift) - 1) as i32;
                    for (m, a) in main.iter_mut().zip(aux.iter_mut()) {
                        let av = *a;
                        *m = m.wrapping_add(av);
                        let v = av as i32;
                        *a = (v.wrapping_add((v >> 31) & mask) >> shift) as i16;
                    }
                }
            }
        }
    }
}

// ============================================================================
// StudioSynth: 48,000 Hz f32 Band-Limited Lanczos-3 + Equal-Power Panning
// ============================================================================

/// Number of fractional polyphase steps in [`StudioSynth`]'s compile-time Lanczos-3 LUT.
const LANCZOS_PHASES: usize = 1024;

const fn sin_const(mut x: f64) -> f64 {
    const PI: f64 = core::f64::consts::PI;
    const TWO_PI: f64 = core::f64::consts::TAU;
    const HALF_PI: f64 = core::f64::consts::FRAC_PI_2;
    let k = (x / TWO_PI) as i64;
    x -= (k as f64) * TWO_PI;
    if x > PI {
        x -= TWO_PI;
    } else if x < -PI {
        x += TWO_PI;
    }
    if x > HALF_PI {
        x = PI - x;
    } else if x < -HALF_PI {
        x = -PI - x;
    }
    let r2 = x * x;
    x * (1.0
        - r2 * (1.0 / 6.0
            - r2 * (1.0 / 120.0
                - r2 * (1.0 / 5_040.0
                    - r2 * (1.0 / 362_880.0
                        - r2 * (1.0 / 39_916_800.0 - r2 / 6_227_020_800.0))))))
}

const fn cos_const(x: f64) -> f64 {
    sin_const(x + core::f64::consts::FRAC_PI_2)
}

const fn lanczos_kernel(x: f64, a: f64) -> f64 {
    let ax = if x < 0.0 { -x } else { x };
    if ax < 1e-12 {
        return 1.0;
    }
    if ax >= a {
        return 0.0;
    }
    let pi_x = core::f64::consts::PI * x;
    let pi_x_a = pi_x / a;
    (sin_const(pi_x) / pi_x) * (sin_const(pi_x_a) / pi_x_a)
}

const fn build_lanczos3_lut() -> [[f32; 8]; LANCZOS_PHASES] {
    let mut table = [[0.0f32; 8]; LANCZOS_PHASES];
    let mut p = 0usize;
    while p < LANCZOS_PHASES {
        let frac = (p as f64) / (LANCZOS_PHASES as f64);
        let mut sum = 0.0f64;
        let mut w = [0.0f64; 6];
        let mut k = 0usize;
        while k < 6 {
            let offset = (k as i32) - 2; // -2, -1, 0, +1, +2, +3
            let val = lanczos_kernel(frac - (offset as f64), 3.0);
            w[k] = val;
            sum += val;
            k += 1;
        }
        let inv_sum = if sum != 0.0 { 1.0 / sum } else { 1.0 };
        let mut j = 0usize;
        while j < 6 {
            table[p][j] = (w[j] * inv_sum) as f32;
            j += 1;
        }
        p += 1;
    }
    table
}

static LANCZOS3_LUT: [[f32; 8]; LANCZOS_PHASES] = build_lanczos3_lut();

/// High-precision 48,000 Hz `f32` studio synthesizer with band-limited Lanczos
/// windowed-sinc resampling and true equal-power stereo panning.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StudioSynth {
    /// Target output sample rate in Hz (default: `48_000.0`).
    pub output_rate_hz: f64,
    /// Lanczos kernel half-width `a` (default: `3`, which uses the compile-time 1,024-phase LUT).
    pub lanczos_a: usize,
}

impl Default for StudioSynth {
    fn default() -> Self {
        Self {
            output_rate_hz: 48_000.0,
            lanczos_a: 3,
        }
    }
}

impl StudioSynth {
    /// Create a new [`StudioSynth`] configured for `output_rate_hz` with Lanczos-3 resampling.
    #[inline]
    #[must_use]
    pub const fn new(output_rate_hz: f64) -> Self {
        Self {
            output_rate_hz,
            lanczos_a: 3,
        }
    }

    /// Compute true equal-power stereo panning gains `(gain_l, gain_r)` for `pan in -1.0..=1.0`.
    #[inline]
    #[must_use]
    pub fn equal_power_pan(pan: f32) -> (f32, f32) {
        let clamped = pan.clamp(-1.0, 1.0) as f64;
        let angle = (clamped + 1.0) * core::f64::consts::FRAC_PI_4;
        (cos_const(angle) as f32, sin_const(angle) as f32)
    }

    /// Render a single [`EadSampleView`] voice additively into `out_l` and `out_r` at
    /// `self.output_rate_hz` using band-limited Lanczos interpolation and equal-power panning.
    ///
    /// Returns `true` if the voice is still active after rendering, or `false` if a one-shot
    /// sample reached its end.
    #[allow(clippy::too_many_arguments)]
    pub fn render_voice_f32(
        &self,
        sample: &EadSampleView<'_>,
        pos: &mut f64,
        playback_rate_hz: f64,
        gain: f32,
        pan: f32,
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) -> bool {
        let frames = out_l.len().min(out_r.len());
        let mut p = *pos;
        if frames == 0
            || sample.is_empty()
            || !p.is_finite()
            || !self.output_rate_hz.is_finite()
            || !playback_rate_hz.is_finite()
            || self.output_rate_hz <= 0.0
            || playback_rate_hz <= 0.0
        {
            return false;
        }

        let (pan_l, pan_r) = Self::equal_power_pan(pan);
        let scale_l = (gain * pan_l) * (1.0 / 128.0);
        let scale_r = (gain * pan_r) * (1.0 / 128.0);
        let step = playback_rate_hz / self.output_rate_hz;
        let pcm = sample.pcm_bytes;
        let pcm_len = pcm.len();
        let desc = sample.desc;
        let (loop_start, loop_end) = if desc.is_looped()
            && (desc.loop_start as usize) < (desc.loop_end as usize)
            && (desc.loop_end as usize) <= pcm_len
        {
            (Some(desc.loop_start as usize), desc.loop_end as usize)
        } else {
            let end = if desc.total_len > 0 {
                (desc.total_len as usize).min(pcm_len)
            } else {
                pcm_len
            };
            (None, end)
        };
        let loop_end_f = loop_end as f64;
        let loop_len_f = loop_start.map(|ls| (loop_end - ls) as f64);

        if self.lanczos_a == 3 {
            const Q32_SCALE: f64 = 4_294_967_296.0;
            // Stack-allocated L1-hot f32 block buffer (512 + 8 guard padding so
            // `b & 511` + 5 is statically < 520 with zero bounds checks).
            const FBUF_MASK: usize = 511;
            const FBUF_CAP: usize = FBUF_MASK + 1;
            let mut fbuf = [0.0f32; FBUF_CAP + 8];
            let safe_end_f = loop_end.saturating_sub(3) as f64;
            let step_fp = (step * Q32_SCALE) as u64;
            let mut i = 0usize;

            while i < frames {
                if p >= loop_end_f {
                    if let Some(ll) = loop_len_f {
                        p -= ll;
                        if p >= loop_end_f {
                            let ls = loop_start.unwrap_or(0) as f64;
                            p = ls + ((p - ls) % ll);
                        }
                    } else {
                        *pos = p;
                        return false;
                    }
                }

                // Interior fast path: 32.32 fixed-point phase + block PCM->f32
                // pre-conversion + 4-way unrolled bounds-check-free Lanczos-3.
                if loop_end >= 6 && p >= 2.0 && p < safe_end_f && step_fp > 0 {
                    let rem = frames - i;
                    let max_n = ((safe_end_f - p) / step) as usize;
                    let mut n = max_n.min(rem);
                    let fp = (p * Q32_SCALE) as u64;
                    while n > 0 {
                        let last_fp = fp.saturating_add(((n - 1) as u64).saturating_mul(step_fp));
                        if ((last_fp >> 32) as usize) + 3 < loop_end {
                            break;
                        }
                        n -= 1;
                    }
                    if n > 0 && (fp >> 32) >= 2 {
                        let mut ph_shifted = fp - (2u64 << 32);
                        let mut done_in = 0usize;
                        while done_in < n {
                            let rem_in = n - done_in;
                            let first_b0 = (ph_shifted >> 32) as usize;
                            // Choose a chunk whose PCM span fits inside `FBUF_CAP`.
                            let mut chunk_n = rem_in;
                            let last_b0 = ((ph_shifted
                                .wrapping_add(((chunk_n - 1) as u64).wrapping_mul(step_fp)))
                                >> 32) as usize;
                            if last_b0 - first_b0 + 6 > FBUF_CAP {
                                let max_advance = (FBUF_CAP - 6) as u64;
                                chunk_n = (((max_advance << 32) / step_fp) as usize)
                                    .max(1)
                                    .min(rem_in);
                                while chunk_n > 1 {
                                    let lb = ((ph_shifted
                                        .wrapping_add(((chunk_n - 1) as u64).wrapping_mul(step_fp)))
                                        >> 32) as usize;
                                    if lb - first_b0 + 6 <= FBUF_CAP {
                                        break;
                                    }
                                    chunk_n -= 1;
                                }
                            }
                            let chunk_last_b0 = ((ph_shifted
                                .wrapping_add(((chunk_n - 1) as u64).wrapping_mul(step_fp)))
                                >> 32) as usize;
                            let span = (chunk_last_b0 - first_b0) + 6;
                            if span <= FBUF_CAP {
                                let src = &pcm[first_b0..first_b0 + span];
                                for (dst_f, &b) in fbuf[..span].iter_mut().zip(src) {
                                    *dst_f = (b as i8) as f32;
                                }
                                let mut ph_local = ph_shifted - ((first_b0 as u64) << 32);
                                let ol_slice = &mut out_l[i + done_in..i + done_in + chunk_n];
                                let or_slice = &mut out_r[i + done_in..i + done_in + chunk_n];
                                let (l_chunks, l_rem) = ol_slice.as_chunks_mut::<4>();
                                let (r_chunks, r_rem) = or_slice.as_chunks_mut::<4>();
                                for (lc, rc) in l_chunks.iter_mut().zip(r_chunks.iter_mut()) {
                                    let ph0 = ph_local;
                                    let ph1 = ph0.wrapping_add(step_fp);
                                    let ph2 = ph1.wrapping_add(step_fp);
                                    let ph3 = ph2.wrapping_add(step_fp);
                                    ph_local = ph3.wrapping_add(step_fp);

                                    let b0 = ((ph0 >> 32) as usize) & FBUF_MASK;
                                    let w0 = &LANCZOS3_LUT
                                        [((ph0 >> 22) as usize) & (LANCZOS_PHASES - 1)];
                                    let v0 = ((fbuf[b0] * w0[0] + fbuf[b0 + 1] * w0[1])
                                        + (fbuf[b0 + 2] * w0[2] + fbuf[b0 + 3] * w0[3]))
                                        + (fbuf[b0 + 4] * w0[4] + fbuf[b0 + 5] * w0[5]);

                                    let b1 = ((ph1 >> 32) as usize) & FBUF_MASK;
                                    let w1 = &LANCZOS3_LUT
                                        [((ph1 >> 22) as usize) & (LANCZOS_PHASES - 1)];
                                    let v1 = ((fbuf[b1] * w1[0] + fbuf[b1 + 1] * w1[1])
                                        + (fbuf[b1 + 2] * w1[2] + fbuf[b1 + 3] * w1[3]))
                                        + (fbuf[b1 + 4] * w1[4] + fbuf[b1 + 5] * w1[5]);

                                    let b2 = ((ph2 >> 32) as usize) & FBUF_MASK;
                                    let w2 = &LANCZOS3_LUT
                                        [((ph2 >> 22) as usize) & (LANCZOS_PHASES - 1)];
                                    let v2 = ((fbuf[b2] * w2[0] + fbuf[b2 + 1] * w2[1])
                                        + (fbuf[b2 + 2] * w2[2] + fbuf[b2 + 3] * w2[3]))
                                        + (fbuf[b2 + 4] * w2[4] + fbuf[b2 + 5] * w2[5]);

                                    let b3 = ((ph3 >> 32) as usize) & FBUF_MASK;
                                    let w3 = &LANCZOS3_LUT
                                        [((ph3 >> 22) as usize) & (LANCZOS_PHASES - 1)];
                                    let v3 = ((fbuf[b3] * w3[0] + fbuf[b3 + 1] * w3[1])
                                        + (fbuf[b3 + 2] * w3[2] + fbuf[b3 + 3] * w3[3]))
                                        + (fbuf[b3 + 4] * w3[4] + fbuf[b3 + 5] * w3[5]);

                                    lc[0] += v0 * scale_l;
                                    lc[1] += v1 * scale_l;
                                    lc[2] += v2 * scale_l;
                                    lc[3] += v3 * scale_l;
                                    rc[0] += v0 * scale_r;
                                    rc[1] += v1 * scale_r;
                                    rc[2] += v2 * scale_r;
                                    rc[3] += v3 * scale_r;
                                }
                                for (ol, or) in l_rem.iter_mut().zip(r_rem.iter_mut()) {
                                    let b0 = ((ph_local >> 32) as usize) & FBUF_MASK;
                                    let w = &LANCZOS3_LUT
                                        [((ph_local >> 22) as usize) & (LANCZOS_PHASES - 1)];
                                    let val = ((fbuf[b0] * w[0] + fbuf[b0 + 1] * w[1])
                                        + (fbuf[b0 + 2] * w[2] + fbuf[b0 + 3] * w[3]))
                                        + (fbuf[b0 + 4] * w[4] + fbuf[b0 + 5] * w[5]);
                                    *ol += val * scale_l;
                                    *or += val * scale_r;
                                    ph_local = ph_local.wrapping_add(step_fp);
                                }
                                ph_shifted =
                                    ph_shifted.wrapping_add((chunk_n as u64).wrapping_mul(step_fp));
                            } else {
                                // Extreme step (> ~500 samples/frame): single-frame direct read.
                                let b0 = first_b0;
                                let w = &LANCZOS3_LUT
                                    [((ph_shifted >> 22) as usize) & (LANCZOS_PHASES - 1)];
                                let s0 = (pcm[b0] as i8) as f32;
                                let s1 = (pcm[b0 + 1] as i8) as f32;
                                let s2 = (pcm[b0 + 2] as i8) as f32;
                                let s3 = (pcm[b0 + 3] as i8) as f32;
                                let s4 = (pcm[b0 + 4] as i8) as f32;
                                let s5 = (pcm[b0 + 5] as i8) as f32;
                                let val = (s0 * w[0] + s1 * w[1])
                                    + (s2 * w[2] + s3 * w[3])
                                    + (s4 * w[4] + s5 * w[5]);
                                out_l[i + done_in] += val * scale_l;
                                out_r[i + done_in] += val * scale_r;
                                ph_shifted = ph_shifted.wrapping_add(step_fp);
                            }
                            done_in += chunk_n;
                        }
                        p += (n as f64) * step;
                        i += n;
                        continue;
                    }
                }

                // Boundary / wrap single-frame fallback
                let idx = p as isize;
                let frac = p - (idx as f64);
                let phase = ((frac * (LANCZOS_PHASES as f64)) as usize).min(LANCZOS_PHASES - 1);
                let w = &LANCZOS3_LUT[phase];
                let val = (fetch_sample_wrapped(pcm, idx - 2, loop_start, loop_end) * w[0]
                    + fetch_sample_wrapped(pcm, idx - 1, loop_start, loop_end) * w[1]
                    + fetch_sample_wrapped(pcm, idx, loop_start, loop_end) * w[2])
                    + (fetch_sample_wrapped(pcm, idx + 1, loop_start, loop_end) * w[3]
                        + fetch_sample_wrapped(pcm, idx + 2, loop_start, loop_end) * w[4]
                        + fetch_sample_wrapped(pcm, idx + 3, loop_start, loop_end) * w[5]);
                out_l[i] += val * scale_l;
                out_r[i] += val * scale_r;
                p += step;
                i += 1;
            }
        } else {
            let a_i32 = self.lanczos_a.max(1) as i32;
            let a_f64 = a_i32 as f64;
            for (ol, or) in out_l[..frames].iter_mut().zip(&mut out_r[..frames]) {
                if p >= loop_end_f {
                    if let Some(ll) = loop_len_f {
                        p -= ll;
                        if p >= loop_end_f {
                            let ls = loop_start.unwrap_or(0) as f64;
                            p = ls + ((p - ls) % ll);
                        }
                    } else {
                        *pos = p;
                        return false;
                    }
                }

                let idx = p as isize;
                let frac = p - (idx as f64);
                let mut acc = 0.0f64;
                let mut wsum = 0.0f64;
                let mut k = -a_i32 + 1;
                while k <= a_i32 {
                    let w = lanczos_kernel(frac - (k as f64), a_f64);
                    let s = fetch_sample_wrapped(pcm, idx + (k as isize), loop_start, loop_end);
                    acc += (s as f64) * w;
                    wsum += w;
                    k += 1;
                }
                let val = if wsum != 0.0 {
                    (acc / wsum) as f32
                } else {
                    0.0
                };
                *ol += val * scale_l;
                *or += val * scale_r;
                p += step;
            }
        }

        *pos = p;
        true
    }
}

#[inline(always)]
fn fetch_sample_wrapped(
    pcm: &[u8],
    mut idx: isize,
    loop_start: Option<usize>,
    loop_end: usize,
) -> f32 {
    if idx < 0 {
        return 0.0;
    }
    if (idx as usize) >= loop_end {
        if let Some(ls) = loop_start {
            let ll = (loop_end - ls) as isize;
            idx = (ls as isize) + ((idx - (ls as isize)) % ll);
        } else {
            return 0.0;
        }
    }
    (pcm[idx as usize] as i8) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::EadSampleDesc;

    #[test]
    fn converter_saturation_edges() {
        let src: [i16; 8] = [0, 127, 128, -128, -129, 1000, -1000, 32767];
        let mut dst = [0i8; 8];
        AccuracySynth::convert_to_dma8(EadSmaVersion::Sma23Shift8, &src, &mut dst);
        assert_eq!(dst, [0, 127, 127, -128, -128, 127, -128, 127]);
        AccuracySynth::convert_to_dma8(EadSmaVersion::Sma4Mla, &src, &mut dst);
        assert_eq!(dst, [0, 0, 1, -1, -1, 7, -7, 127]);
    }

    #[test]
    fn converter_and_combine_match_arm_reference_for_all_i16() {
        fn ref_saturate_low_byte(t: i32) -> i8 {
            let masked = t & 0xFF80;
            if masked != 0 && masked != 0xFF80 {
                (127u32.wrapping_add((t as u32) >> 31) & 0xFF) as u8 as i8
            } else {
                t as i8
            }
        }

        let mut chunk_src = [0i16; 256];
        let mut chunk_dst = [0i8; 256];
        let mut main_buf = [0i16; 256];
        let mut aux_buf = [0i16; 256];

        for block in 0..256i32 {
            for (i, slot) in chunk_src.iter_mut().enumerate() {
                *slot = (block * 256 + i as i32 - 32768) as i16;
            }
            // Exhaustive convert_to_dma8 equivalence for both revisions
            AccuracySynth::convert_to_dma8(EadSmaVersion::Sma23Shift8, &chunk_src, &mut chunk_dst);
            for (i, &v) in chunk_src.iter().enumerate() {
                assert_eq!(chunk_dst[i], ref_saturate_low_byte(v as i32));
            }
            AccuracySynth::convert_to_dma8(EadSmaVersion::Sma4Mla, &chunk_src, &mut chunk_dst);
            for (i, &v) in chunk_src.iter().enumerate() {
                let t = (v as i32).wrapping_add(if v < 0 { 127 } else { 0 }) >> 7;
                assert_eq!(chunk_dst[i], ref_saturate_low_byte(t));
            }

            // Exhaustive combine equivalence across key shift values
            for &shift in &[0u32, 1, 2, 4, 7, 15, 16, 31, 32, 40] {
                main_buf.fill(100);
                aux_buf.copy_from_slice(&chunk_src);
                AccuracySynth::combine(
                    EadSmaVersion::Sma23Shift8,
                    &mut main_buf,
                    &mut aux_buf,
                    shift,
                );
                let bias: i32 = if shift == 0 || shift >= 32 {
                    0
                } else {
                    1i32 << (shift - 1)
                };
                for (i, &v) in chunk_src.iter().enumerate() {
                    let expected_aux = (if shift >= 32 {
                        i32::from(v < 0).wrapping_neg()
                    } else {
                        (v as i32).wrapping_add(bias) >> shift
                    }) as i16;
                    assert_eq!(main_buf[i], 100i16.wrapping_add(v));
                    assert_eq!(aux_buf[i], expected_aux);
                }

                main_buf.fill(-100);
                aux_buf.copy_from_slice(&chunk_src);
                AccuracySynth::combine(
                    EadSmaVersion::Sma4Mla,
                    &mut main_buf,
                    &mut aux_buf,
                    shift,
                );
                let mask: i32 = if shift >= 32 {
                    -1
                } else {
                    ((1u32 << shift) - 1) as i32
                };
                for (i, &v) in chunk_src.iter().enumerate() {
                    let expected_aux = (if shift >= 32 {
                        i32::from(v < 0).wrapping_neg()
                    } else {
                        (v as i32).wrapping_add(if v < 0 { mask } else { 0 }) >> shift
                    }) as i16;
                    assert_eq!(main_buf[i], (-100i16).wrapping_add(v));
                    assert_eq!(aux_buf[i], expected_aux);
                }
            }
        }
    }

    #[test]
    fn mix_voice_edge_cases_do_not_panic_or_loop() {
        let pcm = [10u8, 20, 30, 40];
        let zero_desc = EadSampleDesc {
            total_len: 0,
            rate_word: 0,
            loop_start: 0,
            loop_end: 0,
        };
        let view_zero = EadSampleView::from_parts(zero_desc, &pcm);
        let mut voice = EadVoice::new(256, 128, 64);
        let mut out_l = [0i16; 16];
        let mut out_r = [0i16; 16];
        AccuracySynth::mix_voice(
            EadSmaVersion::Sma23Shift8,
            &view_zero,
            &mut voice,
            &mut out_l,
            &mut out_r,
        );
        assert!(!voice.active);

        // One-shot voice starting past end terminates cleanly without underflow
        let oneshot_desc = EadSampleDesc {
            total_len: 4,
            rate_word: 0,
            loop_start: 0,
            loop_end: 0,
        };
        let view_os = EadSampleView::from_parts(oneshot_desc, &pcm);
        let mut voice_past = EadVoice::with_phase(10 << 8, 256, 128, 64);
        AccuracySynth::mix_voice(
            EadSmaVersion::Sma4Mla,
            &view_os,
            &mut voice_past,
            &mut out_l,
            &mut out_r,
        );
        assert!(!voice_past.active);
    }

    #[test]
    fn studio_synth_renders_finite_stereo_audio() {
        let pcm: [u8; 64] = core::array::from_fn(|i| (((i * 19) % 256) as i32 - 128) as i8 as u8);
        let desc = EadSampleDesc {
            total_len: 64,
            rate_word: 0,
            loop_start: 8,
            loop_end: 56,
        };
        let view = EadSampleView::from_parts(desc, &pcm);
        let studio = StudioSynth::default();
        let mut l = [0.0f32; 128];
        let mut r = [0.0f32; 128];
        let mut pos = 0.0;
        assert!(studio.render_voice_f32(&view, &mut pos, 10_512.0, 0.9, -0.3, &mut l, &mut r));
        assert!(l.iter().all(|v| v.is_finite()));
        assert!(r.iter().all(|v| v.is_finite()));
        assert!(l.iter().any(|&v| v.abs() > 1e-4));

        // Non-finite position returns false cleanly
        let mut bad_pos = f64::NAN;
        assert!(!studio.render_voice_f32(&view, &mut bad_pos, 10_512.0, 0.9, 0.0, &mut l, &mut r));
        let mut inf_pos = f64::INFINITY;
        assert!(!studio.render_voice_f32(&view, &mut inf_pos, 10_512.0, 0.9, 0.0, &mut l, &mut r));

        let studio_a2 = StudioSynth {
            output_rate_hz: 48_000.0,
            lanczos_a: 2,
        };
        let mut l2 = [0.0f32; 64];
        let mut r2 = [0.0f32; 64];
        let mut pos2 = 0.0;
        assert!(studio_a2.render_voice_f32(&view, &mut pos2, 10_512.0, 0.9, 0.2, &mut l2, &mut r2));
        assert!(l2.iter().all(|v| v.is_finite()));
    }
}
