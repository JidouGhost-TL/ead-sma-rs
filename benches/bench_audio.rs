use std::hint::black_box;
use std::time::Instant;

use ead_sma::{
    AccuracySynth, EadSampleDesc, EadSampleView, EadSmaVersion, EadVoice, StudioSynth,
};

fn main() {
    println!("=== ead-sma-rs Audio Performance Benchmarks ===");

    let pcm: Vec<u8> = (0i32..4096)
        .map(|i| (((i * 29 + 13) % 256) - 128) as i8 as u8)
        .collect();
    let desc = EadSampleDesc {
        total_len: 4096,
        rate_word: 0,
        loop_start: 256,
        loop_end: 3840,
    };
    let sample = EadSampleView::from_parts(desc, &pcm);

    let iters = 200_000usize;

    // 1. AccuracySynth Sma23Shift8 voice frame (176 frames)
    {
        let mut out_l = [0i16; 176];
        let mut out_r = [0i16; 176];
        let start = Instant::now();
        for _ in 0..iters {
            out_l.fill(0);
            out_r.fill(0);
            let mut voice = EadVoice::new(256, 200, 64);
            AccuracySynth::mix_voice(
                EadSmaVersion::Sma23Shift8,
                black_box(&sample),
                &mut voice,
                &mut out_l,
                &mut out_r,
            );
            black_box((&out_l, &out_r));
        }
        let elapsed = start.elapsed();
        println!(
            "AccuracySynth::mix_voice Sma23Shift8 (176f):  {:8.2} ns/iter ({:8.2} MSamples/sec)",
            elapsed.as_nanos() as f64 / iters as f64,
            ((iters * 176) as f64 / elapsed.as_secs_f64()) / 1_000_000.0
        );
    }

    // 2. AccuracySynth Sma4Mla voice frame (176 frames)
    {
        let mut out_l = [0i16; 176];
        let mut out_r = [0i16; 176];
        let start = Instant::now();
        for _ in 0..iters {
            out_l.fill(0);
            out_r.fill(0);
            let mut voice = EadVoice::new(256, 200, 64);
            AccuracySynth::mix_voice(
                EadSmaVersion::Sma4Mla,
                black_box(&sample),
                &mut voice,
                &mut out_l,
                &mut out_r,
            );
            black_box((&out_l, &out_r));
        }
        let elapsed = start.elapsed();
        println!(
            "AccuracySynth::mix_voice Sma4Mla (176f):      {:8.2} ns/iter ({:8.2} MSamples/sec)",
            elapsed.as_nanos() as f64 / iters as f64,
            ((iters * 176) as f64 / elapsed.as_secs_f64()) / 1_000_000.0
        );
    }

    // 3. AccuracySynth convert + combine (176 samples)
    {
        let src: Vec<i16> = (0i32..176).map(|i| (i * 37 - 3000) as i16).collect();
        let mut dst = [0i8; 176];
        let mut aux: Vec<i16> = (0i32..176).map(|i| (i * 21 - 900) as i16).collect();
        let mut main_buf = src.clone();
        let start = Instant::now();
        for _ in 0..iters {
            main_buf.copy_from_slice(&src);
            AccuracySynth::convert_to_dma8(
                EadSmaVersion::Sma4Mla,
                black_box(&main_buf),
                &mut dst,
            );
            AccuracySynth::combine(
                EadSmaVersion::Sma23Shift8,
                &mut main_buf,
                &mut aux,
                2,
            );
            black_box((&dst, &main_buf));
        }
        let elapsed = start.elapsed();
        println!(
            "AccuracySynth::convert+combine (176):         {:8.2} ns/iter ({:8.2} MSamples/sec)",
            elapsed.as_nanos() as f64 / iters as f64,
            ((iters * 176) as f64 / elapsed.as_secs_f64()) / 1_000_000.0
        );
    }

    // 4. StudioSynth 48 kHz Lanczos-3 (1024 frames)
    let studio = StudioSynth::default();
    let mut out_l = vec![0.0f32; 1024];
    let mut out_r = vec![0.0f32; 1024];
    let studio_iters = 50_000usize;
    let start = Instant::now();
    for _ in 0..studio_iters {
        let mut pos = 0.0;
        studio.render_voice_f32(
            black_box(&sample),
            &mut pos,
            10_512.0,
            0.85,
            0.25,
            &mut out_l,
            &mut out_r,
        );
        black_box((&out_l, &out_r));
    }
    let elapsed = start.elapsed();
    println!(
        "StudioSynth::render_voice_f32 (48kHz Lanczos3, 1k):{:8.2} ns/iter ({:8.2} MSamples/sec)",
        elapsed.as_nanos() as f64 / studio_iters as f64,
        ((studio_iters * 1024) as f64 / elapsed.as_secs_f64()) / 1_000_000.0
    );
}
