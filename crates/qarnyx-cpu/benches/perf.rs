//! Realistic-shape throughput benchmark (single backend, steady state).
//!
//! Run with: cargo bench -p qarnyx-cpu --bench perf
//! Shapes follow Qwen3.5-2B: FFN [6144, 2048], attention/out [2048, 2048],
//! and a slice of the Q6_K output head. Reports ms/op, weight GB/s and, for
//! matmul, GFLOP/s. Weights are sized to exceed L2 so memory behaviour counts.

use std::hint::black_box;
use std::time::{Duration, Instant};

use qarnyx_core::{Backend, DType, TensorView};
use qarnyx_cpu::CpuBackend;

fn next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn quant_rows(raw_type: u32, rows: usize, columns: usize) -> Vec<u8> {
    let (block_elems, block_bytes, scale_offsets): (usize, usize, &[usize]) = match raw_type {
        12 => (256, 144, &[0, 2]),
        13 => (256, 176, &[0, 2]),
        14 => (256, 210, &[208]),
        _ => unreachable!(),
    };
    let mut state = 0x1234_5678_9abc_def1_u64 ^ raw_type as u64;
    let mut bytes = vec![0_u8; rows * (columns / block_elems) * block_bytes];
    for block in bytes.chunks_exact_mut(block_bytes) {
        for byte in block.iter_mut() {
            *byte = next(&mut state) as u8;
        }
        for &offset in scale_offsets {
            block[offset..offset + 2].copy_from_slice(&0x2c00_u16.to_le_bytes());
        }
    }
    bytes
}

fn f32_rows(rows: usize, columns: usize) -> Vec<u8> {
    let mut state = 0x0bad_cafe_u64;
    (0..rows * columns)
        .flat_map(|_| (((next(&mut state) % 2001) as f32 / 1000.0) - 1.0).to_le_bytes())
        .collect()
}

fn dtype_of(raw_type: u32) -> DType {
    match raw_type {
        0 => DType::F32,
        12 => DType::Q4K,
        13 => DType::Q5K,
        _ => DType::Q6K,
    }
}

/// Runs `op` until at least `budget` has elapsed; returns seconds per call (best of 3 windows).
fn time(budget: Duration, mut op: impl FnMut()) -> f64 {
    op(); // warm caches and thread-local scratch
    let mut best = f64::MAX;
    for _ in 0..3 {
        let start = Instant::now();
        let mut iterations = 0_u32;
        while start.elapsed() < budget {
            op();
            iterations += 1;
        }
        best = best.min(start.elapsed().as_secs_f64() / f64::from(iterations));
    }
    best
}

fn backends(workers: usize) -> Vec<(&'static str, CpuBackend)> {
    let all = vec![
        ("scalar", CpuBackend::new_scalar(workers).unwrap()),
        ("dequant", CpuBackend::new_dequant(workers).unwrap()), // DEQUANT
        ("fused-q4k", CpuBackend::new_fused_q4k(workers).unwrap()),
        ("default", CpuBackend::new(workers).unwrap()),
    ];
    // SKIP_SCALAR=1 drops the (very slow) scalar reference from repeated runs.
    let skip = std::env::var_os("SKIP_SCALAR").is_some();
    all.into_iter()
        .filter(|(name, _)| !(skip && *name == "scalar"))
        .collect()
}

fn main() {
    let workers: usize = std::env::var("WORKERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let budget = Duration::from_millis(300);
    println!("workers={workers}\n\nMatvec (decode):");
    let cases: [(&str, u32, usize, usize); 5] = [
        ("Q4_K ffn_up   [6144, 2048]", 12, 6144, 2048),
        ("Q5_K ffn_up   [6144, 2048]", 13, 6144, 2048),
        ("Q6_K ffn_down [2048, 6144]", 14, 2048, 6144),
        ("Q6_K lm_head  [16384, 2048]", 14, 16384, 2048),
        ("F32  attn_out [2048, 2048]", 0, 2048, 2048),
    ];
    for (label, raw_type, rows, columns) in cases {
        let bytes = if raw_type == 0 {
            f32_rows(rows, columns)
        } else {
            quant_rows(raw_type, rows, columns)
        };
        let shape = [rows, columns];
        let weights = TensorView::new(&bytes, &shape, dtype_of(raw_type));
        let mut state = 7_u64;
        let input: Vec<f32> = (0..columns)
            .map(|_| ((next(&mut state) % 2001) as f32) / 1000.0 - 1.0)
            .collect();
        let mut output = vec![0.0_f32; rows];
        println!("{label}  ({:.1} MB)", bytes.len() as f64 / 1e6);
        for (name, backend) in backends(workers) {
            let seconds = time(budget, || {
                backend
                    .matvec(weights, black_box(&input), &mut output)
                    .unwrap();
                black_box(&output);
            });
            println!(
                "  {name:<10} {:>9.3} ms/op  {:>6.2} GB/s",
                seconds * 1e3,
                bytes.len() as f64 / seconds / 1e9
            );
        }
    }

    println!("\nMatmul (prefill, B is [k, n] f32):");
    let cases: [(&str, u32, usize, usize, usize); 4] = [
        ("F32  [2048,2048] x n=64 ", 0, 2048, 2048, 64),
        ("Q4_K [2048,2048] x n=64 ", 12, 2048, 2048, 64),
        ("Q4_K [2048,2048] x n=256", 12, 2048, 2048, 256),
        ("Q6_K [2048,2048] x n=37 ", 14, 2048, 2048, 37),
    ];
    for (label, raw_type, m, k, n) in cases {
        let bytes = if raw_type == 0 {
            f32_rows(m, k)
        } else {
            quant_rows(raw_type, m, k)
        };
        let shape = [m, k];
        let weights = TensorView::new(&bytes, &shape, dtype_of(raw_type));
        let mut state = 11_u64;
        let b: Vec<f32> = (0..k * n)
            .map(|_| ((next(&mut state) % 2001) as f32) / 1000.0 - 1.0)
            .collect();
        let mut c = vec![0.0_f32; m * n];
        println!("{label}");
        for (name, backend) in backends(workers) {
            let seconds = time(budget, || {
                backend
                    .matmul(weights, black_box(&b), &mut c, m, n, k)
                    .unwrap();
                black_box(&c);
            });
            println!(
                "  {name:<10} {:>9.2} ms/op  {:>6.2} GFLOP/s",
                seconds * 1e3,
                2.0 * (m * n * k) as f64 / seconds / 1e9
            );
        }
    }
}
