//! Dependency-free scalar kernel microbenchmarks.
//!
//! Run with: cargo bench -p qarnyx-cpu --bench kernels

use std::error::Error;
use std::hint::black_box;
use std::time::Instant;

use qarnyx_core::{Backend, DType, TensorView};
use qarnyx_cpu::CpuBackend;

fn encode_f32(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    let mut bytes = Vec::new();
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn main() -> Result<(), Box<dyn Error>> {
    let backend = CpuBackend::new(1)?;

    let (m, k) = (256, 512);
    let weights = encode_f32((0..m * k).map(|index| (index % 31) as f32 / 31.0 - 0.5));
    let shape = [m, k];
    let view = TensorView::new(&weights, &shape, DType::F32);
    let input = vec![0.25_f32; k];
    let mut output = vec![0.0_f32; m];

    let iterations = 30;
    let start = Instant::now();
    for _ in 0..iterations {
        backend.matvec(black_box(view), black_box(&input), black_box(&mut output))?;
    }
    let elapsed = start.elapsed();
    black_box(&output);
    println!(
        "scalar F32 matvec [{m}, {k}]: {iterations} iterations in {elapsed:?} ({:.2} us/op)",
        elapsed.as_secs_f64() * 1.0e6 / iterations as f64
    );

    let (m, k, n) = (96, 128, 64);
    let weights = encode_f32((0..m * k).map(|index| (index % 23) as f32 / 23.0 - 0.5));
    let shape = [m, k];
    let view = TensorView::new(&weights, &shape, DType::F32);
    let input = vec![0.125_f32; k * n];
    let mut output = vec![0.0_f32; m * n];

    let iterations = 10;
    let start = Instant::now();
    for _ in 0..iterations {
        backend.matmul(
            black_box(view),
            black_box(&input),
            black_box(&mut output),
            m,
            n,
            k,
        )?;
    }
    let elapsed = start.elapsed();
    black_box(&output);
    println!(
        "scalar F32 matmul [{m}, {k}] x [{k}, {n}]: {iterations} iterations in {elapsed:?} ({:.2} ms/op)",
        elapsed.as_secs_f64() * 1.0e3 / iterations as f64
    );

    Ok(())
}
