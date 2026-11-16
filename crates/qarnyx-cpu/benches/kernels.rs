//! Scalar kernel microbenchmarks for F32 and GGUF storage formats.
//!
//! Run with: cargo bench -p qarnyx-cpu --bench kernels

use std::error::Error;
use std::hint::black_box;
use std::time::Instant;

use qarnyx_core::{Backend, DType, TensorView};
use qarnyx_cpu::CpuBackend;
use qarnyx_quant::{encoded_len, type_info};

fn encode_f32(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    let mut bytes = Vec::new();
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Builds finite, deterministic row-major storage without allocating an F32
/// copy for quantized matrix formats.
fn matrix_fixture(raw_type: u32, rows: usize, columns: usize) -> Result<Vec<u8>, Box<dyn Error>> {
    let columns_u64 = u64::try_from(columns)?;
    let row_bytes = usize::try_from(encoded_len(raw_type, &[columns_u64])?)?;
    let info = type_info(raw_type).ok_or("unsupported benchmark tensor type")?;
    let block_bytes = usize::try_from(info.type_size)?;
    let total_bytes = row_bytes
        .checked_mul(rows)
        .ok_or("benchmark matrix byte length overflow")?;
    let mut bytes = vec![0_u8; total_bytes];

    for block in bytes.chunks_exact_mut(block_bytes) {
        match raw_type {
            0 => block.copy_from_slice(&0.125_f32.to_le_bytes()),
            1 => block.copy_from_slice(&0x3c00_u16.to_le_bytes()),
            2 => {
                block[..2].copy_from_slice(&0x3800_u16.to_le_bytes());
                block[2..].fill(0x98);
            }
            8 => {
                block[..2].copy_from_slice(&0x3800_u16.to_le_bytes());
                block[2..].fill(2);
            }
            12 => {
                block[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
                block[4..16].fill(1);
                block[16..].fill(0x11);
            }
            13 => {
                block[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
                block[4..16].fill(1);
                block[48..].fill(0x11);
            }
            14 => {
                block[..128].fill(0x11);
                block[128..192].fill(0xaa);
                block[192..208].fill(1);
                block[208..210].copy_from_slice(&0x3c00_u16.to_le_bytes());
            }
            _ => return Err(format!("unsupported benchmark tensor type {raw_type}").into()),
        }
    }

    Ok(bytes)
}

fn benchmark_matvec(
    backend: &CpuBackend,
    label: &str,
    dtype: DType,
    raw_type: u32,
) -> Result<(), Box<dyn Error>> {
    let (m, k) = (128, 512);
    let bytes = matrix_fixture(raw_type, m, k)?;
    let shape = [m, k];
    let weights = TensorView::new(&bytes, &shape, dtype);
    let input = vec![0.25_f32; k];
    let mut output = vec![0.0_f32; m];

    for _ in 0..3 {
        backend.matvec(
            black_box(weights),
            black_box(&input),
            black_box(&mut output),
        )?;
    }

    let iterations = 20;
    let start = Instant::now();
    for _ in 0..iterations {
        backend.matvec(
            black_box(weights),
            black_box(&input),
            black_box(&mut output),
        )?;
    }
    let elapsed = start.elapsed();
    black_box(&output);
    println!(
        "scalar {label} matvec [{m}, {k}]: {iterations} iterations ({:.2} us/op)",
        elapsed.as_secs_f64() * 1.0e6 / iterations as f64
    );
    Ok(())
}

fn benchmark_matmul(
    backend: &CpuBackend,
    label: &str,
    dtype: DType,
    raw_type: u32,
) -> Result<(), Box<dyn Error>> {
    let (m, k, n) = (32, 256, 32);
    let bytes = matrix_fixture(raw_type, m, k)?;
    let shape = [m, k];
    let weights = TensorView::new(&bytes, &shape, dtype);
    let input = vec![0.125_f32; k * n];
    let mut output = vec![0.0_f32; m * n];

    for _ in 0..2 {
        backend.matmul(
            black_box(weights),
            black_box(&input),
            black_box(&mut output),
            m,
            n,
            k,
        )?;
    }

    let iterations = 8;
    let start = Instant::now();
    for _ in 0..iterations {
        backend.matmul(
            black_box(weights),
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
        "scalar {label} matmul [{m}, {k}] x [{k}, {n}]: {iterations} iterations ({:.2} ms/op)",
        elapsed.as_secs_f64() * 1.0e3 / iterations as f64
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let backend = CpuBackend::new(1)?;

    // Keep the original F32 fixtures as a baseline comparable to earlier runs.
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

    // Equal logical dimensions make timings comparable across storage formats.
    let formats = [
        ("F32", DType::F32, 0),
        ("F16", DType::F16, 1),
        ("Q4_0", DType::Q4_0, 2),
        ("Q8_0", DType::Q8_0, 8),
        ("Q4_K", DType::Q4K, 12),
        ("Q5_K", DType::Q5K, 13),
        ("Q6_K", DType::Q6K, 14),
    ];

    println!("\nScalar storage-format comparison (single worker):");
    for (label, dtype, raw_type) in formats {
        benchmark_matvec(&backend, label, dtype, raw_type)?;
    }
    println!();
    for (label, dtype, raw_type) in formats {
        benchmark_matmul(&backend, label, dtype, raw_type)?;
    }

    Ok(())
}
