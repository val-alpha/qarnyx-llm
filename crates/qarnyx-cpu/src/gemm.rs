//! Register-tiled F32 GEMM micro-kernel used by prompt prefill.
//!
//! Computes C[4][n] = A[4][k] * B[k][n], where A contains four decoded rows.
//! Each B vector is loaded once and reused across four output rows.

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

/// Number of weight rows handled by the micro-kernel.
pub(crate) const TILE_ROWS: usize = 4;

/// Computes four row-major weight rows against a row-major input matrix.
///
/// The caller must have validated the shapes and AVX2+FMA support.
pub(crate) fn tile_4(a: &[f32], k: usize, b: &[f32], n: usize, c: &mut [f32]) {
    assert_eq!(a.len(), TILE_ROWS * k, "A must hold exactly four rows");
    assert_eq!(b.len(), k * n, "B must be [k, n]");
    assert_eq!(c.len(), TILE_ROWS * n, "C must be [4, n]");
    // SAFETY: lengths are validated above and the caller checks AVX2+FMA.
    unsafe { tile_4_impl(a, k, b, n, c) }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn tile_4_impl(a: &[f32], k: usize, b: &[f32], n: usize, c: &mut [f32]) {
    unsafe {
        let mut column = 0;
        while column + 16 <= n {
            kernel::<2>(
                a.as_ptr(),
                k,
                b.as_ptr().add(column),
                n,
                c.as_mut_ptr().add(column),
            );
            column += 16;
        }
        if column + 8 <= n {
            kernel::<1>(
                a.as_ptr(),
                k,
                b.as_ptr().add(column),
                n,
                c.as_mut_ptr().add(column),
            );
        }
    }

    // A scalar tail handles fewer than eight remaining columns.
    for column in (n - n % 8)..n {
        for row in 0..TILE_ROWS {
            let mut sum = 0.0_f32;
            for inner in 0..k {
                sum += a[row * k + inner] * b[inner * n + column];
            }
            c[row * n + column] = sum;
        }
    }
}

/// Computes four rows by 8*NV columns, retaining accumulators in registers.
///
/// # Safety
/// A must point to 4*k floats, B to k rows with at least 8*NV values at stride
/// n, and C to four output rows of length n.
#[target_feature(enable = "avx2,fma")]
unsafe fn kernel<const NV: usize>(a: *const f32, k: usize, b: *const f32, n: usize, c: *mut f32) {
    unsafe {
        let mut accumulators = [[_mm256_setzero_ps(); NV]; TILE_ROWS];
        for inner in 0..k {
            let b_row = b.add(inner * n);
            let mut values = [_mm256_setzero_ps(); NV];
            for (vector, slot) in values.iter_mut().enumerate() {
                *slot = _mm256_loadu_ps(b_row.add(vector * 8));
            }

            for (row, row_accumulators) in accumulators.iter_mut().enumerate() {
                let weight = _mm256_broadcast_ss(&*a.add(row * k + inner));
                for (vector, accumulator) in row_accumulators.iter_mut().enumerate() {
                    *accumulator = _mm256_fmadd_ps(weight, values[vector], *accumulator);
                }
            }
        }

        for (row, row_accumulators) in accumulators.iter().enumerate() {
            for (vector, accumulator) in row_accumulators.iter().enumerate() {
                _mm256_storeu_ps(c.add(row * n + vector * 8), *accumulator);
            }
        }
    }
}
