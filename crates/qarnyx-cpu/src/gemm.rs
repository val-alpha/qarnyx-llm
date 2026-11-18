//! Register-tiled F32 GEMM micro-kernel used by prompt prefill.
//!
//! Computes `C[4][n] = A[4][k] * B[k][n]` where `A` holds four already-decoded
//! weight rows. Each `B` row segment is loaded once and reused for four weight
//! rows, so `B` is streamed from cache `m/4` times instead of `m` times, and
//! every loaded vector feeds four FMAs instead of one.

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

/// Rows of `A` handled per call.
pub(crate) const TILE_ROWS: usize = 4;

/// `c = a * b` for four rows. `a` is `[4, k]`, `b` is `[k, n]`, `c` is `[4, n]`,
/// all row-major. The caller must have verified AVX2+FMA support.
pub(crate) fn tile_4(a: &[f32], k: usize, b: &[f32], n: usize, c: &mut [f32]) {
    assert_eq!(a.len(), TILE_ROWS * k, "A must hold exactly four rows");
    assert_eq!(b.len(), k * n, "B must be [k, n]");
    assert_eq!(c.len(), TILE_ROWS * n, "C must be [4, n]");
    // SAFETY: lengths validated above; callers verified AVX2+FMA.
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

    // Fewer than eight columns remain: scalar.
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

/// 4 rows x (8 * NV) columns, accumulated in registers over the whole `k`.
///
/// # Safety
/// `a` must address `4 * k` floats, `b` must address `k` rows of at least
/// `8 * NV` floats at stride `n`, and `c` four rows of `8 * NV` floats at stride `n`.
#[target_feature(enable = "avx2,fma")]
unsafe fn kernel<const NV: usize>(a: *const f32, k: usize, b: *const f32, n: usize, c: *mut f32) {
    unsafe {
        let mut acc = [[_mm256_setzero_ps(); NV]; TILE_ROWS];
        for inner in 0..k {
            let b_row = b.add(inner * n);
            let mut values = [_mm256_setzero_ps(); NV];
            for (vector, slot) in values.iter_mut().enumerate() {
                *slot = _mm256_loadu_ps(b_row.add(vector * 8));
            }
            for (row, row_acc) in acc.iter_mut().enumerate() {
                let weight = _mm256_broadcast_ss(&*a.add(row * k + inner));
                for (vector, slot) in row_acc.iter_mut().enumerate() {
                    *slot = _mm256_fmadd_ps(weight, values[vector], *slot);
                }
            }
        }
        for (row, row_acc) in acc.iter().enumerate() {
            for (vector, slot) in row_acc.iter().enumerate() {
                _mm256_storeu_ps(c.add(row * n + vector * 8), *slot);
            }
        }
    }
}
