/* SPDX-License-Identifier: MPL-2.0 */
use crate::Dict;

/** A saturating right shift returns zero if the shift amount overflows the number of bits */
#[inline(always)]
pub fn saturating_shr(a: u64, v: u32) -> u64 {
    if v >= u64::BITS {
        0
    } else {
        a >> v
    }
}
#[inline(always)]
pub fn mask_for_ubits(ubits: u32) -> u64 {
    if ubits == u64::BITS {
        u64::MAX
    } else if ubits > u64::BITS {
        panic!()
    } else {
        (1 << ubits) - 1
    }
}

/** A (weak) pseudo-random permutation indexed by 'seed', mapping `i` to a unique value in [0..n)  */
fn random_permute(i: u64, n: u64, seed: u64) -> u64 {
    // TODO: replace with something better; can probably afford cryptographic quality + high space
    // overhead, since this is only used to set up dictionary values, although evaluation should be fast
    assert!(n.is_power_of_two());
    let p = (seed.wrapping_mul(0x9e3779b97f4a7c16)) | 0x1;
    ((i as u128 * p as u128 + seed as u128) % (n as u128)) as u64
}

fn ith_random_sampled_32(i: u64, seed: u64) -> u64 {
    /* Now 2^32 and p are coprime, so (x*p)%n is a permutation. */
    let p = (seed.wrapping_mul(0x9e3779b97f4a7c16)) | 0x1;
    ((i as u128 * p as u128) % (1u128 << 32)) as u64
}

/** Return a (k,v) sequence with the property that, following the k->v=k->v=k->v ...
 * chain from k[0] will eventually cycle back and report k[0] */
pub fn make_random_chain_32(n: u64, seed_vals: u64, seed_order: u64) -> Vec<(u64, u64)> {
    assert!(n.is_power_of_two());

    (0..n)
        .map(|x| {
            let i1 = random_permute(x, n, seed_order);
            let i2 = random_permute((x + 1) % n, n, seed_order);

            let k = ith_random_sampled_32(i1, seed_vals);
            let v = ith_random_sampled_32(i2, seed_vals);
            (k, v)
        })
        .collect()
}

fn ith_random_sampled_64(i: u64, seed: u64) -> u64 {
    /* Now 2^64 and p are coprime, so (x*p)%2^64 is a permutation. */
    let p = (seed.wrapping_mul(0x9e3779b97f4a7c16)) | 0x1;
    (i as u128 * p as u128) as u64
}

/** Return a (k,v) sequence with the property that, following the k->v=k->v=k->v ...
 * chain from k[0] will eventually cycle back and report k[0] */
pub fn make_random_chain_64(n: u64, seed_vals: u64, seed_order: u64) -> Vec<(u64, u64)> {
    assert!(n.is_power_of_two());

    (0..n)
        .map(|x| {
            let i1 = random_permute(x, n, seed_order);
            let i2 = random_permute((x + 1) % n, n, seed_order);

            let k = ith_random_sampled_64(i1, seed_vals);
            let v = ith_random_sampled_64(i2, seed_vals);
            (k, v)
        })
        .collect()
}

#[test]
fn test_pseudorand_permute() {
    let n = 128;
    let mut v = vec![false; n];
    for i in 0..n {
        v[random_permute(i as u64, n as u64, 123) as usize] = true;
    }
    assert!(v.iter().all(|x| *x));
}

/** Measure the "true", non-amortized latency of the dictionary, by chasing the pointer chain
 * formed by following the key->value=key->value=key->... sequence in the given `chain`.
 * This may be _much_ slower than `parallel_query`.
 *
 * To properly test the dictionary, `chain` must encode a permutation, preferably one whose
 * order is unpredictable to the dictionary implementations.
 */
pub fn chain_query<T: Dict<u64, u64>>(d: &T, chain: &[(u64, u64)]) -> u64 {
    let mut k = chain[0].0;
    for _ in 0..chain.len() {
        k = d.query(k).unwrap();
    }
    assert!(k == chain[0].0);
    k
}

/** A simple approximate throughput test of the dictionary; to make `n` independent queries of
 * values in the dictionary. A sufficiently smart compiler could inline and hide most memory
 * access latency (although throughput & caching limits would still exist); in practice
 * performance may be worse.
 *
 * To properly test the dictionary, `chain`'s keys should be a permutation of the dictionaries
 * keys, preferably one whose order is unpredictable to the dictionary implementations.
 */
pub fn parallel_query<T: Dict<u64, u64>>(d: &T, chain: &[(u64, u64)]) -> u64 {
    let mut x = 0;
    let mut y = 0;
    for (k, v) in chain {
        x += d.query(*k).unwrap();
        y += v;
    }
    assert!(x == y);
    x
}

#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
unsafe fn clmul_x86(a: u64, b: u64) -> u128 {
    use std::arch::x86_64::*;
    let ma = _mm_set_epi64x(0, a as i64);
    let mb = _mm_set_epi64x(0, b as i64);
    let y = _mm_clmulepi64_si128::<0>(ma, mb);
    ((_mm_extract_epi64::<0>(y) as u64) as u128)
        | (((_mm_extract_epi64::<1>(y) as u64) as u128) << 64)
}

#[cfg(any(not(all(target_arch = "x86_64", target_feature = "pclmulqdq")), test))]
fn clmul_generic(a: u64, b: u64) -> u128 {
    // note: compilers might be able to figure out this idiom
    let mut v = 0_u128;
    for i in 0..64 {
        v ^= ((a as u128) & (1 << i)) * (b as u128);
    }
    v
}
#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
pub fn clmul(a: u64, b: u64) -> u128 {
    unsafe { clmul_x86(a, b) }
}
#[cfg(not(all(target_arch = "x86_64", target_feature = "pclmulqdq")))]
pub fn clmul(a: u64, b: u64) -> u128 {
    clmul_generic(a, b)
}
#[cfg(target_arch = "x86_64")]
#[test]
fn test_clmul() {
    for i in 0..100 {
        for j in 0..100 {
            let f = 1e19 as u64;
            assert!(
                clmul(f.wrapping_mul(i), f.wrapping_mul(j))
                    == clmul_generic(f.wrapping_mul(i), f.wrapping_mul(j))
            );
        }
    }
}

#[cfg(any(not(all(target_arch = "x86_64", target_feature = "bmi2")), test))]
/* Extract the bits in `val` at positions in `mask` to the lowest order bits */
fn pext_generic(val: u128, mask: u128) -> u128 {
    /* Lemma 3.3 suggests a fancy multiply+subhash trick (because the paper
     * needs "constant-time" in its RAM model, and this step only needs to preserve
     * uniqueness); modern CPUs have `pext` or equivalent. */
    let mut x = 0;
    let mut j = 0;
    for i in 0..128 {
        if mask & (1 << i) != 0 {
            x |= (val & (1 << i)) >> (i - j);
            j += 1;
        }
    }
    x
}

#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
unsafe fn pext_x86(val: u128, mask: u128) -> u128 {
    use std::arch::x86_64::*;
    let low = _pext_u64(val as u64, mask as u64);
    let high = _pext_u64((val >> 64) as u64, (mask >> 64) as u64);
    let step = (mask as u64).count_ones();
    (low as u128) | ((high as u128) << step)
}
#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
pub fn pext(val: u128, mask: u128) -> u128 {
    unsafe { pext_x86(val, mask) }
}
#[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
pub fn pext(val: u128, mask: u128) -> u128 {
    pext_generic(val, mask)
}
#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
#[test]
fn test_pext() {
    for i in 0..100 {
        for j in 0..100 {
            let f = 1e38 as u128;
            assert!(
                pext(f.wrapping_mul(i), f.wrapping_mul(j))
                    == pext_generic(f.wrapping_mul(i), f.wrapping_mul(j))
            );
        }
    }
}
