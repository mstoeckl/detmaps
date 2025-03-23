use std::collections::BTreeSet;

/* SPDX-License-Identifier: MPL-2.0 */
use crate::Dict;
use rand::{seq::SliceRandom, Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;
use rand_distr::{Binomial, Distribution, StandardUniform};

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

    // TODO: this is an odd-multiply-mod construction, and it is not obvious that this does not interact weirdly with odd-multiply-shift hashing

    assert!(n.is_power_of_two());
    let p = (seed.wrapping_mul(0x9e3779b97f4a7c16)) | 0x1;
    ((i as u128 * p as u128 + seed as u128) % (n as u128)) as u64
}

fn ith_random_sampled_32(i: u64, seed: u64) -> u64 {
    /* Now 2^32 and p are coprime, so (x*p)%n is a permutation. */
    let p = (seed.wrapping_mul(0x9e3779b97f4a7c16)) | 0x1;
    ((i as u128 * p as u128) % (1u128 << 32)) as u64
}

/** Produces a sorted, uniformly random (assuming a perfect RNG) distribution
 * over fixed size subsets of a given range.
 *
 * The `rand` crate's rand::seq::index::sample unnecessarily allocates,
 * the allocation-free rand::seq::index::sample_array is quadratic, and
 * neither can provide subsets of integers larger than usize::MAX.
 */
struct RandomSubsetIterator<'a> {
    start: u64,
    end_incl: u64,
    count: u64,
    rng: &'a mut ChaCha12Rng,
}

impl RandomSubsetIterator<'_> {
    fn new(start: u64, end_incl: u64, count: u64, rng: &mut ChaCha12Rng) -> RandomSubsetIterator {
        assert!(end_incl >= start);
        if count > 0 {
            assert!(
                end_incl - start >= count - 1,
                "{}..={} {}",
                start,
                end_incl,
                count
            );
        }
        RandomSubsetIterator {
            start,
            end_incl,
            count,
            rng,
        }
    }
}
impl Iterator for RandomSubsetIterator<'_> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.count == 0 {
            return None;
        }

        /* Implementation of Vitter 1984's Algorithm D (using existing distribution samplers,
         * as recommended by Ting 2021's "Simple, Optimal Algorithms for Random Sampling Without
         * Replacement"). This isn't particularly fast, but   */

        /* Sample the position of the first element in the subset */
        let mut nmk = (self.end_incl - self.start) - (self.count - 1);

        // TODO: directly implement binomial sampling, because rand_distr's Binomial
        // can break when floats are produced that don't fit in `i64`. Also, this way we can
        // ensure the calculations are valid and accurate even when operating up to
        // u128::MAX or larger.

        /* Beta distribution sampling in general slow, but can be optimized since alpha=1 is fixed.
         *
         * The form 1-u^(1/k) can be very inaccurate when k is large and u^(1/k) is rounded to values near 1.
         * exp_m1(u / k) gives a more numerically stable calculation of `p`, which should be relatively
         * accurate even if count >= 2^128.
         *
         * This is at least twice as fast as explicitly sampling from a rand_distr::Beta distribution.
         *
         * If `u` is zero, u.ln() _should_ return -inf, which expm1 should map to -1.
         */

        let u: f64 = StandardUniform::default().sample(self.rng); // note: this uses 2^53 equispaced values, and never produces tiny (e.g. 2^-80) values
                                                                  // issue: `exp_m1` and `log` are, per standard library, non-deterministic and may even vary
                                                                  // between _executions_.
        let p = -f64::exp_m1(f64::ln(u) * (1. / (self.count as f64)));

        /* Binomial::sample from rand_distr 0.5.1 currently has a panic issue when nmk is
         * close to u64::MAX.
         *
         * A simple and easy way to reduce the domain for sample invocations is to add
         * together multiple binomial samples from smaller distributions; this increases
         * the sampling cost proportionally.
         *
         * NOTE: Binomial (which samples floating point values) is affected by floating
         * point quantization and does not properly sample odd values when n is large.
         * The workaround (of unclear accuracy) is to ensure one of te samples being added
         * has magnitude 1<<53 or smaller.
         */
        let mut offset = 0;
        let first_n = 1u64 << 50;
        if nmk > first_n {
            let bin = Binomial::new(first_n, p).unwrap();
            offset += bin.sample(self.rng);
            nmk -= first_n;
        }
        let max_n = 1u64 << 62; // 1<<63 is too large
        while nmk > max_n {
            let bin = Binomial::new(max_n, p).unwrap();
            offset += bin.sample(self.rng);
            nmk -= max_n;
        }
        let bin = Binomial::new(nmk, p).unwrap();
        offset += bin.sample(self.rng);

        let first_pos = self.start + offset;

        self.start = first_pos + 1;
        self.count -= 1;
        Some(first_pos)
    }
}

#[test]
fn test_random_subset() {
    let max = (1u64 << 48) - 1;
    let mut rng = ChaCha12Rng::seed_from_u64(13);
    let start = std::time::Instant::now();
    /* Takes ~8 seconds in `test` mode for 1<<20 elements,
     * and ~180 seconds in `release` mode for 1<<30 elements. */
    for params in [
        (0, 10, 0),              /* empty set */
        (0, 0, 1),               /* simple set */
        (0, 100, 100),           /* almost sample full set */
        (1 << 20, max, 1 << 16), /* large sampling */
    ] {
        let mut last_entry = None;
        for entry in RandomSubsetIterator::new(params.0, params.1, params.2, &mut rng) {
            if let Some(l) = last_entry {
                assert!(entry > l, "{} {}", l, entry);
            }
            last_entry = Some(entry);
        }
    }

    let mid = std::time::Instant::now();
    std::hint::black_box(make_random_chain_64(1 << 20, 1, 2));
    let end = std::time::Instant::now();
    println!(
        "{} {}",
        mid.duration_since(start).as_secs_f64(),
        end.duration_since(mid).as_secs_f64()
    );
}

/** Return a (k,v) sequence with the property that, following the k->v=k->v=k->v ...
 * chain from k[0] will eventually cycle back and report k[0]. All entries are < 1<<u_bits */
pub fn make_random_chain(n: u64, u_bits: u32, seed_vals: u64, seed_order: u64) -> Vec<(u64, u64)> {
    /* Note: seeds are not guaranteed to produce consistent outputs between architectures
     * due to allowable differences in floating point behavior; library improvements for
     * sampling routines may also affect the pattern.
     */
    let mut rng_vals = ChaCha12Rng::seed_from_u64(seed_vals);
    let mut rng_order = ChaCha12Rng::seed_from_u64(seed_order);

    let mut data = Vec::new();
    data.reserve_exact(n.try_into().unwrap());
    data.resize(n as usize, (0, 0));

    assert!(u_bits <= u64::BITS);
    let max_val = if u_bits == u64::BITS {
        u64::MAX
    } else {
        (1 << u_bits) - 1
    };

    /* Fill with random distinct elements */
    for (i, entry) in RandomSubsetIterator::new(0, max_val, n, &mut rng_vals).enumerate() {
        data[i].0 = entry;
    }
    /* Randomize order */
    data.shuffle(&mut rng_order);
    /* Build chain */
    let mut prev = data.last().unwrap().0;
    for x in data.iter_mut() {
        x.1 = prev;
        prev = x.0;
    }
    /* Randomize order of chain */
    data.shuffle(&mut rng_order);
    data
}

/** Return a (k,v) sequence with the property that, following the k->v=k->v=k->v ...
 * chain from k[0] will eventually cycle back and report k[0] */
pub fn make_random_chain_64(n: u64, seed_vals: u64, seed_order: u64) -> Vec<(u64, u64)> {
    let mut rng_vals = ChaCha12Rng::seed_from_u64(seed_vals);
    let mut rng_order = ChaCha12Rng::seed_from_u64(seed_order);

    // note: rand's index::sample works for u32, but requires allocation and does not
    // work for integer types >= usize.
    // TODO: implement the near linear-time, zero overhead, sorted subset sampling algorithm
    // Another approach would be to use a switching/shuffling network, but they are often slow
    assert!(n <= (u64::MAX / 2) as u64); /* algorithm is only somewhat fast when sample size is <=1/2 universe */
    let mut s = BTreeSet::<u64>::new();
    while s.len() < n as usize {
        let x = rng_vals.random_range(0..=(u64::MAX as u64));
        s.insert(x);
    }
    let mut v: Vec<u64> = s.into_iter().collect();
    v.shuffle(&mut rng_order);
    let mut chain: Vec<(u64, u64)> = Vec::new();
    let wrap_pair = (*v.last().unwrap(), *v.first().unwrap());
    chain.push(wrap_pair);
    for w in v.windows(2) {
        chain.push((w[0], w[1]));
    }
    chain.shuffle(&mut rng_order);
    chain
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
pub fn chain_query<T: Dict<u64, u64>>(d: &T, steps: u64, start: u64) -> u64 {
    let mut k = start;
    for _ in 0..steps {
        k = d.query(k).unwrap();
    }
    assert!(k == start);
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
