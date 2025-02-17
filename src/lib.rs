/* SPDX-License-Identifier: MPL-2.0 */
use std::collections::{BTreeMap, HashMap};
use std::hash::Hash;

pub trait Dict<K, V> {
    fn new(data: &[(K, V)]) -> Self;
    fn query(&self, key: K) -> Option<V>;
}

/* ---------------------------------------------------------------------------- */

/** A very simple and space efficient data structure: to build, produce a sorted array
 * of (key,value) pairs, and to query, use binary search. However, performance drops
 * significantly once most queries fall outside the CPU caches or when cache associativity
 * limits are hit, since O(log n) cache lines need to be loaded.
 *
 * (This can be improved through prefetching, and by changing the memory layout to e.g. that
 * of Eytzinger, of a B-tree, or of van Emde Boas.) */
pub struct BinSearchDict<K, V>
where
    K: Ord + Clone,
    V: Clone,
{
    data: Vec<(K, V)>,
}

impl<K: Ord + Clone, V: Clone> Dict<K, V> for BinSearchDict<K, V> {
    fn new(data: &[(K, V)]) -> BinSearchDict<K, V> {
        let mut d: Vec<(K, V)> = Vec::from_iter(data.iter().map(|x| (x.0.clone(), x.1.clone())));
        d.sort_by_key(|x| x.0.clone());
        BinSearchDict { data: d }
    }
    fn query(&self, key: K) -> Option<V> {
        if let Ok(index) = self.data.binary_search_by_key(&key, |x| x.0.clone()) {
            Some(self.data[index].1.clone())
        } else {
            None
        }
    }
}

/* ---------------------------------------------------------------------------- */

/** Rust's BTreeMap. Deterministic, O(log n) query time but with better constants. */
pub struct BTreeDict<K, V>
where
    K: Ord + Clone,
    V: Clone,
{
    data: BTreeMap<K, V>,
}

impl<K: Ord + Clone, V: Clone> Dict<K, V> for BTreeDict<K, V> {
    fn new(data: &[(K, V)]) -> BTreeDict<K, V> {
        BTreeDict {
            data: BTreeMap::from_iter(data.iter().map(|x| (x.0.clone(), x.1.clone()))),
        }
    }
    fn query(&self, key: K) -> Option<V> {
        self.data.get(&key).cloned()
    }
}

/* ---------------------------------------------------------------------------- */

/** Rust's HashMap. Randomized, uses a PRF (SipHash-1-3). */
pub struct HashDict<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    data: HashMap<K, V>,
}

impl<K: Hash + Eq + Clone, V: Clone> Dict<K, V> for HashDict<K, V> {
    fn new(data: &[(K, V)]) -> HashDict<K, V> {
        HashDict {
            data: HashMap::from_iter(data.iter().map(|x| (x.0.clone(), x.1.clone()))),
        }
    }
    fn query(&self, key: K) -> Option<V> {
        self.data.get(&key).cloned()
    }
}

/* ---------------------------------------------------------------------------- */

pub struct HagerupMP01Dict {
    /** Mask indicating which bits of the encoded input to use. */
    distinguishing_bits: u128,
    /** Number of bits for steps to hash down to */
    r_bits: u32,
    /** Instead of using an implicit trie indexed by this hash, just use the hash
     * directly to reduce the input by iteratively mapping [2 log(n)]->[log n] bits */
    hashes: Vec<HMPQuadHash>,
    /** Table of key-value pairs indexed by the perfect hash construction */
    table: Vec<(u64, u64)>,
    /** A value which is not a valid key */
    non_key: u64,
}

#[cfg(target_arch = "x86_64")]
unsafe fn clmul_x86(a: u64, b: u64) -> u128 {
    use std::arch::x86_64::*;
    let ma = _mm_set_epi64x(0, a as i64);
    let mb = _mm_set_epi64x(0, b as i64);
    let y = _mm_clmulepi64_si128::<0>(ma, mb);
    ((_mm_extract_epi64::<0>(y) as u64) as u128)
        | (((_mm_extract_epi64::<1>(y) as u64) as u128) << 64)
}

#[cfg(any(not(target_arch = "x86_64"), test))]
fn clmul_generic(a: u64, b: u64) -> u128 {
    // note: compilers might be able to figure out this idiom
    let mut v = 0_u128;
    for i in 0..64 {
        v ^= ((a as u128) & (1 << i)) * (b as u128);
    }
    v
}
#[cfg(target_arch = "x86_64")]
pub fn clmul(a: u64, b: u64) -> u128 {
    unsafe { clmul_x86(a, b) }
}
#[cfg(not(target_arch = "x86_64"))]
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

/** Error correcting code from u32->u128 with ẟ = 30/128. */
fn error_correcting_code(x: u32) -> u128 {
    /* Per Prop 3.1 of HagerupMP01, a random (2,2)-universal function
     * is an error corecting code of relative distance ẟ with some
     * probability; however, the bounds do not guarantee the existence
     * of a good [w]->[4w] code when ẟ ⪆ 0.105. Meanwhile, the Gilbert-Varshamov
     * bound implies that the asympototic best relative distance linear
     * codes at rate 1/4 is ⪆ H^-1(3/4)~0.213; the Plotkin bound(?) implies
     * the best distance at rate 1/4 is ≤(1-R)/2=3/8. MRRW gives via
     * KalaiLineal95, R <= H(1/2 - sqrt(ẟ(1-ẟ))), which implies at
     * R=1/4 we have 1/2-sqrt(ẟ(1-ẟ)) >= 0.0414 hence
     * ẟ <= 1/2 - 1/2 sqrt(1 - 4 * (1/2-0.0414)^2) = 0.301 (but this may
     * be _asymptotic_).
     *
     * Per M. Grassl, "Bounds on the minimum distance of linear codes and quantum codes.",
     * https://www.codetables.de, the best minimum distance for linear:
     * dimension 16, length 32 codes is 8/32.
     * dimension 16, length 64 codes is 24/64.
     * dimension 16, length 128 codes is between 52/128 and 56/128.
     * dimension 32, length  64 codes is betweeen 12/64 and 16/64.
     * dimension 32, length 128 codes is betweeen 36/128 and 46/128.
     * dimension 32, length 256 codes is betweeen 96/256 and 106/256.
     * dimension 64, length 256 codes is betweeen 65/256 and 90/256.
     * dimension 128, length 256 codes is between 38/256 and 56/256.
     *
     * Also: Lemma 3.1 and following logic in fact work for _any_ width;
     * it just happens that longer outputs tend to have better relative
     * distance, so we can trade time doing universe reduction, in both
     * preprocessing and query phases, for number of lookups during the
     * query phase. (distance ẟ yields outputs in n^{-2 / log(1-ẟ)}, so
     * ẟ=1/2 maps outputs into [n^2]; and ẟ=1-2^{-2/k} gets [n^k].
     */

    /* Instead of using the suggested integer multiplicative code, use
     * a construction based on carryless multiplication, for which the
     * minimum distance with any particular set of constants can be
     * computed in O(2^w) time and O(1) space (or even faster with specific
     * algorithms)); this is practical for _short_ keys; long keys (like
     * 256-bit hashes) will need a different construction (e.g. expander code).
     *
     * This one gets ẟ=30/128.
     */
    let k1 = 4989641109508346571;
    let k2 = 3213968901338125373;
    clmul(x as u64, k1) ^ (clmul(x as u64, k2) << 64)
}

/** Given an array of encoded values, produce a small bitmask indicating
 * positions that distinguish them all.)
 *
 * As the number of bits found may be up to 2 log(n) / log(1/(1-ẟ)) ~= 6 log n,
 * this step can be skipped in practice if e.g. n >= 2^(w/6). For u64 keys,
 * 2^{w/6}≈1626 is small enough for binary or linear search to be entirely
 * in cache and thus much faster.
 *
 * The "word" RAM model is somewhat misleading for computers with SIMD; at any rate,
 * the runtime in a 'bit-RAM' model is O(n polylog n poly w), with (if properly
 * implemented) a "multiple scan" type access pattern using O(log n) scans.
 */
#[inline(never)]
fn find_distinguishing_positions(input: &[u128]) -> u128 {
    fn comb2(x: u32) -> u64 {
        // When x=0, x-1 overflows, but the result is still 0
        (x as u64) * ((x.wrapping_sub(1)) as u64) / 2
    }

    assert!(input.len() <= u32::MAX as usize);
    // The paper's design is fairly complicated, using variable sub field
    // width tricks to store counters for each bit. Instead of going fully variable,
    // consider writing special cases for u8/u16/u32/u64 counter sizes, and using SIMD
    // for each. Also: it may be possible to tweak Lemma 3.1's badness metric to
    // use x^2 instead of x(x-1)/2

    // As 128*4 is just 512B, brute force should work OK for a first try.

    // The paper recommends linked lists for the clusters; Vec<u128> works just
    // as well. Doing an incremental radix sort, bit by bit, may be more efficient,
    // although handling the different cluster size classes could get complicated.
    let mut clusters: Vec<Vec<u128>> = vec![Vec::from(input)];
    let mut d = 0_u128;

    if input.len() == 1 {
        /* No nontrivial clusters, so no need to process them or find any distinguishing bit */
        return 0;
    }

    // Expected number of iterations: O(log n)
    while !clusters.is_empty() {
        // Measure badness for each possible new bit to add
        let mut total_improvement = [0_u64; 128];
        let mut net_badness = 0_u64;

        for cluster in &clusters {
            let mut ones_by_index = [0_u32; 128];
            let cn = cluster.len() as u32;
            if true {
                /* A slightly optimized calculation; operate in small batches for which
                 * the intermediate counter sizes are small (only 128 bytes) and
                 * hopefully more SIMD-amenable, periodically updating the larger counters. */
                for chunk in cluster.chunks(128) {
                    let mut ones_u8 = [0_u8; 128];
                    // This is the hot loop (takes maybe ~80% of function time on one test input)
                    for x in chunk {
                        for (i, c) in ones_u8.iter_mut().enumerate() {
                            *c += (x & (1 << i) != 0) as u8;
                        }
                    }
                    for (x32, x8) in ones_by_index.iter_mut().zip(ones_u8.iter()) {
                        *x32 += *x8 as u32;
                    }
                }
            } else {
                for x in cluster {
                    for (i, o) in ones_by_index.iter_mut().enumerate() {
                        *o += (x & (1 << i) != 0) as u32;
                    }
                }
            }

            net_badness += comb2(cn);
            for i in 0..128 {
                /* The improvement is: comb(a+b,2) - comb(a, 2) + comb(b,2) = a*b */
                total_improvement[i] +=
                    (ones_by_index[i] as u64) * ((cn - ones_by_index[i]) as u64);
            }
        }

        // Choose the best bit
        let best_bit = total_improvement
            .iter()
            .enumerate()
            .max_by_key(|(_i, c)| *c)
            .unwrap()
            .0;
        d |= 1 << best_bit;

        // If ẟ >= 0.10, the badness _should_ shrink to 9/10 of its original value.
        // (This assertion may fail if the input data has duplicates.)
        assert!(
            total_improvement[best_bit] >= net_badness.div_ceil(10),
            "{:?} {} {}",
            total_improvement,
            total_improvement.iter().min().unwrap(),
            net_badness
        );

        // Note: avoiding these allocations and just manipulating fixed arrays
        // may give a ~5% speedup.
        let mut new_clusters = Vec::new();
        for cluster in &clusters {
            let mut cluster_0 = Vec::new();
            let mut cluster_1 = Vec::new();
            for x in cluster {
                if x & (1 << best_bit) != 0 {
                    cluster_0.push(*x);
                } else {
                    cluster_1.push(*x);
                }
            }
            // Minor optimization: drop clusters of length 1, already fully split
            if cluster_0.len() >= 2 {
                new_clusters.push(cluster_0);
            }
            if cluster_1.len() >= 2 {
                new_clusters.push(cluster_1);
            }
        }
        clusters = new_clusters;
    }

    d
}

#[test]
fn test_hmp01_universe_reduction() {
    let mut x: Vec<u32> = (0..1000).map(|x| x * x).collect();
    x.sort_unstable();
    for w in x.windows(2) {
        assert!(w[0] != w[1]);
    }
    let y: Vec<u128> = x.iter().map(|a| error_correcting_code(*a)).collect();
    println!("y[0]: {:0128b}", y[0]);
    println!("y[1]: {:0128b}", y[1]);
    // As of writing, this is very slow (30 seconds/1e6 elements)
    let d = find_distinguishing_positions(&y);
    println!("d: {:0128b}, popcount {}", d, d.count_ones());
    let mut z: Vec<u128> = y.iter().map(|a| pext(*a, d)).collect();
    z.sort_unstable();
    for w in z.windows(2) {
        assert!(w[0] != w[1]);
    }
}

/** A perfect hash construction from [n^2] to [n], where n is a power of 2,
 * and n <= u64::MAX / 8. Using the derandomized double displacement design of HMP01.
 *
 * Q: does any sort of "triple displacement" work, and would it improve the input range?
 *
 * Q: does the algorithm still work if the (f,g) pairs contain duplicates? (or does this
 * distort the search process too much)
 *
 * Q: in the second displacement round, we will not have any collisions; so (assuming
 * duplicates have been removed) the values in m are guaranteed to be bounded as a function
 * of the level; if tightly packing counters in levels, `m` could be stored using just
 * `2^r` _bits_, in total.
 */
struct HMPQuadHash {
    r_bits: u32, // need r<=64, and (to be effective) r >= 8n

    // displacement tables: size 2^r, output range 2^r
    disp_0: Vec<u64>,
    disp_1: Vec<u64>,
}

/** Perform a displacement round (Lemma 4.1).
 *
 * Note: the deterministic approach actually gives slightly stronger results
 * (sub-average) than the randomized one with Markov (sub-2x-average). Consider
 * redoing parameters if this approach proves remotely competitive.
 *
 * Also: the second displacement round _should_ always find slots which previously
 * had weight zero; it may be possible to find a better data structure for this
 * special case. (Note: Ružić09 appears to have dealt with a similar problem.)
 */
#[inline(never)]
fn compute_displacement(input: &[(u64, u64)], r: u32) -> Vec<u64> {
    let mut bucket_sizes = vec![0_u64; 1 << r];
    for (f, _g) in input {
        bucket_sizes[*f as usize] += 1;
    }
    /* Since we will be reorganizing data anyway, place _large_ buckets first */
    // How often each frequency of bucket occurs; max frequency is of course 'n'
    let nsize_classes = input.len() + 1;
    let mut bucket_size_frequencies = vec![0_u64; nsize_classes];
    for c in bucket_sizes.iter() {
        bucket_size_frequencies[*c as usize] += 1;
    }
    /* Compute offsets for the _start_ of each collection of buckets in the size class */
    let mut bucket_size_class_offsets = bucket_size_frequencies;
    let mut offset = 0;
    for (s, c) in bucket_size_class_offsets.iter_mut().enumerate().rev() {
        let n_bucket = *c;
        *c = offset;
        offset += n_bucket * s as u64;
    }
    /* Locate starts of buckets */
    let mut bucket_offsets = bucket_sizes;
    for c in bucket_offsets.iter_mut() {
        let bucket_size = *c;
        *c = bucket_size_class_offsets[bucket_size as usize];
        bucket_size_class_offsets[bucket_size as usize] += bucket_size;
    }
    /* Place (f,g) pairs into buckets (ensuring contiguous layout for later use) */
    /* Note: in theory, could pull bucket contents on demand using an earlier indexing
     * step, but this would probably not be a large improvement (or might even work worse). */
    let mut fg = vec![(0_u64, 0_u64); input.len()];
    for (f, g) in input {
        fg[bucket_offsets[*f as usize] as usize] = (*f, *g);
        bucket_offsets[*f as usize] += 1;
    }
    /* Now 'bucket_offsets[i]' indicates the _end_ of the `i`th bucket, sorted by size */

    /* Eytzinger layout for m: leaf entries at 1<<r..1<<(r+1); root at
     * depth 0/position 1; depth i in range 1<<i..1<<(i+1). */
    let mut m = vec![0_u64; 1 << (r + 1)]; // maximum value of 'm' is 'n', at the root
    let mut disp = vec![0; (1 << r).try_into().unwrap()];

    /* Process buckets in decreasing order of size. */
    let mut last_sc_offset = 0;
    for (class_sz, class_end) in bucket_size_class_offsets.iter().enumerate().rev() {
        let class_start = last_sc_offset;
        last_sc_offset = *class_end;

        if class_start == *class_end {
            // Most buckets size classes are empty and the loop will (quickly) skip them
            continue;
        }
        assert!(class_start < *class_end);

        let mut bucket_offset = class_start as usize;
        while bucket_offset < *class_end as usize {
            let entries = &fg[bucket_offset..bucket_offset + class_sz];
            let f = entries[0].0; // will be the same for all bucket entries

            /* Determine the displacement to use for `f` */
            /* Note: it is possible to pick more than one bit at a time, if the table/tree `m`
             * is laid out so that u is built from the _highest_ bits down; then table entries
             * (g&msk)^(u|0x00), (g&msk)^(u|0x01), (g&msk)^(u|0x10), etc. will be in the same
             * cache line. */
            let mut u = 0;
            for i in 0..r {
                let level = &m[(2 << i) as usize..(4 << i) as usize];
                let shift = r - i - 1;
                let u0 = u << 1;
                let u1 = (u << 1) | 1;

                let mut zero_wt = 0;
                let mut one_wt = 0;
                for (_f, g) in entries {
                    zero_wt += level[((g >> shift) ^ u0) as usize];
                    one_wt += level[((g >> shift) ^ u1) as usize];
                }

                // println!("i {} u {} level len {} {} {}", i, u, level.len(), zero_wt, one_wt);
                // Choose the next bit value for which the weight is sub-average
                if one_wt < zero_wt {
                    u = u1;
                } else {
                    u = u0;
                }
            }

            disp[f as usize] = u;

            /* Update `m` */
            for i in 0..=r {
                let level = &mut m[(1 << i) as usize..(2 << i) as usize];
                let shift = r - i;
                // println!("i {} u {} shift {} level len {}", i, u, shift, level.len());
                for (_f, g) in entries {
                    level[((g ^ u) >> shift) as usize] += 1;
                }
            }

            // println!(
            //     "bucket {}, {}, value of u is {}",
            //     bucket_offset, class_sz, u
            // );
            // for (f, g) in &fg[bucket_offset..bucket_offset + class_sz] {
            //     let end = &m[(1 << r) as usize..(2 << r) as usize];
            //     println!("{} {} -> {} @ {}", f, g, g ^ u, end[(g^u) as usize]);
            // }

            bucket_offset += class_sz;
        }
    }

    disp
}

impl HMPQuadHash {
    fn new(keys: &[(u64, u64)], r_bits: u32) -> HMPQuadHash {
        // Ensure r_bits <= 63 to avoid weird edge cases
        assert!(r_bits <= 64 - 4);
        assert!(2 * keys.len() <= (1 << r_bits));
        for (f, g) in keys {
            assert!(f >> r_bits == 0);
            assert!(g >> r_bits == 0);
        }
        let disp_0 = compute_displacement(keys, r_bits);
        let disp_keys: Vec<(u64, u64)> = keys
            .iter()
            .map(|(f, g)| (*g ^ disp_0[*f as usize], *f))
            .collect();
        let disp_1 = compute_displacement(&disp_keys, r_bits);

        HMPQuadHash {
            r_bits,
            disp_0,
            disp_1,
        }
    }

    fn query(&self, f: u64, g: u64) -> u64 {
        f ^ self.disp_1[(g ^ self.disp_0[f as usize]) as usize]
    }
}

/** Given: given input pairs (f, g) where the f and g are both unique, with no collisions, check that running
 * a displacement step does not increase the number of collisions. */
#[test]
fn test_hmp01_displacement() {
    // r=n=17 is a slightly out of spec, but the algorithm _should_ still place each element in an empty slot
    let r_bits = 17;
    let x: Vec<(u64, u64)> = (0..((1 << 17) as u64)).map(|x| (x, x)).collect();
    let disp = compute_displacement(&x, r_bits);
    let mut freqs = vec![0_usize; 1 << r_bits];
    for (f, g) in x {
        let (u, _v) = (g ^ disp[f as usize], f);
        freqs[u as usize] += 1;
    }
    let max_col = freqs.iter().max().unwrap();
    assert!(*max_col == 1, "{}", max_col);
}

#[test]
fn test_hmp01_quad_hash() {
    let n_bits = 16;
    let x: Vec<(u64, u64)> = (0..(u16::MAX as u64))
        .map(|x| ((x * x) >> n_bits, ((x * x) & ((1 << n_bits) - 1)) as u64))
        .collect();
    let hash = HMPQuadHash::new(&x, n_bits + 1);
    let mut dst = vec![false; 1 << (n_bits + 1)];
    for (f, g) in x {
        let u = hash.query(f, g);
        assert!(!dst[u as usize], "f {} g {} u {}", f, g, u);
        dst[u as usize] = true;
    }
}

impl Dict<u64, u64> for HagerupMP01Dict {
    fn new(data: &[(u64, u64)]) -> HagerupMP01Dict {
        // The paper technically special cases constant construction when 'n < w'
        // but in practice this is not important
        let ext_keys: Vec<u128> = data
            .iter()
            .map(|(k, _v)| {
                // require inputs of size <u32::MAX, for now
                let sk: u32 = (*k).try_into().unwrap();
                error_correcting_code(sk)
            })
            .collect();
        let d = find_distinguishing_positions(&ext_keys);

        let reduced_data: Vec<(u128, u64)> = data
            .iter()
            .map(|(k, v)| {
                let sk: u32 = (*k).try_into().unwrap();
                (pext(error_correcting_code(sk), d), *v)
            })
            .collect();

        let n_bits = reduced_data
            .len()
            .checked_next_power_of_two()
            .unwrap()
            .trailing_zeros();
        let val_bits = d.count_ones();
        // at ẟ=29/128, output space should be in [2 n^{5.4}]
        assert!(
            val_bits <= n_bits * 6,
            "val bits {} n bits {} | {}",
            val_bits,
            n_bits,
            data.len()
        );

        let r_bits = n_bits + 1;

        /* Instead of implicitly encoding a trie (Tarjan+Yao approach, recommended by
         * HMP01), use the trick mentioned by Ružić09 to peel off r_bits at a time.
         * This should reduce indirection. */
        let mut lead: Vec<u64> = reduced_data
            .iter()
            .map(|(k, _v)| (*k & ((1 << r_bits) - 1)) as u64)
            .collect();

        let mut hashes: Vec<HMPQuadHash> = Vec::new();
        // Constant number of rounds, to behave more like the input is worst case.
        // (On random input, fewer rounds are needed.)
        // Number of steps: ceil{val_bits / r_bits} <= 6.
        for round in 0..5 {
            let mut pairs: Vec<(u64, u64)> = lead
                .iter()
                .zip(reduced_data.iter())
                .map(|(f, (k, _v))| {
                    (
                        *f,
                        ((*k >> (r_bits * (round + 1))) & ((1 << r_bits) - 1)) as u64,
                    )
                })
                .collect();

            // Deduplicate pairs, because we do not yet know for certain if the quadratic reduction hash can handle them
            // TODO: extract into a function and test independently
            pairs.sort_unstable();
            let mut w = 1;
            for r in 1..pairs.len() {
                if pairs[r] != pairs[w - 1] {
                    pairs[w] = pairs[r];
                    w += 1;
                }
            }
            pairs.truncate(w);

            let hash = HMPQuadHash::new(&pairs, r_bits);

            lead = lead
                .iter()
                .zip(reduced_data.iter())
                .map(|(f, (k, _v))| {
                    hash.query(
                        *f,
                        ((*k >> (r_bits * (round + 1))) & ((1 << r_bits) - 1)) as u64,
                    )
                })
                .collect();

            hashes.push(hash);
        }

        // Identify a value which is _not_ a valid key.
        let mut is_key: Vec<bool> = vec![false; data.len()];
        for (k, _v) in data {
            if *k < data.len() as u64 {
                is_key[*k as usize] = true;
            }
        }
        let non_key = is_key.iter().position(|x| !x).unwrap_or(data.len()) as u64;

        // Fill table entries
        let r_bits = hashes[0].r_bits;
        let mut table: Vec<(u64, u64)> = vec![(non_key, 0); 1 << r_bits];
        for (i, v) in lead.iter().enumerate() {
            table[*v as usize] = data[i];
        }

        HagerupMP01Dict {
            distinguishing_bits: d,
            r_bits,
            hashes,
            table,
            non_key,
        }
    }
    fn query(&self, key: u64) -> Option<u64> {
        assert!(key <= u32::MAX as u64);
        // First, reduce universe
        let k: u32 = key.try_into().unwrap();
        let d = pext(error_correcting_code(k), self.distinguishing_bits);

        let mut val = (d & ((1 << self.r_bits) - 1)) as u64;
        for (i, h) in self.hashes.iter().enumerate() {
            let nxt = ((d >> ((i as u32 + 1) * (self.r_bits))) & ((1 << self.r_bits) - 1)) as u64;
            val = h.query(val, nxt);
        }
        // Note: to reduce end-to-end latency, when there are >=4 parts, there are better designs
        // than plain iteration; using e.g. balanced binary tree construction ((0,1),(2,3)) would
        // reduce the depth. (Although it requires double displacement to do symmetric merges.)

        let entry = self.table[val as usize];
        if entry.0 != key || key == self.non_key {
            None
        } else {
            Some(entry.1)
        }
    }
}

#[test]
fn test_hmp01_dict() {
    let x: Vec<(u64, u64)> = (0..(u16::MAX as u64))
        .map(|x| ((x * x) & (u32::MAX as u64), x))
        .collect();
    let hash = HagerupMP01Dict::new(&x);
    for (k, v) in x {
        assert!(Some(v) == hash.query(k));
    }
}

/* ---------------------------------------------------------------------------- */

/** Like [HagerupMP01Dict], but skipping the universe reduction because it isn't that
 * useful when the word size is small. */
pub struct HMP01UnreducedDict {
    /** Number of bits for steps to hash down to */
    r_bits: u32,
    /** Instead of using an implicit trie indexed by this hash, just use the hash
     * directly to reduce the input by iteratively mapping [2 log(n)]->[log n] bits */
    hashes: Vec<HMPQuadHash>,
    /** Table of key-value pairs indexed by the perfect hash construction */
    table: Vec<(u64, u64)>,
    /** A value which is not a valid key */
    non_key: u64,
}

impl Dict<u64, u64> for HMP01UnreducedDict {
    fn new(data: &[(u64, u64)]) -> HMP01UnreducedDict {
        let val_bits = data
            .iter()
            .map(|(k, _v)| (64 - k.leading_zeros()))
            .max()
            .unwrap();

        let n_bits = data
            .len()
            .checked_next_power_of_two()
            .unwrap()
            .trailing_zeros();

        let r_bits = n_bits + 1;
        let steps = val_bits.max(1).div_ceil(r_bits) - 1;

        let mut lead: Vec<u64> = data
            .iter()
            .map(|(k, _v)| *k & ((1 << r_bits) - 1))
            .collect();

        let mut hashes: Vec<HMPQuadHash> = Vec::new();
        for round in 0..steps {
            let mut pairs: Vec<(u64, u64)> = lead
                .iter()
                .zip(data.iter())
                .map(|(f, (k, _v))| (*f, (*k >> (r_bits * (round + 1))) & ((1 << r_bits) - 1)))
                .collect();

            // Deduplicate pairs, because we do not yet know for certain if the quadratic reduction hash can handle them
            // TODO: extract into a function and test independently
            pairs.sort_unstable();
            let mut w = 1;
            for r in 1..pairs.len() {
                if pairs[r] != pairs[w - 1] {
                    pairs[w] = pairs[r];
                    w += 1;
                }
            }
            pairs.truncate(w);

            let hash = HMPQuadHash::new(&pairs, r_bits);

            lead = lead
                .iter()
                .zip(data.iter())
                .map(|(f, (k, _v))| {
                    hash.query(*f, (*k >> (r_bits * (round + 1))) & ((1 << r_bits) - 1))
                })
                .collect();

            hashes.push(hash);
        }

        // Identify a value which is _not_ a valid key.
        let mut is_key: Vec<bool> = vec![false; data.len()];
        for (k, _v) in data {
            if *k < data.len() as u64 {
                is_key[*k as usize] = true;
            }
        }
        let non_key = is_key.iter().position(|x| !x).unwrap_or(data.len()) as u64;

        // Fill table entries
        let mut table: Vec<(u64, u64)> = vec![(non_key, 0); 1 << r_bits];
        for (i, v) in lead.iter().enumerate() {
            table[*v as usize] = data[i];
        }

        HMP01UnreducedDict {
            r_bits,
            hashes,
            table,
            non_key,
        }
    }
    fn query(&self, key: u64) -> Option<u64> {
        let mut val = key & ((1 << self.r_bits) - 1);
        for (i, h) in self.hashes.iter().enumerate() {
            let nxt = (key >> ((i as u32 + 1) * (self.r_bits))) & ((1 << self.r_bits) - 1);
            val = h.query(val, nxt);
        }

        let entry = self.table[val as usize];
        if entry.0 != key || key == self.non_key {
            None
        } else {
            Some(entry.1)
        }
    }
}

#[test]
fn test_hmp01_unreduced() {
    let x: Vec<(u64, u64)> = (0..(u16::MAX as u64))
        .map(|x| ((x * x * x * x), x))
        .collect();
    let hash = HMP01UnreducedDict::new(&x);
    for (k, v) in x {
        assert!(Some(v) == hash.query(k));
    }
}
