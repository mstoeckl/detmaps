/* SPDX-License-Identifier: MPL-2.0 */
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::hash::Hash;
use std::ops::RangeInclusive;

pub mod util;

pub trait Dict<K, V> {
    /** Construct a dictionary for the given key-value pairs. All keys must be unique
     * and be < 1<<u_bits. */
    fn new(data: &[(K, V)], u_bits: u32) -> Self;
    /** Find the value for a key, if one exists */
    fn query(&self, key: K) -> Option<V>;
}

// TODO: split _construction_ and _evaluation_ for perfect hash, to more easily share evaluation
// procedures and allow comparing different construction methods (e.g., with varying approximation
// parameters. Similarly, make a 'CollidingHash<K, V>' trait?)

// One approach: parameterize HDict by <T: fn(&keys, u_bits )->PerfectHash>

pub trait PerfectHash<K, V> {
    /** Construct a perfect hash for a set of _unique_ keys, where each key is <1<<u_bits. */
    fn new(keys: &[K], u_bits: u32) -> Self;
    /** Map a key to a value, with no two inputs in the set mapping to the same value */
    fn query(&self, key: K) -> V;
    /** Number of output bits */
    fn output_bits(&self) -> u32;
}

/* ---------------------------------------------------------------------------- */

/** Generic construction of a dictionary from a perfect hash */
pub struct HDict<H: PerfectHash<u64, u64>> {
    hash: H,
    /** Table of key-value pairs indexed by the perfect hash construction */
    table: Vec<(u64, u64)>,
    /** A value which is not a valid key */
    non_key: u64,
}

impl<T: PerfectHash<u128, u64>> PerfectHash<u64, u64> for T {
    fn new(keys: &[u64], u_bits: u32) -> Self {
        let keys_u128: Vec<u128> = keys.iter().map(|x| *x as u128).collect();
        T::new(&keys_u128, u_bits)
    }
    fn query(&self, key: u64) -> u64 {
        PerfectHash::<u128, u64>::query(self, key as u128)
    }
    fn output_bits(&self) -> u32 {
        PerfectHash::<u128, u64>::output_bits(self)
    }
}

impl<H: PerfectHash<u64, u64>> Dict<u64, u64> for HDict<H> {
    fn new(data: &[(u64, u64)], u_bits: u32) -> Self {
        let keys: Vec<u64> = data.iter().map(|x| x.0).collect();
        let hash = H::new(&keys, u_bits);
        drop(keys);

        if false {
            /* Validate hash correctness */
            let mut check = vec![false; 1 << hash.output_bits()];
            for (k, _v) in data.iter() {
                // println!("{} -> {}", k, hash.query(*k));
                let mut b = true;
                std::mem::swap(&mut check[hash.query(*k) as usize], &mut b);
                assert!(!b, "collision, from {} to collided {}", k, hash.query(*k));
            }
        }

        // note: can reduce space usage of this very slightly with deterministic MIF alg.
        let mut is_key: Vec<bool> = vec![false; data.len()];
        for (k, _v) in data {
            if *k < data.len() as u64 {
                is_key[*k as usize] = true;
            }
        }
        let output_bits = hash.output_bits();
        let non_key = is_key.iter().position(|x| !x).unwrap_or(data.len()) as u64;
        let mut table: Vec<(u64, u64)> = vec![(non_key, 0); 1 << output_bits];
        for (k, v) in data.iter() {
            let h = hash.query(*k);
            table[h as usize] = (*k, *v);
        }
        HDict {
            hash,
            table,
            non_key,
        }
    }
    fn query(&self, key: u64) -> Option<u64> {
        let val = self.hash.query(key);
        let entry = self.table[val as usize];
        if entry.0 != key || key == self.non_key {
            None
        } else {
            Some(entry.1)
        }
    }
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
    fn new(data: &[(K, V)], _u_bits: u32) -> BinSearchDict<K, V> {
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
    fn new(data: &[(K, V)], _u_bits: u32) -> BTreeDict<K, V> {
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
    fn new(data: &[(K, V)], _u_bits: u32) -> HashDict<K, V> {
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
    util::clmul(x as u64, k1) ^ (util::clmul(x as u64, k2) << 64)
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
    let mut z: Vec<u128> = y.iter().map(|a| util::pext(*a, d)).collect();
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
 *
 * The input can be viewed as pair (f,g), where `f` has t bit values and
 * `g` has r bit values.
 */
#[inline(never)]
fn compute_displacement(input: &[(u64, u64)], t: u32, r: u32) -> Vec<u64> {
    let mut bucket_sizes = vec![0_u64; 1 << t];
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
        // TODO: the threshold should be 'sqrt(2)*keys.len()' instead of `2*keys.len()`
        assert!(next_log_power_of_two(2 * keys.len()) <= (1 << r_bits));
        for (f, g) in keys {
            assert!(f >> r_bits == 0);
            assert!(g >> r_bits == 0);
        }
        let disp_0 = compute_displacement(keys, r_bits, r_bits);
        let disp_keys: Vec<(u64, u64)> = keys
            .iter()
            .map(|(f, g)| (*g ^ disp_0[*f as usize], *f))
            .collect();
        let disp_1 = compute_displacement(&disp_keys, r_bits, r_bits);

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
    let disp = compute_displacement(&x, r_bits, r_bits);
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
    fn new(data: &[(u64, u64)], u_bits: u32) -> HagerupMP01Dict {
        assert!(u_bits <= u32::BITS);

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
                (util::pext(error_correcting_code(sk), d), *v)
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
        let d = util::pext(error_correcting_code(k), self.distinguishing_bits);

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
    let hash = HagerupMP01Dict::new(&x, u32::BITS);
    for (k, v) in x {
        assert!(Some(v) == hash.query(k));
    }
}

/* ---------------------------------------------------------------------------- */

/** Like [HagerupMP01Dict], but skipping the universe reduction because it isn't that
 * useful when the word size is small. */
pub struct IteratedHMP01Hash {
    /** Number of bits for steps to hash down to */
    r_bits: u32,
    /** Instead of using an implicit trie indexed by this hash, just use the hash
     * directly to reduce the input by iteratively mapping [2 log(n)]->[log n] bits */
    hashes: Vec<HMPQuadHash>,
}
pub type HMP01UnreducedDict = HDict<IteratedHMP01Hash>;

impl PerfectHash<u64, u64> for IteratedHMP01Hash {
    fn new(keys: &[u64], u_bits: u32) -> IteratedHMP01Hash {
        let n_bits = next_log_power_of_two(keys.len());

        let r_bits = n_bits + 1;
        let steps = u_bits.max(1).div_ceil(r_bits) - 1;

        let mut lead: Vec<u64> = keys.iter().map(|k| *k & ((1 << r_bits) - 1)).collect();

        let mut hashes: Vec<HMPQuadHash> = Vec::new();
        for round in 0..steps {
            let mut pairs: Vec<(u64, u64)> = lead
                .iter()
                .zip(keys.iter())
                .map(|(f, k)| (*f, (*k >> (r_bits * (round + 1))) & ((1 << r_bits) - 1)))
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
                .zip(keys.iter())
                .map(|(f, k)| hash.query(*f, (*k >> (r_bits * (round + 1))) & ((1 << r_bits) - 1)))
                .collect();

            hashes.push(hash);
        }

        IteratedHMP01Hash { r_bits, hashes }
    }
    fn query(&self, key: u64) -> u64 {
        let mut val = key & ((1 << self.r_bits) - 1);
        for (i, h) in self.hashes.iter().enumerate() {
            let nxt = (key >> ((i as u32 + 1) * (self.r_bits))) & ((1 << self.r_bits) - 1);
            val = h.query(val, nxt);
        }
        val
    }
    fn output_bits(&self) -> u32 {
        self.r_bits
    }
}

#[test]
fn test_hmp01_unreduced() {
    let x: Vec<(u64, u64)> = (0..(u16::MAX as u64))
        .map(|x| ((x * x * x * x), x))
        .collect();
    let hash = HMP01UnreducedDict::new(&x, u64::BITS);
    for (k, v) in x {
        assert!(Some(v) == hash.query(k));
    }
}

/* ---------------------------------------------------------------------------- */

struct Ruzic09BReduction {
    /* Multipliers; with each mask-multiply-shift-add step, domain size is reduced by half, approximately, from 64 to 32+O(log n) to 16+O(log n) bits. */
    steps: Vec<(u64, u32)>,
    // Number of bits of the final output
    end_bits: u32,
}

fn find_good_multiplier_fast(keys: &[u64], s: u32, max_mult_bits: u32) -> u64 {
    fn f(x: u64, a: u64, s: u32) -> u64 {
        (x >> s) + a * (x & ((1 << s) - 1))
    }

    /* Get an upper bound, no more than 2x the actual value, of the number of colliding
     * (real valued) parameters in the doubly-open range, counting multiplicities. */
    fn est_bad_parameters(keys: &[u64], s: u32, parameters: RangeInclusive<u64>) -> u64 {
        /* alternative approach: use rank queries? */
        let (low, high) = parameters.into_inner();
        if high <= low {
            return 0;
        }

        /* cmp_h = h_x < h_y || (h_x == h_y && l_x < l_y)
         * cmp_l = l_x < l_y || (l_x == l_y && (h_x < h_y || (h_x == h_y && l_x < l_y)))
         *       = l_x < l_y || (l_x == l_y && h_x < h_y)
         * (the simplification follows since l_x==l_y and h_x==h_y implies x=y.)
         */

        let key_cmp_low = |x, y| {
            f(x, low, s) < f(y, low, s)
                || (f(x, low, s) == f(y, low, s) && f(x, high, s) < f(y, high, s))
        };
        let key_cmp_high = |x, y| {
            f(x, high, s) < f(y, high, s)
                || (f(x, high, s) == f(y, high, s) && f(x, low, s) < f(y, low, s))
        };

        let mut phi_indices: Vec<usize> = (0..keys.len()).collect();
        let mut f_indices: Vec<usize> = (0..keys.len()).collect();
        phi_indices.sort_unstable_by(|i, j| {
            if key_cmp_low(keys[*i], keys[*j]) {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        });
        f_indices.sort_unstable_by(|i, j| {
            if key_cmp_high(keys[*i], keys[*j]) {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        });

        let mut f_inv: Vec<usize> = vec![0; keys.len()];
        for (i, v) in f_indices.iter().enumerate() {
            f_inv[*v] = i;
        }
        let mut pi_1 = vec![0u64; keys.len()];
        for (i, v) in pi_1.iter_mut().enumerate() {
            *v = f_inv[phi_indices[i]] as u64;
        }

        // This overestimates inversions by up to a factor of 2
        /* TODO: Chan + Patrascu provide an O(n/epsilon) time epsilon-approximation algorithm.
         *
         * (The paper claims an O(n) time approximation with epsilon
         * = O(1/polylog(n)) is achievable, which _may_ lead to total
         * O(1)-bit overhead in R09B's framework.)
         */
        let mut est_invs = 0;
        for (i, v) in pi_1.iter().enumerate() {
            est_invs += (i as u64).abs_diff(*v);
        }
        2 * est_invs
    }

    let mut low = 0;
    let mut high = (1 << max_mult_bits) - 1;

    /* Upper bound on the number of bad parameters. */
    let mut bad_param_est = (keys.len().checked_mul(keys.len()).unwrap() / 2) as u64;
    while bad_param_est > 0 {
        let mid = low + (high - low) / 2;
        assert!(low < mid && mid < high);

        let m1est = est_bad_parameters(keys, s, low..=mid);
        /* Each iteration should reduce the estimate to <= 2/3 of the previous value */
        if m1est <= (2 * bad_param_est) / 3 {
            high = mid;
        } else {
            low = mid;
        }
        /* To keep the number of iterations nonadaptive/similar to the worst case
         * behavior, reduce the bad parameter estimate only to what is guaranteed,
         * not to m1est. */
        bad_param_est = (2 * bad_param_est) / 3;
    }
    assert!(bad_param_est == 0);

    low + (high - low) / 2
}

/* Count the number of inversions in a permutation */
fn count_inversions(perm: &[u64]) -> u64 {
    /* Standard divide-and-conquer implementation.
     *
     * TODO: implement and compare: Chan & Pǎtraşcu 2010, "Counting Inversions,
     * Offline Orthogonal Range Counting, and Related Problems"
     */

    fn sorted_merge(left: &[u64], right: &[u64], output: &mut [u64]) {
        assert!(left.len() + right.len() == output.len());
        assert!(left.len() >= 1 && right.len() >= 1);

        let mut output_iter = output.iter_mut();
        let mut left_iter = left.iter();
        let mut right_iter = right.iter();

        let mut left_val = *left_iter.next().unwrap();
        let mut right_val = *right_iter.next().unwrap();

        loop {
            if left_val < right_val {
                *output_iter.next().unwrap() = left_val;

                if let Some(x) = left_iter.next() {
                    left_val = *x;
                } else {
                    /* Only right values remaining */
                    *output_iter.next().unwrap() = right_val;
                    while let Some(y) = right_iter.next() {
                        *output_iter.next().unwrap() = *y;
                    }
                    assert!(output_iter.next().is_none());
                    return;
                }
            } else {
                *output_iter.next().unwrap() = right_val;

                if let Some(x) = right_iter.next() {
                    right_val = *x;
                } else {
                    /* Only left values remaining */
                    *output_iter.next().unwrap() = left_val;
                    while let Some(y) = left_iter.next() {
                        *output_iter.next().unwrap() = *y;
                    }
                    assert!(output_iter.next().is_none());
                    return;
                }
            }
        }
    }

    fn count_inv_rec(a: &mut [u64], scratch: &mut [u64]) -> u64 {
        if a.len() <= 1 {
            return 0;
        }

        let mid = a.len() / 2;
        let (left, right) = a.split_at_mut(mid);
        let inv_left = count_inv_rec(left, &mut scratch[..left.len()]);
        let inv_right = count_inv_rec(right, &mut scratch[..right.len()]);

        /* Number of inversions (where an element in `left` is greater than one in `right`. */
        let mut inv_cross = 0;
        /* right_pos is least index for which right[right_pos] > el */
        let mut right_pos = 0;
        for el in left.iter() {
            while right_pos < right.len() && *el >= right[right_pos] {
                right_pos += 1;
            }

            let nright_lt_el = right_pos as u64;
            inv_cross += nright_lt_el;
        }

        sorted_merge(left, right, scratch);
        a.copy_from_slice(scratch);

        inv_left + inv_right + inv_cross
    }

    let mut mut_perm = Vec::from(perm);
    /* Temporary buffer to merge lists in */
    let mut scratch = vec![0u64; perm.len()];
    let count = count_inv_rec(&mut mut_perm, &mut scratch);

    if false {
        /* brute force count */
        let mut brute_count = 0;
        for i in 0..perm.len() {
            for j in 0..i {
                if perm[j] > perm[i] {
                    brute_count += 1;
                }
            }
        }
        assert!(brute_count == count);
    }
    count
}

fn find_good_multiplier_precise(keys: &[u64], s: u32, max_mult_bits: u32) -> u64 {
    fn f(x: u64, a: u64, s: u32) -> u64 {
        (x >> s) + a * (x & ((1 << s) - 1))
    }

    /** Count the number of collisions for a specific multiplier `a`.
     *
     * This _permutes_ the provided key array, but does not change its contents. */
    fn count_collisions(keys: &mut [u64], a: u64, s: u32) -> u64 {
        assert!(keys.len() <= u32::MAX as usize);

        keys.sort_unstable_by_key(|k| f(*k, a, s));

        /* count collisions for this value */
        let mut ncoll = 0;
        let mut class_start = 0;
        // TODO: use an iterator
        while class_start < keys.len() {
            let mut class_end = class_start;
            let cur_val = f(keys[class_start], a, s);
            while class_end < keys.len() && f(keys[class_end], a, s) == cur_val {
                class_end += 1;
            }
            let class_sz = (class_end - class_start) as u64;
            ncoll += class_sz * (class_sz.max(1) - 1) / 2;
            class_start = class_end;
        }

        ncoll
    }

    /** Count the number of bad (real-valued) parameters in the given doubly-open interval;
     * this is an upper bound on the number of collisions occuring under integral parameters.
     */
    fn count_bad_parameters(keys: &[u64], s: u32, parameters: RangeInclusive<u64>) -> u64 {
        /* alternative approach: use rank queries? */
        let (low, high) = parameters.into_inner();
        if high <= low {
            return 0;
        }

        /* cmp_h = h_x < h_y || (h_x == h_y && l_x < l_y)
         * cmp_l = l_x < l_y || (l_x == l_y && (h_x < h_y || (h_x == h_y && l_x < l_y)))
         *       = l_x < l_y || (l_x == l_y && h_x < h_y)
         * (the simplification follows since l_x==l_y and h_x==h_y implies x=y.)
         */

        let key_cmp_low = |x, y| {
            f(x, low, s) < f(y, low, s)
                || (f(x, low, s) == f(y, low, s) && f(x, high, s) < f(y, high, s))
        };
        let key_cmp_high = |x, y| {
            f(x, high, s) < f(y, high, s)
                || (f(x, high, s) == f(y, high, s) && f(x, low, s) < f(y, low, s))
        };

        let mut phi_indices: Vec<usize> = (0..keys.len()).collect();
        let mut f_indices: Vec<usize> = (0..keys.len()).collect();
        phi_indices.sort_unstable_by(|i, j| {
            if key_cmp_low(keys[*i], keys[*j]) {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        });
        f_indices.sort_unstable_by(|i, j| {
            if key_cmp_high(keys[*i], keys[*j]) {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        });

        let mut f_inv: Vec<usize> = vec![0; keys.len()];
        for (i, v) in f_indices.iter().enumerate() {
            f_inv[*v] = i;
        }
        let mut pi_1 = vec![0u64; keys.len()];
        for (i, v) in pi_1.iter_mut().enumerate() {
            *v = f_inv[phi_indices[i]] as u64;
        }

        let inv_count = count_inversions(&pi_1);
        // println!("inv count: {:?}", inv_count);
        if false {
            /* Count the number of key pairs colliding in the target interval. */
            let mut brute_count = 0;
            for i in 0..keys.len() {
                for j in 0..i {
                    let (ki, kj) = (keys[i], keys[j]);
                    let in_m1 = |x, y| f(x, low, s) < f(y, low, s) && f(x, high, s) > f(y, high, s);
                    if in_m1(ki, kj) || in_m1(kj, ki) {
                        brute_count += 1;
                    }
                }
            }
            assert!(inv_count == brute_count);

            if false {
                /* Count the number of _integral_ collisions. This runs in O(n^3) time */
                let mut key_copy = Vec::from(keys);
                let mut actual_m1: u64 = 0;
                for a in low + 1..high {
                    let ncoll = count_collisions(&mut key_copy, a, s);
                    actual_m1 += ncoll;
                }
                assert!(actual_m1 <= inv_count);
            }
        }
        inv_count
    }

    /* R09B binary searches in [1..(max_mult) - 1]; but 0 is also a valid
     * multiplier, so check it. (The search procedure in fact also works
     * for negative multipliers.) */
    let mut low = 0;
    let mut high = (1 << max_mult_bits) - 1;
    let mut key_copy = Vec::from(keys);
    let low_col = count_collisions(&mut key_copy, low, s);
    let high_col = count_collisions(&mut key_copy, high, s);
    let mut ok_multiplier = None;
    if low_col == 0 {
        /* This and the other short-circuit exits would make the reduction
         * return _significantly_ faster then worst-case on random input,
         * just as if we were to short-circuit test against a few random
         * multipliers. To make performance more similar to the worst
         * case, delay the choice of which output to make until the very
         * end -- only using the short circuit value if the final `low`
         * value has a collision. */
        ok_multiplier = Some(low);
    }
    if high_col == 0 {
        ok_multiplier = Some(high);
    }

    let mut bad_params = count_bad_parameters(keys, s, low..=high);
    while low < high {
        let mid = low + (high - low) / 2;

        let mid_col = count_collisions(&mut key_copy, mid, s);
        if mid_col == 0 {
            ok_multiplier = Some(mid);
        }

        let col_bound_low = count_bad_parameters(keys, s, low..=mid);
        let col_bound_high = count_bad_parameters(keys, s, mid..=high);

        /* Total number of collisions in (low,high)=(low,mid)+mid+(mid, high) */
        if low < mid && mid < high {
            assert!(
                col_bound_low + col_bound_high + mid_col == bad_params,
                "{} {} {} | {} {} {} ?= {}",
                low,
                mid,
                high,
                col_bound_low,
                col_bound_high,
                mid_col,
                bad_params
            );
        }

        if col_bound_low <= col_bound_high {
            high = mid;
            bad_params = col_bound_low;
        } else {
            low = mid;
            bad_params = col_bound_high;
        }
    }

    let candidate_col = count_collisions(&mut key_copy, low, s);
    if candidate_col == 0 {
        low
    } else {
        assert!(ok_multiplier.is_some());
        ok_multiplier.unwrap()
    }
}

impl Ruzic09BReduction {
    fn new_fast(keys: &[u64], mut u_bits: u32) -> Ruzic09BReduction {
        let mut keys: Vec<u64> = Vec::from(keys);
        let n_bits = next_log_power_of_two(keys.len());

        let max_mult_bits = ((keys.len() as f64).log2() * 2.0 / 1.5_f64.log2()).ceil() as u32;

        // Number of bits approaches 3.42*n_bits
        let mut mults = Vec::new();
        while u_bits > max_mult_bits + n_bits {
            assert!(max_mult_bits <= u64::BITS);

            // TODO: choose s to minimize 'next_ubits'
            let s = (u_bits - max_mult_bits) / 2;
            let next_ubits = (u_bits - s).max(s + max_mult_bits) + 1;
            if next_ubits == u_bits {
                /* at small n, `u_bits + max_mult_bits + n_bits` might always be true. */
                break;
            }

            // println!("n {} u {} s {}", n_bits, u_bits, s);
            let a = find_good_multiplier_fast(&keys, s, max_mult_bits);
            mults.push((a, s));

            for k in keys.iter_mut() {
                *k = (*k >> s) + a * (*k & ((1 << s) - 1));
            }

            u_bits = next_ubits;

            if true {
                // Verify multiplier actually works
                let mut b = BTreeSet::new();
                for k in keys.iter() {
                    assert!(b.insert(k), "duplicate value: {}", k);
                }
            }
            if true {
                for k in keys.iter() {
                    assert!(k >> u_bits == 0);
                }
            }
        }

        Ruzic09BReduction {
            steps: mults,
            end_bits: u_bits,
        }
    }
    #[allow(dead_code)]
    fn new_precise(kvs: &[u64], mut u_bits: u32) -> Ruzic09BReduction {
        /* There have been improvements to the problem of counting the number of permutation inversions;
         * since Ružić09 was written. See e.g. Chan & Pǎtraşcu 2010,
         * "Counting Inversions, Offline Orthogonal Range Counting, and Related Problems" */
        let mut keys: Vec<u64> = Vec::from(kvs);

        if true {
            let s = BTreeSet::from_iter(keys.iter());
            assert!(s.len() == keys.len(), "invalid input, has duplicates");
        }

        let n_bits = next_log_power_of_two(keys.len());

        if kvs.len() > u32::MAX as usize {
            /*  */
            return Ruzic09BReduction {
                steps: vec![],
                end_bits: u64::BITS,
            };
        }

        /* A valid multiplier _exists_ in [0..n(n-1)/2+1); round search space up to nearest multiple of 2 */
        let max_mult_bits =
            next_log_power_of_two(kvs.len().checked_mul(kvs.len().max(1) - 1).unwrap() / 2 + 1);
        assert!(max_mult_bits <= u64::BITS);

        // Number of bits approaches 2*n_bits
        let mut mults = Vec::new();
        while u_bits > max_mult_bits + n_bits {
            assert!(max_mult_bits <= u64::BITS);

            // TODO: choose s to minimize 'next_ubits'
            let s = (u_bits - max_mult_bits) / 2;
            let next_ubits = (u_bits - s).max(s + max_mult_bits) + 1;
            if next_ubits == u_bits {
                /* at small n, `u_bits + max_mult_bits + n_bits` might always be true. */
                break;
            }

            // println!("n {} u {} s {}", n_bits, u_bits, s);
            let a = find_good_multiplier_precise(&keys, s, max_mult_bits);
            mults.push((a, s));

            for k in keys.iter_mut() {
                *k = (*k >> s) + a * (*k & ((1 << s) - 1));
            }

            u_bits = next_ubits;

            if true {
                for k in keys.iter() {
                    assert!(k >> u_bits == 0);
                }
            }
            if false {
                // Verify multiplier actually works
                let mut b = BTreeSet::new();
                for k in keys.iter() {
                    assert!(b.insert(k), "duplicate value: {}", k);
                }
            }
        }

        Ruzic09BReduction {
            steps: mults,
            end_bits: u_bits,
        }
    }

    fn apply(&self, mut key: u64) -> u64 {
        for (a, s) in self.steps.iter() {
            key = (key >> s) + a * (key & ((1 << s) - 1));
        }
        key
    }
}

pub struct R09BfxHMP01Hash {
    reduction: Ruzic09BReduction,
    // TODO: domain is [n^4 or n^5]; use a binary tree merge instead of a sequence of hashes
    main_hash: IteratedHMP01Hash,
}
pub type R09BfxHMP01Dict = HDict<R09BfxHMP01Hash>;

impl PerfectHash<u64, u64> for R09BfxHMP01Hash {
    fn new(data: &[u64], u_bits: u32) -> R09BfxHMP01Hash {
        let reduction = Ruzic09BReduction::new_fast(&data, u_bits);

        let reduced_keys: Vec<u64> = data.iter().map(|k| reduction.apply(*k)).collect();

        let main_hash = IteratedHMP01Hash::new(&reduced_keys, reduction.end_bits);

        R09BfxHMP01Hash {
            reduction,
            main_hash,
        }
    }
    fn query(&self, key: u64) -> u64 {
        let d = self.reduction.apply(key);
        self.main_hash.query(d)
    }

    fn output_bits(&self) -> u32 {
        self.main_hash.output_bits()
    }
}

pub struct R09BpxHMP01Hash {
    reduction: Ruzic09BReduction,
    // TODO: domain should be [O(n^3)], explicitly use just two hashes
    main_hash: IteratedHMP01Hash,
}
pub type R09BpxHMP01Dict = HDict<R09BpxHMP01Hash>;

impl PerfectHash<u64, u64> for R09BpxHMP01Hash {
    fn new(data: &[u64], u_bits: u32) -> R09BpxHMP01Hash {
        let reduction = Ruzic09BReduction::new_precise(&data, u_bits);

        let reduced_keys: Vec<u64> = data.iter().map(|k| reduction.apply(*k)).collect();

        let main_hash = IteratedHMP01Hash::new(&reduced_keys, reduction.end_bits);

        R09BpxHMP01Hash {
            reduction,
            main_hash,
        }
    }
    fn query(&self, key: u64) -> u64 {
        let d = self.reduction.apply(key);
        self.main_hash.query(d)
    }

    fn output_bits(&self) -> u32 {
        self.main_hash.output_bits()
    }
}

#[test]
fn test_r09xhmp01_dict() {
    let x: Vec<(u64, u64)> = (0..(1 << 5) as u64)
        .map(|x| ((x * 13) % (1 << 6), x))
        .collect();
    let hash = R09BfxHMP01Dict::new(&x, u64::BITS);
    for (k, v) in x {
        assert!(Some(v) == hash.query(k));
    }
    let x: Vec<(u64, u64)> = (0..(u8::MAX as u64)).map(|x| (x.pow(8), x)).collect();
    let hash = R09BfxHMP01Dict::new(&x, u64::BITS);
    for (k, v) in x {
        assert!(Some(v) == hash.query(k));
    }
    let x: Vec<(u64, u64)> = (0..(1u64 << 10)).map(|x| (x.pow(6), x)).collect();
    let hash = R09BfxHMP01Dict::new(&x, u64::BITS);
    for (k, v) in x {
        assert!(Some(v) == hash.query(k));
    }
}

/* ---------------------------------------------------------------------------- */

/** Deterministic variant of tabulation hashing from `[N]^k to [N^2]`, followed
 * by double displacement hashing for the last `[N^2] -> [N]` step.
 *
 * This is rather memory intensive and the most efficient form of [Ruzic09BReduction]
 * likely uses much less memory and is faster; but it should query fewer cache lines
 * than [HMP01UnreducedDict] on fully random u64 input.
 *
 * TODO: make generic and replace u128/u64 with smallest types that fit 2*r_bits and r_bits
 *
 * Requires: key space <2^64. */
pub struct XorReducedDict {
    /** For [2^r]^k -> [2^{2r}], this table has 'k-2' entries; the processing for the
     * first two table entries is explicit.
     *
     * The first implicit entry uses the identity map; the second
     * uses 'x -> x << r'. */
    xor_table: Vec<Vec<u128>>,
    quad: HMPQuadHash,
    r_bits: u32,
    table: Vec<(u64, u64)>,
    non_key: u64,
}

/** Return value `x` so that $A \cap {b \oplus x}_{x \in B} = \emptyset$.
 *
 * Assumes all entries of A and B are <2^u, and produces output <2^u.
 *
 * This _should_ run in O(n log n) time once fully implemented, but for now
 * does O(n (log n)^2) due to not radix sorting.
 *
 * Only the trivial Ω(n) lower bound is known. Can this be solved more efficiently?
 */
fn xor_distinguish(ra: &[u128], rb: &[u128], u_bits: u32) -> u128 {
    assert!(ra.len().checked_mul(rb.len()).unwrap() < 1usize.checked_shl(u_bits).unwrap());

    // minor optimization: access through &mut a[..], &mut b[..] instead of &[], &[]
    // as contents only need to be permuted
    let mut a = Vec::from(ra);
    let mut b = Vec::from(rb);

    let mut x = 0;
    const CHUNK_BITS: u32 = 4;
    for i in 0..u_bits.div_ceil(CHUNK_BITS) {
        /* Upper bounds on the number of collisions if the next byte of `x`
         * is chosen to be a given value */
        let mut collision_counters = [0u64; 1 << CHUNK_BITS];

        // TODO: incrementally sort bits C*i..C*(i+1) in a and b instead of doing a full sort
        // (On 1M-sized inputs, this may give factor >=2 speedups)
        a.sort_unstable_by_key(|v| v.reverse_bits());
        b.sort_unstable_by_key(|v| (v ^ x).reverse_bits());

        let prefix_mask: u128 = (1 << (CHUNK_BITS * i)) - 1;
        let chunk_mask: u128 = (1 << CHUNK_BITS) - 1;

        /* Iterate over distinct (little-endian) prefixes of 'a' and 'b^x'; and count,
         * for each overlapping prefix group */
        let mut sa = 0;
        let mut sb = 0;
        while sa < a.len() && sb < b.len() {
            let apr = (a[sa] & prefix_mask).reverse_bits();
            let bpr = ((b[sb] ^ x) & prefix_mask).reverse_bits();

            match apr.cmp(&bpr) {
                Ordering::Less => {
                    sa += 1;
                    continue;
                }
                Ordering::Greater => {
                    sb += 1;
                    continue;
                }
                Ordering::Equal => (),
            }
            let common_prefix = a[sa] & prefix_mask;
            assert!(common_prefix == (b[sb] ^ x) & prefix_mask);
            let mut ia = sa;
            while ia < a.len() && (a[ia] & prefix_mask) == common_prefix {
                ia += 1;
            }
            let mut ib = sb;
            while ib < b.len() && ((b[ib] ^ x) & prefix_mask) == common_prefix {
                ib += 1;
            }

            /* Now: &a[sa..ia] and &b[sb..ib] should all share the same prefix,
             * so process collisions in this sub-block */
            for v in &a[sa..ia] {
                assert!(*v & prefix_mask == common_prefix);
            }
            for v in &b[sb..ib] {
                assert!((*v ^ x) & prefix_mask == common_prefix);
            }

            let mut a_counters = [0u64; 1 << CHUNK_BITS];
            let mut b_counters = [0u64; 1 << CHUNK_BITS];
            for v in &a[sa..ia] {
                a_counters[((v >> (CHUNK_BITS * i)) & chunk_mask) as usize] += 1;
            }
            assert!((x >> (CHUNK_BITS * i)) & chunk_mask == 0);
            for v in &b[sb..ib] {
                b_counters[((v >> (CHUNK_BITS * i)) & chunk_mask) as usize] += 1;
            }
            // println!("{} {:?} {:?}", common_prefix, a_counters, b_counters);
            for x in 0..(1 << CHUNK_BITS) {
                for i in 0..(1 << CHUNK_BITS) {
                    // todo: overflow risk if both a,b lengths are >= 2^32
                    collision_counters[x] += a_counters[i] * b_counters[x ^ i];
                }
            }

            /* Move to next block. */
            sa = ia;
            sb = ib;
        }

        let best_disp = collision_counters
            .iter()
            .enumerate()
            .min_by_key(|x| *x.1)
            .unwrap()
            .0;
        // println!("{:?} {:x} {:x}", collision_counters, best_disp, x);
        x ^= (best_disp as u128) << (CHUNK_BITS * i);
    }

    x
}

#[cfg(test)]
fn xor_distinguish_simple(a: &[u128], b: &[u128]) -> u128 {
    let mut x = vec![false; a.len() * b.len()];
    for va in a.iter() {
        for vb in b.iter() {
            let v = va ^ vb;
            if v < x.len() as u128 {
                x[v as usize] = true;
            }
        }
    }
    let f = x.iter().position(|x| !x).unwrap_or(a.len() * b.len()) as u128;
    f
}

#[test]
fn test_xor_distinguish() {
    fn test_disjointness(a: &[u128], b: &[u128], x: u128) {
        let sa: BTreeSet<u128> = BTreeSet::from_iter(a.iter().map(|v| *v));
        let sbx: BTreeSet<u128> = BTreeSet::from_iter(b.iter().map(|v| (*v ^ x)));
        assert!(sa.is_disjoint(&sbx));
    }

    fn test_pair(a: Vec<u128>, b: Vec<u128>, u_bits: u32) {
        let x = xor_distinguish_simple(&a, &b);
        let y = xor_distinguish(&a, &b, u_bits);
        println!("x {} y {}", x, y);
        test_disjointness(&a, &b, x);
        test_disjointness(&a, &b, y);
    }

    test_pair((0..100).collect(), (50..300).collect(), 16);
    test_pair((50..300).collect(), (0..100).collect(), 16);
    test_pair((0..256).collect(), (0..255).map(|x| (x << 8)).collect(), 16);
    test_pair((0..255).map(|x| (x << 8)).collect(), (0..256).collect(), 16);
    test_pair(
        (0..255).map(|x| x / 4).collect(),
        (0..256).map(|x| x / 4).collect(),
        16,
    );
    test_pair(vec![0, 1, 3], vec![0, 7], 3);
}

/** Return least `i` so that (1<<i) >= `v`.
 * 0=>0, 1=>0, 2=>1, 3=>2, 4=>2, 5=>3...
 */
fn next_log_power_of_two(v: usize) -> u32 {
    usize::BITS - (v.max(1) - 1).leading_zeros()
}
#[test]
fn test_ceil_log2() {
    assert!(next_log_power_of_two(0) == 0);
    assert!(next_log_power_of_two(1) == 0);
    assert!(next_log_power_of_two(3) == 2);
    assert!(next_log_power_of_two(4) == 2);
    assert!(next_log_power_of_two(5) == 3);
}

/** Given r-bit keys and arbitrary 2r-bit `prev`, return a vector `v` with 2r bit
 * entries so that { prev[i] ^ v[keys[i]] }_i is 1-1 with { (prev[i], keys[i]) }_i.
 *
 * Runtime: O(n (log n)^2), space usage O(n) */
#[inline(never)]
fn construct_xor_table_entry(prev: &[u128], keys: &[u64], r_bits: u32) -> Vec<u128> {
    let n_bits = next_log_power_of_two(keys.len());
    assert!(prev.len() == keys.len());
    assert!(r_bits >= n_bits);

    /* First: group rows by identical keys */
    let mut key_freq_offset = vec![0_u64; 1 << r_bits];
    for k in keys {
        key_freq_offset[*k as usize] += 1;
    }
    let mut offset = 0;
    for ko in key_freq_offset.iter_mut() {
        offset += *ko;
        *ko = offset;
    }
    let mut index_table = vec![0_u64; keys.len()];
    for (i, k) in keys.iter().enumerate() {
        key_freq_offset[*k as usize] -= 1;
        let o = key_freq_offset[*k as usize];
        index_table[o as usize] = i as u64;
    }

    /* Note: it _may_ be possible to reduce random reads/writes by, instead of just
     * storing 'u64' row ids, storing their associated prev & output values, like
     * Vec<(u64,u128,u128)>, and writing back the output values at the very end.  */
    let mut groups: Vec<Vec<Vec<u64>>> = Vec::new();
    for _ in 0..64 {
        groups.push(Vec::new());
    }
    for i in 0..(1 << r_bits) {
        let s = key_freq_offset[i];
        let e = key_freq_offset
            .get(i + 1)
            .copied()
            .unwrap_or(keys.len() as u64);
        if e <= s {
            continue;
        }
        let group = Vec::from(&index_table[s as usize..e as usize]);
        groups[next_log_power_of_two(group.len()) as usize].push(group);
    }
    drop(key_freq_offset);
    drop(index_table);

    // todo: try the other construction approach (with a single pass incremental radix tree)
    // using e.g. branching factor 1<<4 and falling back to sorted lists for nodes with few descendants
    let mut outputs_by_row = vec![0_u128; keys.len()];
    for sz in 0..groups.len() - 1 {
        let [class, next] = &mut groups[sz..sz + 2] else {
            break;
        };

        while class.len() >= 2 {
            let g1: Vec<u64> = class.pop().unwrap();
            let g2: Vec<u64> = class.pop().unwrap();

            /* Merge two groups of rows, uniformly modifying all output values
             * for one of the two groups to ensure there are no collisions. */
            let mut vals_1 = vec![0u128; g1.len()];
            let mut vals_2 = vec![0u128; g2.len()];
            for (v, i) in vals_1.iter_mut().zip(g1.iter()) {
                *v = prev[*i as usize] ^ outputs_by_row[*i as usize];
            }
            for (v, i) in vals_2.iter_mut().zip(g2.iter()) {
                *v = prev[*i as usize] ^ outputs_by_row[*i as usize];
            }
            let x = xor_distinguish(&vals_1, &vals_2, r_bits * 2);
            for i in g1.iter() {
                outputs_by_row[*i as usize] ^= x;
            }
            next.push([g1, g2].concat());
        }
        if let Some(g) = class.pop() {
            next.push(g);
        }
        assert!(class.is_empty());
    }

    assert!(groups.iter().map(|x| x.len()).sum::<usize>() == 1);
    drop(groups);

    let mut outputs_by_key = vec![0_u128; 1 << r_bits];
    for (i, k) in keys.iter().enumerate() {
        if outputs_by_key[*k as usize] != 0 {
            assert!(outputs_by_key[*k as usize] == outputs_by_row[i]);
        }
        outputs_by_key[*k as usize] = outputs_by_row[i];
    }
    outputs_by_key
}

#[test]
fn test_xor_table_entry() {
    fn test_pattern(base: Vec<u128>, keys: Vec<u64>, r_bits: u32) {
        for v in base.iter() {
            assert!(v >> (2 * r_bits) == 0);
        }
        for v in keys.iter() {
            assert!(v >> (r_bits) == 0);
        }

        let table = construct_xor_table_entry(&base, &keys, r_bits);
        let inputs = BTreeSet::from_iter(base.iter().zip(keys.iter()));
        let outputs = BTreeSet::from_iter(
            base.iter()
                .zip(keys.iter())
                .map(|(x, y)| x ^ table[*y as usize]),
        );
        println!(
            "unique input pairs {}, unique xor'd outputs {}",
            inputs.len(),
            outputs.len()
        );
        assert!(inputs.len() == outputs.len());
    }

    test_pattern(
        (0..(u16::MAX as u128)).map(|x| x * x).collect(),
        (0..(u16::MAX as u64)).map(|x| x / 3).collect(),
        16,
    );
    test_pattern(
        (0..(u16::MAX as u128)).map(|x| (x / 3) * (x / 3)).collect(),
        (0..(u16::MAX as u64)).map(|x| x).collect(),
        16,
    );
    test_pattern(
        (0..(u16::MAX as u128))
            .map(|x| (x & 0xff00) * 0x10001)
            .collect(),
        (0..(u16::MAX as u64)).map(|x| x & 0x00ff).collect(),
        16,
    );
    test_pattern(
        (0..(u16::MAX as u128)).map(|x| (x / 6) * 0x12345).collect(),
        (0..(u16::MAX as u64)).map(|x| x / 15).collect(),
        16,
    );
}

impl Dict<u64, u64> for XorReducedDict {
    fn new(data: &[(u64, u64)], u_bits: u32) -> Self {
        let n_bits = next_log_power_of_two(data.len());
        let r_bits = n_bits + 1;

        let mut xor_table = Vec::new();
        let mask_kp01 = if 2 * r_bits < u_bits {
            (1 << (2 * r_bits)) - 1
        } else {
            u64::MAX
        };
        let mut comp_keys: Vec<u128> = data.iter().map(|x| (x.0 & mask_kp01) as u128).collect();

        for i in 2..u_bits.div_ceil(r_bits) {
            let next: Vec<u64> = data
                .iter()
                .map(|kv| (kv.0 >> (r_bits * i)) & ((1 << r_bits) - 1))
                .collect();
            let table = construct_xor_table_entry(&comp_keys, &next, r_bits);
            for (c, kv) in comp_keys.iter_mut().zip(data.iter()) {
                let kp = (kv.0 >> (r_bits * i)) & ((1 << r_bits) - 1);
                *c = *c ^ table[kp as usize];
            }
            xor_table.push(table);
        }

        if false {
            assert!(BTreeSet::<u128>::from_iter(comp_keys.iter().map(|x| *x)).len() == data.len());
        }

        let pairs: Vec<(u64, u64)> = comp_keys
            .iter()
            .map(|x| (((*x) & ((1 << r_bits) - 1)) as u64, (*x >> r_bits) as u64))
            .collect();

        let quad = HMPQuadHash::new(&pairs, r_bits);
        let keys: Vec<u64> = pairs.iter().map(|x| quad.query(x.0, x.1)).collect();

        // TODO: extract this table setup into a common function (or: create a PerfectHash
        // trait from which dictionaries can be generically derived)

        // Identify a value which is _not_ a valid key.
        let mut is_key: Vec<bool> = vec![false; data.len()];
        for kv in data.iter() {
            if kv.0 < data.len() as u64 {
                is_key[kv.0 as usize] = true;
            }
        }
        let non_key = is_key.iter().position(|x| !x).unwrap_or(data.len()) as u64;
        let mut table: Vec<(u64, u64)> = vec![(non_key, 0); 1 << r_bits];
        for (k, p) in keys.iter().zip(data.iter()) {
            assert!(table[*k as usize].0 == non_key);
            table[*k as usize] = *p;
        }

        XorReducedDict {
            xor_table,
            quad,
            r_bits,
            table,
            non_key,
        }
    }
    fn query(&self, key: u64) -> Option<u64> {
        let mut x = 0;

        let rmask = (1 << self.r_bits) - 1;

        let kp0 = (key & rmask) as u128;
        let kp1 = (((key >> self.r_bits) & rmask) as u128) << self.r_bits;
        x ^= kp0;
        x ^= kp1;

        // Specializing by the table length, would probably help significantly
        for (i, t) in self.xor_table.iter().enumerate() {
            let kp = (key >> (self.r_bits * (i + 2) as u32)) & rmask;
            x ^= t[kp as usize];
        }

        let idx = self.quad.query(
            (x & ((1 << self.r_bits) - 1)) as u64,
            (x >> self.r_bits) as u64,
        );
        let entry = self.table[idx as usize];
        if entry.0 != key || key == self.non_key {
            None
        } else {
            Some(entry.1)
        }
    }
}

#[test]
fn test_xor_reduced_dict() {
    let x: Vec<(u64, u64)> = (0..(u8::MAX as u64)).map(|x| (x.pow(8), x)).collect();
    let hash = XorReducedDict::new(&x, u64::BITS);
    for (k, v) in x {
        assert!(
            Some(v) == hash.query(k),
            "{:?} {:?} {:?}",
            k,
            v,
            hash.query(k)
        );
    }
    let x: Vec<(u64, u64)> = (0..(1u64 << 10)).map(|x| (x.pow(6), x)).collect();
    let hash = XorReducedDict::new(&x, u64::BITS);
    for (k, v) in x {
        assert!(Some(v) == hash.query(k));
    }
}

/* ---------------------------------------------------------------------------- */

/** Ruzic09a hash construction. */
struct Ruzic09AHash {
    // TODO: in practice, this hash is used to reduce after universe reduction, from
    // e.g. [n^2] or [n^5/2] to [n]. If we had a fast and simple universe reduction,
    // then we could set Φ= 1/2 Ψ or Φ= 3/4 Ψ, which in large dimensions would
    // allow for _significantly_ smaller displacement tables,
    // slightly smaller buckets, and possibly faster construction times.
    phi_bits: u32,
    psi_bits: u32,
    output_bits: u32,
    displacement: Vec<u64>,

    /* There are several possible bucket implementations; Ružić09A suggests Raman95Dict
     * or the general O(n^2 log n)-time construction of Ružić08b.
     * In practice on _pseudo random_ inputs, bucket sizes are small and a linear
     * scan over a a list of elements be fast, but this would not provide any
     * strong worst-case query guarantees.
     *
     * Note: in practice, bucket construction takes about half the time of
     * `Ruzic09AHash`, so even slight improvements on `Raman95Dict` (like the
     * 'ax mod p' approach, if implemented divisionless using multiplications
     * and shifts) would be worth it.
     */
    bucket_hashes: Vec<Option<Raman95Dict>>,

    /* Fallback hash structure operates on O(n / log n) elements, so
     * slower construction is OK. Could alternatively use a variant of
     * Ruzic09A.
     *
     * A possible alternative to the fallback hash structure would be to apply
     * a construction like HMP01 _to the specific large buckets_, avoiding
     * the quadratic space costs of directly using Raman95Dict, and possibly
     * saving construction time. (This may increase the worst-case memory-load
     * chain length by one, since with the current design one can speculatively
     * query the fallback hash; but because the [n^2/polylog n] input size is below
     * the easy [n^2] non-collision threshold, the Ruzic09A construction does
     * not minimize latency anyway.
     */
    fallback_hash: HMPQuadHash,
}

fn get_bit_pos_sequence(psi_bits: u32, phi_bits: u32) -> Vec<u32> {
    let i_star =
        (f32::log2(psi_bits as f32) - f32::log2(f32::log2(phi_bits as f32)) - 1.0).floor() as u32;
    let mut seq = vec![0];
    let floor_logphi = (phi_bits as f32).log2().floor() as u32;
    for i in 1..=i_star {
        let p =
            ((1.0 - 0.5f32.powi(i as i32)) * (psi_bits as f32)).floor() as u32 - i * floor_logphi;
        seq.push(p);
    }
    // println!(
    //     "{:?}, psi_bits {} floor logphi = {}",
    //     seq, psi_bits, floor_logphi
    // );
    for i in i_star + 1..=2 * i_star + 2 {
        // Paper states: p_i = p_{i-1} + floor_logphi for i > i_star,
        // but it does not appear as though: p_{i_star} + (i_star + 2) floor_logphi = psi
        // (it usually is a bit less, as 2^-i_star <= 2 phi/psi. This may be fine; perhaps
        // not all bits actually need to be set.)
        seq.push(psi_bits - floor_logphi * (2 * i_star + 2 - i));
    }
    assert!(
        *seq.last().unwrap() == psi_bits,
        "{:?} Φ {} Ψ {} i_star {}",
        seq,
        phi_bits,
        psi_bits,
        i_star
    );
    for w in seq.windows(2) {
        assert!(w[0] < w[1]);
    }

    seq
}

/* Algorithm 1 of Ruzic09A to identify leaf sets, assuming they are already sorted */
fn ruzic09a_alg_1(
    keys: &[(u64, u64)],
    phi_bits: u32,
    psi_bits: u32,
    pl: u32,
    ph: u32,
    displacement: &[u64],
) -> Vec<(u64, u32, u64)> {
    fn h(k: (u64, u64), disp: &[u64]) -> u64 {
        k.1 ^ disp[k.0 as usize]
    }
    // println!("\nalg 1");

    let floorlogphi = (phi_bits as f32).log2().floor() as u32; // todo: replace with bit hack
    let eta = 1 << (ph - pl + floorlogphi);

    let mut leaf_list: Vec<(u64, u32, u64)> = Vec::new();
    let mut j = phi_bits;
    let mut k = 0;
    let mut v = h(keys[0], displacement) >> (psi_bits - pl);
    let mut count = 1;
    /* Note: paper uses 1-indexing, while this implementation uses 0-indexing, so qstart=0 instead of 1 */
    let mut qstart = 0;
    for q in 1..keys.len() {
        let hq = h(keys[q], displacement) >> (psi_bits - pl);
        // println!("loop: q {} (h={}, phi={}), cur vjk={:?}", q, hq, keys[q].0, (v,j,k));
        if v != hq {
            // hq change: always appends in progress set
            // but: last (v,j,k) could be out of date?
            // maybe the inner loop is appropriate, and should make (87, 1, 1)
            // ?: inner loop, > vs >= ?

            if count > 0 {
                // println!("first: {:?}", (v,j,k));
                leaf_list.push((v, j, k));
            }
            j = phi_bits;
            k = 0;
            v = hq;
            count = 1;
            qstart = q;
        } else if keys[q].0 >= ((k + 1) << j) {
            // key 0 moved ahead by at least 1

            if count > 0 {
                // println!("second: {:?}", (v,j,k));
                leaf_list.push((v, j, k));
            }
            // Note: this sort of loop could probably be optimized in practice

            /* Typo in original paper: the < in `(k+2)2^j < keys[q].0` should
             * be <= to ensure that, if the current keys[q].0 is >=2 steps larger
             * than the previous, the leaf size increases to cover it. */
            while ((k + 2) << j) <= keys[q].0 {
                j += 1;
                k >>= 1;
            }
            k = k + 1;
            count = 1;
            qstart = q;
        } else {
            count += 1;
            if count > eta {
                // println!("splitting: {} > {}", count, eta);
                while j > 0 && (k + 1) << j > keys[q].0 {
                    j -= 1;
                    k <<= 1;
                }
                // println!("now vjk={:?}, q={}, qstart={}", (v,j,k), q, qstart);
                if (k + 1) << j <= keys[q].0 {
                    /* typo in original? phi(x_qstart), not phi(q_start) ? */
                    if keys[qstart].0 < (k + 1) << j {
                        leaf_list.push((v, j, k));
                        // println!("inner: {:?}", (v,j,k));
                        while keys[qstart].0 < (k + 1) << j {
                            qstart += 1;
                            count -= 1;
                        }
                    }
                    // println!("incr k");
                    k += 1;
                }
            }
        }
    }

    // TODO: this crashes / can be wrong when this ends up being the _only_ leaf, fix and reenable
    // or: when first leaf is '(0,0,2)'?
    if count > 0 {
        /* Paper forgets to add the trailing leaf set,
         * if one exists.
         *          *
         * Note: this uses the same logic as if we were to add a new
         * sentinel value for h(x_q)>>(psi-pl) that is distinct from all
         * previous values. Is this correct?
         */
        // println!("final?: ({},{},{})", v, j, k);
        leaf_list.push((v, j, k));
    }

    leaf_list
}

/** Ruzic09A selection of a bit range, inefficient implementation */
fn select_bits_slow(
    displacement: &mut [u64],
    keys: &[u128],
    phi_bits: u32,
    psi_bits: u32,
    pl: u32,
    ph: u32,
) {
    // println!("SELECT BITS");
    assert!(phi_bits > 0 && psi_bits > 0 && ph > pl && psi_bits >= ph);
    assert!(
        keys.len() <= u32::MAX as usize,
        "possible multiplication overflow if n>u32::MAX"
    );
    let floorlogphi = (phi_bits as f32).log2().floor() as u32; // todo: replace with bit hack
    let eta = 1 << (ph - pl + floorlogphi);

    fn h(k: (u64, u64), disp: &[u64]) -> u64 {
        k.1 ^ disp[k.0 as usize]
    }

    /* Split keys by displaced prefix */
    let mut split_keys: Vec<(u64, u64)> = Vec::new();
    for k in keys.iter() {
        let upper = k >> psi_bits;
        let lower = (k & ((1 << psi_bits) - 1)) as u64;
        split_keys.push((upper as u64, lower as u64));
    }

    // Later optimizations: radix/bucket sort
    split_keys.sort_unstable_by_key(|k| (h(*k, displacement) >> (psi_bits - pl), k.0));
    // println!("{:?} / {}", split_keys, displacement.len());

    let leaf_list = ruzic09a_alg_1(&split_keys, phi_bits, psi_bits, pl, ph, displacement);

    assert!(leaf_list.len() <= (4 * keys.len() * (phi_bits as usize + 1)).div_ceil(eta as usize));

    fn jk_index(j: u32, k: u64, phi_bits: u32) -> usize {
        k as usize + (1 << (phi_bits - j))
    }

    // println!("leaves {:?} eta {}", leaf_list, eta); // issue: eta & leaves....

    /* TODO: fast exit, to start with, if leaf_list.is_empty()? but j=0 should indicate leaves */

    #[derive(Clone)]
    struct SubTable {
        v: u64,
        x: Vec<u64>,
    }

    // bit-by-bit, search table.
    // let mut x_table: Vec<Vec<u64>> = Vec::new();
    // x_table.resize_with(1 << (phi_bits + pl + 1), || Vec::new()); // TODO: this table is inefficient when pl is close to psi_bits; 1<<(psi+phi) is >> n. Use a different structure/list of lists for the leaves.

    let mut x_table: Vec<Vec<SubTable>> = Vec::new();
    x_table.resize_with(1 << (phi_bits + 1), || Vec::new());

    // X_table: by 'v'

    // Construct X(v,j,k) sets for all leaf sets
    let mut leaf_start = 0;

    for leaf_vjk in leaf_list.iter() {
        assert!(
            split_keys[leaf_start].0 >> leaf_vjk.1 == leaf_vjk.2,
            "leaf start: {}->{:?}; leaf {:?}, parts {} {}, h prefix {}",
            leaf_start,
            split_keys[leaf_start],
            leaf_vjk,
            split_keys[leaf_start].0 >> leaf_vjk.1,
            leaf_vjk.2,
            // wide range of 'h' prefixes.
            (h(split_keys[leaf_start], displacement) >> (psi_bits - pl)),
        );

        let mut leaf_end = leaf_start;
        while leaf_end < split_keys.len()
            && (h(split_keys[leaf_end], displacement) >> (psi_bits - pl))
                == (h(split_keys[leaf_start], displacement) >> (psi_bits - pl))
            && (split_keys[leaf_end].0 >> leaf_vjk.1) == leaf_vjk.2
        {
            leaf_end += 1;
            // println!(
            //     "next: {} {}",
            //     h(split_keys[leaf_end], displacement) >> (psi_bits - pl),
            //     (split_keys[leaf_end].0 >> leaf_vjk.1)
            // );
        }

        // println!(
        //     "leaf: {:?} | psi {} phi {} pl={}..{}=ph | {}..{} | jk: {}/{}",
        //     leaf_vjk,
        //     psi_bits,
        //     phi_bits,
        //     pl,
        //     ph,
        //     leaf_start,
        //     leaf_end,
        //     jk_index(leaf_vjk.1, leaf_vjk.2, phi_bits),
        //     x_table.len()
        // );

        /* Compute frequency vector for the set X(v,j,k) */
        let mut x_subtable: Vec<u64> = Vec::new();

        x_subtable.resize(1 << (ph - pl + 1), 0);
        for a in leaf_start..leaf_end {
            // Extract bits between pl..ph (counting from MSB, so psi-ph..psi-pl from LSB)
            let h_bits =
                (h(split_keys[a], displacement) >> (psi_bits - ph)) & ((1 << (ph - pl)) - 1);
            x_subtable[(1 << (ph - pl)) + h_bits as usize] += 1;
        }
        for d in (0..(ph - pl)).rev() {
            for y in 0..(1 << d) {
                x_subtable[(1 << d) + y] =
                    x_subtable[(2 << d) + 2 * y] + x_subtable[(2 << d) + 2 * y + 1];
            }
        }

        /* Note: leaf_vjk.0 is in sorted order, so the individual lists will also be */
        x_table[jk_index(leaf_vjk.1, leaf_vjk.2, phi_bits)].push(SubTable {
            v: leaf_vjk.0,
            x: x_subtable,
        });

        // println!("subtable {:?}", x_subtable);

        leaf_start = leaf_end
    }

    /* Outermost loop, over _bits of table coordinates_; from fine grained to coarse grained */
    for j in 0..phi_bits {
        // (x,j,k)

        for k in 0..(1 << (phi_bits - j - 1)) {
            /* Choose delta for this (k,j), picking bits from MSB to LSB. */
            let mut delta: u64 = 0;

            let xt0seq = &x_table[jk_index(j, 2 * k, phi_bits)];
            let xt1seq = &x_table[jk_index(j, 2 * k + 1, phi_bits)];

            for bit in 0..(ph - pl) {
                let mut mu0 = 0;
                let mut mu1 = 0;
                let delta1 = delta | 1 << (ph - bit - 1);

                let mut xt0iter = xt0seq.iter();
                let mut xt1iter = xt1seq.iter();
                let Some(mut xt0) = xt0iter.next() else {
                    // No collisions
                    continue;
                };
                let Some(mut xt1) = xt1iter.next() else {
                    // No collisions
                    continue;
                };
                loop {
                    // println!("loop diff: {} {}", xt0.v, xt1.v);
                    if xt0.v < xt1.v {
                        let Some(n) = xt0iter.next() else {
                            break;
                        };
                        xt0 = n;
                        continue;
                    } else if xt0.v > xt1.v {
                        let Some(n) = xt1iter.next() else {
                            break;
                        };
                        xt1 = n;
                        continue;
                    }

                    /* Matching v values; sets may collide */

                    // sanity check the range? 4<<bit?
                    // println!("loop");

                    // todo: to avoid overflow, need n<2^32
                    for q in 0..(1u64 << bit) {
                        let sub_base = 1 << bit;
                        // println!("query: pl={} ph={} ; q={}", pl, ph, q);
                        // TODO: sanity check this
                        mu0 += xt0.x[sub_base + q as usize]
                            * xt1.x[sub_base + (q ^ (delta >> (ph - bit - 1))) as usize];
                        mu1 += xt1.x[sub_base + q as usize]
                            * xt1.x[sub_base + (q ^ (delta1 >> (ph - bit - 1))) as usize];
                    }

                    let Some(n0) = xt0iter.next() else {
                        break;
                    };
                    xt0 = n0;
                    let Some(n1) = xt1iter.next() else {
                        break;
                    };
                    xt1 = n1;
                }

                if mu1 < mu0 {
                    // TODO: or: offset relative to psi_bits?
                    delta |= 1 << (ph - bit - 1);
                }
            }

            // TODO: to which table entries does delta apply? Everything with prefix k?
            // i.e, are we picking the top bits of the displacement table coordinates in the outer loop?
            for l in ((2 * k + 1) << j)..((2 * k + 2) << j) {
                displacement[l as usize] ^= delta << (psi_bits - ph);
            }

            /* Join the {X(v,j,2k)}_v,{X(v,j,2k+1)}_v sets after displacing the latter  */
            let mut joined: Vec<SubTable> = Vec::new();
            {
                let mut xt0iter = xt0seq.iter();
                let mut xt1iter = xt1seq.iter();
                let mut xt0 = xt0iter.next();
                let mut xt1 = xt1iter.next();

                while xt0.is_some() || xt1.is_some() {
                    let ov0 = xt0.map(|a| a.v);
                    let ov1 = xt1.map(|a| a.v);
                    // println!("join: {:?} {:?}", ov0, ov1);
                    let order = if let Some(v0) = ov0 {
                        if let Some(v1) = ov1 {
                            v0.cmp(&v1)
                        } else {
                            Ordering::Less
                        }
                    } else {
                        // v0 is None, done
                        Ordering::Greater
                    };
                    /* xt0.v ? xt1.v */
                    match order {
                        Ordering::Less => {
                            /* Note: cloning can be avoided and one can move the table instead */
                            joined.push(xt0.unwrap().clone());

                            xt0 = xt0iter.next();
                        }
                        Ordering::Greater => {
                            let t1 = xt1.unwrap();

                            let mut twisted = vec![0; 1 << (ph - pl + 1)];
                            // TODO: is this correct?
                            for d in 0..=(ph - pl) {
                                for y in 0..(1u64 << d) {
                                    twisted[(1 << d) + y as usize] +=
                                        t1.x[(1 << d) + (y ^ (delta >> (ph - d))) as usize]
                                }
                            }
                            joined.push(SubTable {
                                v: t1.v,
                                x: twisted,
                            });

                            xt1 = xt1iter.next();
                        }
                        Ordering::Equal => {
                            let t0 = xt0.unwrap();
                            let t1 = xt1.unwrap();
                            assert!(t0.v == t1.v);

                            let mut combo = t0.x.clone();
                            // TODO: is this correct?
                            for d in 0..=(ph - pl) {
                                for y in 0..(1u64 << d) {
                                    combo[(1 << d) + y as usize] +=
                                        t1.x[(1 << d) + (y ^ (delta >> (ph - d))) as usize]
                                }
                            }
                            joined.push(SubTable { v: t1.v, x: combo });

                            xt0 = xt0iter.next();
                            xt1 = xt1iter.next();
                        }
                    }
                }
            }

            let mut prev = Vec::new();
            std::mem::swap(&mut x_table[jk_index(j + 1, k, phi_bits)], &mut prev);
            let mut merged = Vec::new();
            {
                /* Merge 'joined' and 'prev' into 'merged', preserving sort order by v. */
                let mut j_iter = joined.into_iter();
                let mut m_iter = prev.into_iter();
                let mut j_o = j_iter.next();
                let mut m_o = m_iter.next();
                while j_o.is_some() || m_o.is_some() {
                    // println!("merge: {:?} {:?}", j_o.as_ref().map(|x| x.v), m_o.as_ref().map(|x| x.v));
                    let order = if let Some(ref j) = j_o {
                        if let Some(ref m) = m_o {
                            j.v.cmp(&m.v)
                        } else {
                            Ordering::Less
                        }
                    } else {
                        // v0 is None, done
                        Ordering::Greater
                    };
                    /* j ? m */
                    match order {
                        Ordering::Less => {
                            merged.push(j_o.unwrap());
                            j_o = j_iter.next();
                        }
                        Ordering::Greater => {
                            merged.push(m_o.unwrap());
                            m_o = m_iter.next();
                        }
                        Ordering::Equal => {
                            unreachable!("merged/joined lists should be disjoint");
                        }
                    }
                }
            }

            x_table[jk_index(j + 1, k, phi_bits)] = merged;
        }
    }
}

fn get_r09a_hash_bits(n_bits: u32) -> (u32, u32, u32, u32, u32, u32) {
    let psi_bits = n_bits;
    let prev_log2 = u32::BITS - psi_bits.leading_zeros(); // 1000 / 0xffff
    let phi_bits = n_bits.checked_sub(2 * prev_log2).unwrap();
    assert!(
        phi_bits >= 1,
        "n: {}, psi: {} prev_log2: {}",
        n_bits,
        psi_bits,
        prev_log2
    );
    let u_bits = psi_bits + phi_bits;
    /* Collision bound is <= 3 phi^2 n, and each bucket with > phi^3 psi entries has
     * >= (phi^3 psi) / 2 collisions per entry, so there can be at most 6 n / (psi*phi)
     * buckets of size > phi^3 psi. */
    let large_limit = (6 << n_bits) / (phi_bits * psi_bits) as usize;
    /* f_bits is computed based on the worst case fallback list size to avoid
     * having input-dependent parameters */
    let f_bits = next_log_power_of_two(large_limit);
    let r_bits = u_bits.div_ceil(2).max(f_bits + 1);
    let output_bits = psi_bits.max(r_bits);
    (u_bits, output_bits, psi_bits, phi_bits, f_bits, r_bits)
}

impl Ruzic09AHash {
    fn new(keys: &[u128], n_bits: u32) -> Self {
        let (u_bits, output_bits, psi_bits, phi_bits, f_bits, r_bits) = get_r09a_hash_bits(n_bits);
        assert!(u_bits <= u64::BITS); // TODO: reduce key space to u64, and/or parameterize it
        assert!(next_log_power_of_two(keys.len()) <= n_bits);
        for k in keys.iter() {
            assert!(k >> u_bits == 0);
        }
        /*
        println!(
            "keys {} nbits {} ubits {} phi {} psi {}",
            keys.len(),
            n_bits,
            u_bits,
            psi_bits,
            phi_bits
        );*/

        assert!(u_bits <= psi_bits + phi_bits);

        let mut displacement = vec![0_u64; 1 << phi_bits];

        let bit_pos_seq = get_bit_pos_sequence(psi_bits, phi_bits);
        for p in bit_pos_seq.windows(2) {
            let (pl, ph) = (p[0], p[1]);
            // println!("segment: {} {}", pl, ph);

            /* Set bits pl..ph (counting from MSB) of the displacement table entries */
            select_bits_slow(&mut displacement, &keys, phi_bits, psi_bits, pl, ph)
        }

        let mut bucket_lists: Vec<Option<Vec<u64>>> = Vec::new();
        bucket_lists.resize_with(1 << psi_bits, || Some(Vec::new()));

        let max_bucket_size = (phi_bits * phi_bits * phi_bits * psi_bits) as usize;

        /* Compute buckets and construct the fallback key list */
        let mut fallback_list: Vec<u64> = Vec::new();
        for k in keys.iter() {
            let upper = k >> psi_bits;
            let lower = (k & ((1 << psi_bits) - 1)) as u64;
            let h = lower ^ displacement[upper as usize];

            let bucket = &mut bucket_lists[h as usize];

            if let Some(ref mut b) = bucket {
                b.push((*k).try_into().unwrap());

                if b.len() > max_bucket_size {
                    for k in b.drain(..) {
                        fallback_list.push(k);
                    }
                    *bucket = None;
                }
            } else {
                fallback_list.push((*k).try_into().unwrap());
            }
        }
        /* Construct a hash function for `fallback_list` and record which values
         * it takes */
        let split_fallback: Vec<(u64, u64)> = fallback_list
            .iter()
            .map(|x| (((*x >> r_bits) as u64, (x & ((1 << r_bits) - 1)) as u64)))
            .collect();
        let fallback_hash = HMPQuadHash::new(&split_fallback, r_bits);

        let mut taken_spots = vec![false; 1 << output_bits];
        let mut free_i = 0;
        for k in split_fallback.iter() {
            taken_spots[fallback_hash.query(k.0, k.1) as usize] = true;
        }

        /* Construct bucket hash functions and select unique fallback values
         * which are not used by the fallback hash */
        let mut bucket_hashes: Vec<Option<Raman95Dict>> = Vec::new();
        for bucket in bucket_lists {
            let Some(b) = bucket else {
                bucket_hashes.push(None);
                continue;
            };

            let mut b2: Vec<(u64, u64)> = Vec::new();
            for k in b.iter() {
                while taken_spots[free_i] {
                    free_i += 1;
                }
                taken_spots[free_i] = true;

                b2.push((*k, free_i as u64));
            }
            bucket_hashes.push(Some(Raman95Dict::new(&b2, u64::BITS)));
        }

        // println!(
        //     "fallback count: {} of {}, sz {}, lim {}, u_bits {} n_bits {}",
        //     fallback_list.len(),
        //     keys.len(),
        //     max_bucket_size,
        //     1 << f_bits,
        //     u_bits,
        //     n_bits
        // );

        assert!(
            next_log_power_of_two(fallback_list.len()) <= f_bits,
            "fallback limit: {} <!= {}, psi {} phi {} max_bucket_size {}",
            fallback_list.len(),
            1 << f_bits,
            psi_bits,
            phi_bits,
            max_bucket_size
        );

        Ruzic09AHash {
            phi_bits,
            psi_bits,
            output_bits,
            displacement,
            bucket_hashes,
            fallback_hash,
        }
    }
    fn query(&self, k: u128) -> u64 {
        let upper = k >> self.psi_bits;
        let lower = (k & ((1 << self.psi_bits) - 1)) as u64;
        let h = lower ^ self.displacement[upper as usize];

        let bucket = &self.bucket_hashes[h as usize];
        if let Some(ref v) = bucket.as_ref() {
            // TODO: using (h, upper) and the displacement table, one can
            // derive the original key, since `lower = h XOR disp[upper]`.
            // Therefore: it suffices for this inner hash function to query
            // _just_ using the (much fewer) upper bits
            v.query(k.try_into().unwrap()).unwrap_or(0)
            // If no match, the value does not matter
        } else {
            let fhi = (k >> self.fallback_hash.r_bits) as u64;
            let flo = (k & ((1 << self.fallback_hash.r_bits) - 1)) as u64;
            let v = self.fallback_hash.query(fhi, flo);
            v
        }
    }
}

#[test]
fn test_ruzic09a_hash() {
    println!("get_bit_pos_sequence: {:?}", get_bit_pos_sequence(128, 114));
    println!("get_bit_pos_sequence: {:?}", get_bit_pos_sequence(16, 8));

    let n_bits = 16;
    let u_bits = 22;
    let x: Vec<u128> = (0..(1u64 << n_bits))
        .map(|x| ((111 * x) % ((1 << u_bits) - 1)) as u128)
        .collect();
    assert!(x.len() == BTreeSet::<u128>::from_iter(x.iter().copied()).len());

    let hash = Ruzic09AHash::new(&x, n_bits);
    let mut dst = vec![false; 1 << hash.output_bits];
    for v in x.iter() {
        let u = hash.query(*v);
        assert!(!dst[u as usize], "v {} u {}", v, u);
        dst[u as usize] = true;
    }
}

// Dictionary: start with 'unreduced' style

pub struct Ruzic09Dict {
    reduction: Ruzic09BReduction,
    hashes: Vec<Ruzic09AHash>,

    u_bits: u32,
    output_bits: u32,

    /** Table of key-value pairs indexed by the perfect hash construction */
    table: Vec<(u64, u64)>,
    /** A value which is not a valid key */
    non_key: u64,
}

impl Dict<u64, u64> for Ruzic09Dict {
    fn new(data: &[(u64, u64)], u_bits: u32) -> Ruzic09Dict {
        let keys: Vec<u64> = data.iter().map(|(k, _v)| *k).collect();

        let reduction = Ruzic09BReduction::new_fast(&keys, u_bits);
        drop(keys);

        let reduced_data: Vec<(u64, u64)> = data
            .iter()
            .map(|(k, v)| (reduction.apply(*k), *v))
            .collect();

        /* Minimum size for construction to _work_ is 10, but then only one bit is removed per hash
         * iteration; n_bits >= 13 is needed to ensure >= 4 bits/iteration. Ultimately, increasing
         * n_bits will speed up the low end, at the cost of memory overhead. */
        // TODO: try memory-parameterized benchmarking to find pareto-optimal hyperparameters across all dictionary types?
        const MIN_N_BITS: u32 = 10;
        let n_bits = next_log_power_of_two(reduced_data.len()).max(MIN_N_BITS);

        let (u_bits, output_bits, _psi_bits, _phi_bits, _f_bits, _r_bits) =
            get_r09a_hash_bits(n_bits);
        assert!(
            u_bits > output_bits,
            "n: {} u: {} output: {}",
            n_bits,
            u_bits,
            output_bits
        );

        /* Number of bits compressed per round of hashing */
        let extra_bits = u_bits - output_bits;

        let steps = if reduction.end_bits > output_bits {
            (reduction.end_bits - output_bits).div_ceil(extra_bits)
        } else {
            0
        };

        // println!("n: {} u: {} output: {}, steps: {}", n_bits, u_bits, output_bits, steps);
        let mut lead: Vec<u64> = reduced_data
            .iter()
            .map(|(k, _v)| *k & ((1 << u_bits) - 1))
            .collect();

        let mut hashes: Vec<Ruzic09AHash> = Vec::new();
        for round in 0..steps {
            let sub_keys0: Vec<u128> = lead
                .iter()
                .zip(reduced_data.iter())
                .map(|(f, (k, _v))| {
                    let next = (*k >> (round * extra_bits + output_bits)) & ((1 << extra_bits) - 1);
                    (*f as u128) | (next as u128) << output_bits
                })
                .collect();

            let mut sub_keys = sub_keys0.clone();
            // TODO: is deduplicating input keys necessary for Ruzic09AHash ?
            // TODO: extract into subfunction
            sub_keys.sort_unstable();
            let mut w = 1;
            for r in 1..sub_keys.len() {
                if sub_keys[r] != sub_keys[w - 1] {
                    sub_keys[w] = sub_keys[r];
                    w += 1;
                }
            }
            sub_keys.truncate(w);

            let hash = Ruzic09AHash::new(&sub_keys, n_bits);

            lead = sub_keys0.iter().map(|k| hash.query(*k)).collect();

            hashes.push(hash);
        }

        // TODO: deduplicate this / create GenericDict from PerfectHash

        // Identify a value which is _not_ a valid key.
        let mut is_key: Vec<bool> = vec![false; data.len()];
        for (k, _v) in data {
            if *k < data.len() as u64 {
                is_key[*k as usize] = true;
            }
        }

        let non_key = is_key.iter().position(|x| !x).unwrap_or(data.len()) as u64;
        // Fill table entries
        let mut table: Vec<(u64, u64)> = vec![(non_key, 0); 1 << output_bits];
        for (i, v) in lead.iter().enumerate() {
            table[*v as usize] = data[i];
        }

        Ruzic09Dict {
            reduction,
            hashes,
            u_bits,
            output_bits,
            table,
            non_key,
        }
    }
    fn query(&self, key: u64) -> Option<u64> {
        let d = self.reduction.apply(key);

        let extra_bits = self.u_bits - self.output_bits;

        let mut val = d & ((1 << self.output_bits) - 1);
        for (i, h) in self.hashes.iter().enumerate() {
            let nxt = (d >> ((i as u32) * extra_bits + self.output_bits)) & ((1 << extra_bits) - 1);
            val = h.query(val as u128 | ((nxt as u128) << self.output_bits));
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
fn test_ruzic09_dict() {
    let x: Vec<(u64, u64)> = (0..(1u64 << 8)).map(|x| (x.wrapping_pow(8), x)).collect();
    let hash = Ruzic09Dict::new(&x, u64::BITS);
    for (k, v) in x {
        assert!(Some(v) == hash.query(k));
    }
    let x: Vec<(u64, u64)> = (0..(1u64 << 10)).map(|x| (x.wrapping_pow(6), x)).collect();
    let hash = Ruzic09Dict::new(&x, u64::BITS);
    for (k, v) in x {
        assert!(Some(v) == hash.query(k));
    }
}

/* ---------------------------------------------------------------------------- */

/** A simple and very fast hash function.
 *
 * Typically `a` is an odd integer, and shift = u_bits - output_bits.
 *
 * However:
 * - Universes smaller than u64::BITS are handled by shifting 'a' left so that u64
 *   wrapping implements the mod-u for free, and then shifting the result back
 *   afterwards.
 * - When hashing empty or single element sets, we set `a = 0` because shifting
 *   by default wraps instead of saturating, so (a*k)>>64 doesn't return zero.
 *
 * For specific applications, other conventions may make sense (and might hint to
 * the compiler what the structure of `a` and `shift` are.) (But this could
 * so be done using assert_unchecked!...)
 */
#[derive(Clone, Copy)]
struct MultiplyShiftHash {
    a: u64,
    shift: u32, /* Stored as u32, but u8 would suffice. */
}
impl MultiplyShiftHash {
    #[inline(always)]
    fn query(&self, key: u64) -> u64 {
        key.wrapping_mul(self.a) >> self.shift
    }
    #[inline(always)]
    fn output_bits(&self) -> u32 {
        if self.a == 0 {
            0
        } else {
            u64::BITS - self.shift
        }
    }
}

/** Multiply shift construction optimized for small sets */
fn small_set_multiply_shift(keys: &[u64]) -> MultiplyShiftHash {
    assert!(keys.len() <= 2);
    if keys.len() <= 1 {
        return MultiplyShiftHash { a: 0, shift: 0 };
    };

    let (x, y) = (keys[0], keys[1]);
    assert!(x != y);
    let j = u64::BITS - (x ^ y).leading_zeros() - 1;
    MultiplyShiftHash {
        a: (1 << (u64::BITS - j - 1)),
        shift: u64::BITS - 1,
    }
}

#[test]
fn test_small_multiply_shift() {
    let x = [0, 1 << 17];
    let h = small_set_multiply_shift(&x);
    assert!(h.output_bits() == 1);
    assert!(h.query(x[0]) != h.query(x[1]));
}

/* ---------------------------------------------------------------------------- */

/** The Raman96 hash construction; construction time is roughly quadratic in the
 * input set size */
#[derive(Clone, Copy)]
pub struct Raman95Hash(MultiplyShiftHash);

fn compute_raman95_multiplier(keys: &[u64], u_bits: u32) -> (u64, u32) {
    // TODO: special case for n=1 and n=2
    let output_bits =
        next_log_power_of_two(keys.len().wrapping_mul(keys.len().wrapping_sub(1)) + 1).max(1);
    assert!(output_bits < u64::BITS);

    /* Shift value to apply to the multiplier */
    let u_compensation_shift = u64::BITS - u_bits;
    assert!(keys.iter().all(|k| util::saturating_shr(*k, u_bits) == 0));

    /** The number of extensions `a` of α for which
     * `a(x-y) mod (2^u) is in [-2^{u-o}+1,...2^{u-o}-1]`
     *
     * Note: Raman95/96 uses a different convention, in which
     * the multiplier is '2 α + 1' (i.e, α excludes the first bit.)
     * This adds a lot of +1/-1 terms to expressions, so we stay
     * with `α` being having least significant bit 1.
     */
    fn compute_delta(
        x: u64,
        y: u64,
        alpha: u64,
        alpha_bits: u32,
        u_bits: u32,
        output_bits: u32,
    ) -> u64 {
        assert!(alpha % 2 == 1);

        // TODO: handle sub-u64 wrapping (would save time if this hash construction is ever used with fewer input bits)
        assert!(x != y && u_bits == 64);

        let diff = x.wrapping_sub(y);
        let f = alpha.wrapping_mul(diff);

        let ep = (1u64 << (u_bits - output_bits)) - 1;

        let l: u64 = 0u64.wrapping_sub(ep).wrapping_sub(f);
        let u: u64 = ep.wrapping_sub(f);
        let i = diff.trailing_zeros();

        if alpha_bits + i >= u_bits {
            /* Case: a'(x-y) is _always_ 0, so test if 0 is inside the interval l..=u
             *
             * (Should be simple to handle as a batch: sort and two-pointer-scan the αx
             * sequence.)
             */
            // println!("path: fast exit: {}, alpha_bits={}", l > u, alpha_bits);
            if l > u {
                // except: case 'large'?
                return 1 << (u_bits - alpha_bits);
            }
            return 0;
        }

        /* Raman95, "Improved data structures for predecessor queries in integer sets" has
         * a typo: "simply reduces to counting the number integral multiples of X" has
         * X be 2^{b-|alpha|-i-1} in the paper, but X should be 2^{|alpha|+i+1} instead.
         * (Note: this implementation uses a slightly different alpha definition than Raman95/96) */
        let mult_bits = alpha_bits + i;
        let nmults = 1u64 << (u_bits - mult_bits);

        let nmultm1_lt_l = l.wrapping_sub(1) >> mult_bits;
        let nmultm1_lt_up1 = u >> mult_bits;

        /* Multiples of 2^{mult_bits} in l..=u */
        let mults_in_range = nmultm1_lt_up1.wrapping_sub(nmultm1_lt_l) % nmults;

        if mult_bits < u_bits - output_bits + 1 {
            /* Simple regime: multiples smaller than interval length */
            let est_count = 1 << (u_bits - output_bits + 1 - mult_bits);
            assert!(
                est_count >= mults_in_range && mults_in_range + 1 >= est_count,
                "{} {}",
                est_count,
                mults_in_range
            );
        } else {
            /* Interesting regime: multiples larger than interval length, so
             * the interval will contain either 0 or 1 multiples in range.
             *
             * The interval, rounded up, is [-2^{u-o},2^{u-o})-b(x-y) for
             * b = 2alpha+1, which contains a multiple of 2^{|a|+i+1} if
             * b(x-y)=bx-by is 2^{u-o} close to a multiple of 2^{|a|+i+1}.
             *
             * Also: since b is odd, the number of trailing zeros of b(x-y)
             * and of (x-y) is the same. So one can _preprocess_ the input vector
             * by premultiplying it with `b`.
             *
             * Is there any efficient way to count the number of bx-by which
             * are 2^{u-o} close to a multiple of 2^{|a|+{ctz(bx-by)}+1}?
             */
            assert!(f.trailing_zeros() == diff.trailing_zeros());
            assert!(mults_in_range <= 1);
        }

        // println!("path: mults in range: {}, {}..={}, mult bits {}", mults_in_range, l, u, mult_bits);
        let extensions_in_range = mults_in_range << i;
        return extensions_in_range;
    }

    let mut a = 1;
    let mut last_wt = None;
    /* With each bit chosen, the sum over all pairs of keys of the "bad extension count"
     * reduces to at most half the previous value. */
    // println!("{:?}", keys);
    for b in 1..u64::BITS {
        let mut wt0 = 0;
        let mut wt1 = 0;
        let a1 = a | (1 << b);
        for i in 0..keys.len() {
            for j in 0..i {
                let ki = keys[i] << u_compensation_shift;
                let kj = keys[j] << u_compensation_shift;

                // should have have keys[i]/keys[j] symmetry, for now
                let delta0 = compute_delta(ki, kj, a, b + 1, u64::BITS, output_bits);
                let delta1 = compute_delta(ki, kj, a1, b + 1, u64::BITS, output_bits);
                if delta0 == 0 {
                    assert!(
                        a.wrapping_mul(ki) >> (u64::BITS - output_bits)
                            != a.wrapping_mul(kj) >> (u64::BITS - output_bits),
                        "{} {} diff {} {} | {} {}",
                        a.wrapping_mul(ki) >> (u64::BITS - output_bits),
                        a.wrapping_mul(kj) >> (u64::BITS - output_bits),
                        a.wrapping_mul(ki.wrapping_sub(kj)) >> (u64::BITS - output_bits),
                        a.wrapping_mul(kj.wrapping_sub(ki)) >> (u64::BITS - output_bits),
                        a.wrapping_mul(ki.wrapping_sub(kj)),
                        a.wrapping_mul(kj.wrapping_sub(ki)),
                    );
                }
                if delta1 == 0 {
                    assert!(
                        a1.wrapping_mul(ki) >> (u64::BITS - output_bits)
                            != a1.wrapping_mul(kj) >> (u64::BITS - output_bits)
                    );
                }

                // also: delta0+delta1 = {delta with old a value}
                assert!(
                    delta0 <= (1 << (u64::BITS - b - 1)) && delta1 <= (1 << (u64::BITS - b - 1)),
                    "delta0 {} delta1 {} lim {}",
                    delta0,
                    delta1,
                    1u64 << (u64::BITS - b - 1)
                );
                wt0 += delta0 as u128;
                wt1 += delta1 as u128;
            }
        }
        if let Some(w) = last_wt {
            assert!(wt0 + wt1 == w, "{} + {} ?= {}", wt0, wt1, w);
        }
        if wt1 < wt0 {
            a = a1;
            last_wt = Some(wt1);
        } else {
            last_wt = Some(wt0);
        }
        // println!("a: {:064b}, b {}, wt0 {}, wt1 {}", a, b, wt0, wt1);
    }
    assert!(util::saturating_shr(a, u_bits) == 0);
    (a << u_compensation_shift, output_bits)
}

impl PerfectHash<u64, u64> for Raman95Hash {
    fn new(keys: &[u64], u_bits: u32) -> Self {
        if keys.len() <= 2 {
            return Raman95Hash(small_set_multiply_shift(keys));
        }
        let (a, output_bits) = compute_raman95_multiplier(keys, u_bits);

        Raman95Hash(MultiplyShiftHash {
            a,
            shift: (u64::BITS - output_bits),
        })
    }
    fn query(&self, key: u64) -> u64 {
        self.0.query(key)
    }
    fn output_bits(&self) -> u32 {
        self.0.output_bits()
    }
}

/** Dictionary constructed from [Raman95Hash] */
pub type Raman95Dict = HDict<Raman95Hash>;

#[cfg(test)]
fn test_perfect_hash<H: PerfectHash<u64, u64>>() {
    let mut sizes: Vec<usize> = vec![0, 1, 2, 3];
    for j in 2..10 {
        sizes.push(2 << j);
        sizes.push(3 << j);
    }
    sizes.reverse();

    let mut sizes_and_u: Vec<(usize, u32)> = sizes.iter().map(|s| (*s, u64::BITS)).collect();
    sizes_and_u.push((128, 32));
    sizes_and_u.push((16, 5));

    for (s, u_bits) in sizes_and_u {
        assert!(util::saturating_shr(s as u64, u_bits) == 0);
        let pow = u_bits / next_log_power_of_two(s + 2);
        let keys: Vec<u64> = (0..s as u64).map(|x| x.wrapping_pow(pow)).collect();
        assert!(keys.len() == BTreeSet::from_iter(keys.iter().copied()).len());
        let hash = H::new(&keys, u_bits);

        let mut outputs = Vec::new();
        for k in keys {
            outputs.push((hash.query(k), k));
        }
        outputs.sort_unstable();
        for w in outputs.windows(2) {
            assert!(
                w[0].0 != w[1].0,
                "hash collision at size {}: {}->{} and {}->{}",
                s,
                w[0].1,
                w[0].0,
                w[1].1,
                w[1].0
            );
        }
    }
}

#[test]
fn test_raman95_hash() {
    test_perfect_hash::<Raman95Hash>();
}

/** Same as [Raman95Hash], but hopefully with faster construction. */
// using a separate type
pub struct FastOMS(MultiplyShiftHash);
pub type OMSDict = HDict<FastOMS>;

impl PerfectHash<u64, u64> for FastOMS {
    fn new(keys: &[u64], u_bits: u32) -> Self {
        if keys.len() <= 2 {
            return FastOMS(small_set_multiply_shift(keys));
        }

        let output_bits =
            next_log_power_of_two(keys.len().wrapping_mul(keys.len().wrapping_sub(1)) + 1).max(1);
        assert!(output_bits < u64::BITS);

        FastOMS(oms_opt_batch_construction(keys, u_bits, output_bits))
    }
    fn query(&self, key: u64) -> u64 {
        self.0.query(key)
    }
    fn output_bits(&self) -> u32 {
        self.0.output_bits()
    }
}

#[cfg(test)]
struct OmsOptBitClass {
    /* Common suffix of all keys, of length `i`, where `i` is implicit */
    common_suffix: u64,
    /* Keys where bit `i+1` is 0, in arbitrary order */
    keys_0: Vec<u64>,
    /* Keys where bit `i+1` is 1, in arbitrary order */
    keys_1: Vec<u64>,
}

#[cfg(test)]
struct OmsOptAux {
    // Indexed by the common suffix length `i`.
    // All keys should be unique
    by_last_bits: Vec<BTreeMap<u64, OmsOptBitClass>>,
}

/** Build an index of keys by common suffix length. Requires: all keys to be unique. */
#[cfg(test)]
fn build_oms_opt_aux(keys: &[u64]) -> OmsOptAux {
    let mut aux = OmsOptAux {
        by_last_bits: Vec::new(),
    };
    for i in 0..u64::BITS {
        let mut table: BTreeMap<u64, OmsOptBitClass> = BTreeMap::new();
        for k in keys.iter() {
            let suffix = k & ((1 << i) - 1);
            let zbit = k & (1 << i) == 0;
            let entry = table.entry(suffix).or_insert(OmsOptBitClass {
                common_suffix: suffix,
                keys_0: Vec::new(),
                keys_1: Vec::new(),
            });
            if zbit {
                entry.keys_0.push(*k);
            } else {
                entry.keys_1.push(*k);
            }
        }
        // TODO: minor (practical) optimization: can entirely skip entries for which at least one of keys_0 or
        // keys_1 is empty because they will have no matches. (May want to skip this to keep behavior closer to
        // worst case.)
        aux.by_last_bits.push(table);
    }

    if true {
        /* Check that the index works. */
        for k in keys {
            let mut n_alt = 0;
            for i in 0..u64::BITS {
                let suffix = k & ((1 << i) - 1);
                let zbit = k & (1 << i) == 0;
                let entry = &aux.by_last_bits[i as usize][&suffix];
                assert!(entry.common_suffix == suffix);
                let alt_list = if zbit { &entry.keys_1 } else { &entry.keys_0 };
                for alt in alt_list {
                    assert!(alt.wrapping_sub(*k).trailing_zeros() == i);
                    n_alt += 1;
                }
            }
            assert!(n_alt == keys.len() - 1);
        }
    }

    aux
}

/** Like compute_delta() of the original Raman95 construction, except only checking for extensions that map
 * the difference to [0..2^{u-o}). (The other direction will be handled when this is called with ax/ay reversed.) */
#[cfg(test)]
fn compute_delta_mod(ax: u64, ay: u64, alpha_bits: u32, u_bits: u32, output_bits: u32) -> u64 {
    assert!(ax != ay && u_bits == 64);
    let diff = ax.wrapping_sub(ay);

    let ep = (1u64 << (u_bits - output_bits)) - 1;

    let l: u64 = 0u64.wrapping_sub(diff);
    let u: u64 = ep.wrapping_sub(diff);
    let i = diff.trailing_zeros();

    assert!(i < u_bits - output_bits);

    if alpha_bits + i >= u_bits {
        if l > u {
            return 1 << (u_bits - alpha_bits);
        }
        return 0;
    }
    let mult_bits = alpha_bits + i;
    let nmults = 1u64 << (u_bits - mult_bits);
    let nmultm1_lt_l = l.wrapping_sub(1) >> mult_bits;
    let nmultm1_lt_up1 = u >> mult_bits;
    let mults_in_range = nmultm1_lt_up1.wrapping_sub(nmultm1_lt_l) % nmults;

    if mult_bits <= u_bits - output_bits {
        let est_count = 1 << (u_bits - output_bits - mult_bits);
        assert!(mults_in_range == est_count);
    } else {
        assert!(mults_in_range <= 1);
    }
    mults_in_range << i
}

/* Variant of Raman95 using a pessimistic estimator for 'bad extensions'
 * that may be more amenable to batch optimizations, but with only iteration
 * order improvements enabled for easier debugging and comparison. */
#[cfg(test)]
fn oms_opt_slow_construction(keys: &[u64], u_bits: u32) -> MultiplyShiftHash {
    assert!(u_bits == u64::BITS);
    // 2n^2 space, because the bad extension count may be doubled with the modified criterion
    let output_bits = 1 + next_log_power_of_two(keys.len()).max(1) * 2;
    assert!(output_bits < u64::BITS);

    let aux = build_oms_opt_aux(keys);

    let mut a = 1;
    let mut last_wt = None;

    /* Note: for _batch_ operations, may not actually need the full BTreeMap
     * but here it is needed to quickly find look up keys matching this one */
    for b in 1..u_bits {
        let mut wt0 = 0;
        let mut wt1 = 0;
        let a1 = a | (1 << b);
        for k in keys {
            for i in 0..(u64::BITS - output_bits) {
                let suffix = k & ((1 << i) - 1);
                let zbit = k & (1 << i) == 0;
                let entry = &aux.by_last_bits[i as usize][&suffix];
                let alt_list = if zbit { &entry.keys_1 } else { &entry.keys_0 };
                for k2 in alt_list {
                    assert!(k.wrapping_sub(*k2).trailing_zeros() == i);
                    assert!(
                        k.wrapping_mul(a)
                            .wrapping_sub(k2.wrapping_mul(a))
                            .trailing_zeros()
                            == i
                    );

                    // delta(k,alt) is computed now, delta(alt, k) in a different iteration
                    let delta0 = compute_delta_mod(
                        k.wrapping_mul(a),
                        k2.wrapping_mul(a),
                        b + 1,
                        u_bits,
                        output_bits,
                    );
                    let delta1 = compute_delta_mod(
                        k.wrapping_mul(a1),
                        k2.wrapping_mul(a1),
                        b + 1,
                        u_bits,
                        output_bits,
                    );
                    wt0 += delta0 as u128;
                    wt1 += delta1 as u128;
                }
            }
        }
        if let Some(w) = last_wt {
            assert!(wt0 + wt1 == w, "{} + {} ?= {}", wt0, wt1, w);
        }
        if wt1 < wt0 {
            a = a1;
            last_wt = Some(wt1);
        } else {
            last_wt = Some(wt0);
        }
        // println!("a: {:064b}, b {}, wt0 {}, wt1 {}", a, b, wt0, wt1);
    }
    MultiplyShiftHash {
        a,
        shift: u64::BITS - output_bits,
    }
}

#[test]
fn test_oms_fast_hash() {
    test_perfect_hash::<FastOMS>();

    /* Check that the fast and slow constructions produce _exactly_ the same hash functions */
    let mut seqs: Vec<Vec<u64>> = Vec::new();

    for s in [500, 1000] {
        let pow = 64 / next_log_power_of_two(s + 2);
        let keys: Vec<u64> = (0..s as u64).map(|x| x.wrapping_pow(pow)).collect();
        assert!(keys.len() == BTreeSet::from_iter(keys.iter().copied()).len());
        seqs.push(keys);
    }
    for t in [300u64, 500u64] {
        let keys: Vec<u64> = (0..t).chain((1..t).map(|x| x.reverse_bits())).collect();
        assert!(keys.len() == BTreeSet::from_iter(keys.iter().copied()).len());
        seqs.push(keys);
    }

    for keys in seqs {
        let hash2 = oms_opt_slow_construction(&keys, u64::BITS);
        let hash1 = oms_opt_batch_construction(&keys, u64::BITS, hash2.output_bits());
        assert!(hash1.a == hash2.a && hash1.output_bits() == hash2.output_bits());
    }
}

/** Compute _one_ direction (L - R) of batch delta case where extension differences are spaced more widely
 *.than the target interval. Note: sorted_l/sorted_r must be sorted ascending by |k| k & mod_mask */
fn compute_oms_batch_ring_count(
    sorted_l: &[u64],
    sorted_r: &[u64],
    gap: u64,
    mod_mask: u64,
) -> u128 {
    let all_r_equal = sorted_r.windows(2).all(|x| x[0] == x[1]);
    if all_r_equal {
        /* Special case: all values are equal. (The two-pointer scanning approach uses when
         * values are not all equal does not work here.) */
        let mut count = 0u64;
        let kr = *sorted_r.first().unwrap();
        for kl in sorted_l.iter() {
            count = count.wrapping_add(((kl.wrapping_sub(kr) & mod_mask) < gap) as u64);
        }
        return (count as u128) * (sorted_r.len() as u128);
    }

    /* Main case: there are at least two distinct values, so a pointer scan works.
     *
     * Note: a two pointer scan approach is complicated by wraparound; see
     * examples showing how the desired region, marked with x, could evolve.
     *
     * xx-------xx
     * ---xxxx----
     * ----xxxx---
     * -----xxxx--
     *
     * -----xxxxxx
     * xxxxxxxxxxx
     * xxxxx------
     * -----------
     *
     * Target pairs are those for which: `(kl.wrapping_sub(*kr) & mod_mask) < gap`.
     * (So roughly, want to track all `kr` in (kl - gap, kl] mod m ).
     *
     * Since sorted_r is increasing, (kl-sorted_r) is _decreasing_; the smallest
     * value is on the right end, and the largest on the left. When kl increases,
     * `kl-sorted_r` increases and the region in [0,gap) moves rightwards.
     *
     * ---------
     *      xx
     *        x
     * xx      x
     *   xx
     *     x
     * ---------
     *
     * Right index: rightmost and "smallest" value of `kl-kr`.
     * (i.e, boundary where kl - kr >= 0.)
     *
     * Left index: leftmost and "largest" value of `kl - kr - gap`.
     * (i.e, boundary where kl - kr < gap, a.k.a. `kl - kr - gap < 0`)
     *
     * When: left = right+1, there are either no valid `kr` values matching `kl`,
     * or all values match.
     */

    let mut brute_count = 0u128;
    if false {
        // TODO: brute force to start with
        for kl in sorted_l.iter() {
            for kr in sorted_r.iter() {
                let diff = kl.wrapping_sub(*kr) & mod_mask;
                // println!("new test: {} {} -> {}", kl, kr, (diff < gap) as u32);
                if diff < gap {
                    brute_count += 1;
                }
            }
        }
    }

    let mut count = 0u128;

    let first_kl = *sorted_l.first().unwrap();

    fn get_right_pos(kl: u64, sorted_r: &[u64], mod_mask: u64) -> usize {
        // NOTE: above some size threshold, could use (a variant of) binary search instead
        // (but get_left_pos/get_right_pos take only ~20% of batch-ring count time)
        let mut right = 0;
        let mut min_diff = mod_mask; /* any value >= mod_mask is OK for initial value */
        let mut next_value = sorted_r[0];
        for (i, val) in sorted_r.iter().enumerate().rev() {
            if *val == next_value {
                /* Not a candidate for a rightmost extremum in a circularly sorted list */
                continue;
            }
            next_value = *val;

            let d = kl.wrapping_sub(*val) & mod_mask;
            if d <= min_diff {
                right = i;
                min_diff = d;
            }
        }
        right
    }

    fn get_left_pos(kl: u64, sorted_r: &[u64], gap: u64, mod_mask: u64) -> usize {
        let mut left = 0;
        let mut max_gap_diff = 0;
        let mut prev_value = *sorted_r.last().unwrap();
        for (i, val) in sorted_r.iter().enumerate() {
            if *val == prev_value {
                /* Not a candidate for a leftmost extremum in a circularly sorted list */
                continue;
            }
            prev_value = *val;
            let d = kl.wrapping_sub(*val).wrapping_sub(gap) & mod_mask;
            if d >= max_gap_diff {
                left = i;
                max_gap_diff = d;
            }
        }
        left
    }

    fn next_mod(val: usize, len: usize) -> usize {
        if val >= len - 1 {
            0
        } else {
            val + 1
        }
    }

    let mut left = get_left_pos(first_kl, &sorted_r, gap, mod_mask);
    let mut right = get_right_pos(first_kl, &sorted_r, mod_mask);

    let mut prev_left_value = sorted_r[(left + sorted_r.len() - 1) % sorted_r.len()];

    /* Note: on the first iteration, right and left pointers should not move */
    for kl in sorted_l.iter() {
        /* Update pointer positions for new (nondecreased) kl. For the right pointer,
         * as it minimizes, it is enough to move it right as long as that reduces the value */
        while (kl.wrapping_sub(sorted_r[next_mod(right, sorted_r.len())]) & mod_mask)
            <= (kl.wrapping_sub(sorted_r[right]) & mod_mask)
        {
            // println!("advance right");
            right = next_mod(right, sorted_r.len());
        }
        /* The left pointer maximizes a quantity, so the naive way to find that would be to iterate
         * in order of increasing values (leftwards), which is inefficient. Instead, move right _to
         * the next distinct value_ while the position to the left has a strictly larger quantity. */
        while (kl.wrapping_sub(prev_left_value).wrapping_sub(gap) & mod_mask)
            > (kl.wrapping_sub(sorted_r[left]).wrapping_sub(gap) & mod_mask)
        {
            // println!(
            //     "advance left: {} {}",
            //     (kl.wrapping_sub(
            //         sorted_r[(left + sorted_r.len() - 1) % sorted_r.len()]
            //     )
            //     .wrapping_sub(gap)
            //         & mod_mask),
            //     (kl.wrapping_sub(sorted_r[left]).wrapping_sub(gap) & mod_mask)
            // );

            let cval = sorted_r[left];
            while sorted_r[left] == cval {
                prev_left_value = sorted_r[left];
                left = next_mod(left, sorted_r.len());
            }
        }

        if false {
            let diffs: Vec<f64> = sorted_r
                .iter()
                .map(|r| (kl.wrapping_sub(*r) & mod_mask) as f64 / (gap as f64))
                .collect();
            let vals: Vec<bool> = diffs.iter().map(|d| *d < 1.0).collect();
            assert!(
                left == get_left_pos(*kl, &sorted_r, gap, mod_mask),
                "left {}, ideal {}, diffs {:?}, vals {:?}",
                left,
                get_left_pos(*kl, &sorted_r, gap, mod_mask),
                diffs,
                vals
            );
            assert!(
                right == get_right_pos(*kl, &sorted_r, mod_mask),
                "right {}, ideal {}, diffs {:?}, vals {:?}",
                right,
                get_right_pos(*kl, &sorted_r, mod_mask),
                diffs,
                vals
            );
        }

        /* local count is: (right - left + 1), which _may_ equal zero (with no matches) or count (with all matches) */
        let mut local_count = (sorted_r.len() + right + 1 - left) % sorted_r.len();
        if local_count == 0 {
            if (kl.wrapping_sub(sorted_r[right]) & mod_mask) < gap {
                // println!("pseudoempty");
                local_count += sorted_r.len();
            }
        }

        // println!("kl={}, scan: l={} r={}, idl={} local_count = {}, diffs {:?} vals {:?}", kl, left, right, get_left_pos(*kl, &sorted_r, gap, mod_mask),local_count, diffs, vals);

        if false {
            let mut brute_local = 0;
            for kr in sorted_r.iter() {
                let diff = kl.wrapping_sub(*kr) & mod_mask;
                if diff < gap {
                    brute_local += 1;
                }
            }

            assert!(
                brute_local == local_count,
                "brute {} local {}",
                brute_local,
                local_count,
            );
        }

        count = count.wrapping_add(local_count as u128);
    }

    if false {
        assert!(
            brute_count == count,
            "brute force {} scan {}",
            brute_count,
            count
        );
    }

    count
}

/** Count the number of "directed interactions" between keys_l and keys_r:
 * oriented pairs (x,y) of keys for which `Pr[a(x-y) in 2^{u-o}] > 0` for
 * extensions of `alpha`. */
fn compute_oms_batch_delta(
    keys_l: &[u64],
    keys_r: &[u64],
    scratch_l: &mut [u64],
    scratch_r: &mut [u64],
    alpha: u64,
    alpha_bits: u32,
    i_bits: u32,
    u_bits: u32,
    output_bits: u32,
) -> u128 {
    let mult_bits = alpha_bits + i_bits;
    assert!(mult_bits > u_bits - output_bits);

    /* Tricky case: extension differences are _less_ fine grained than the target interval.
     *
     * For how many pairs 'k_l, k_r' does `k_r - k_l + [0,2^{u-o})` contain a multiple
     * of 2^{alpha_bits+i}? To count these, mod by 'm = min(2^{alpha_bits+i},2^u)', and then count
     * the number of cases where `k_r - k_l + [0,2^{u-o})` contains 0 under this mod;
     * in other words, where `kl - kr mod m` is < 2^{u-o}`.
     *
     * Is there any way to amortize costs across function calls when mult_bits >= u_bits?
     * (This may be difficult to do since the lists might (or might not) change significantly
     * with each alpha update; but if this can be done, then one could significantly reduce
     * the cost of querying.)
     */
    let mod_mask = if mult_bits < u_bits {
        (1 << mult_bits) - 1
    } else {
        u64::MAX
    };

    if keys_l.is_empty() || keys_r.is_empty() {
        /* Note: to have more 'worst-case' like behavior, move this check after the sorting is done.
         * (The most average-case-friendly optimization may be to build an index of nontrivial
         * key/value pairs; on fully random input, most common lsb-prefixes are short so all long
         * prefixes can be entirely skipped.) */
        return 0;
    }

    /* Note: `mod_mask` is also applied during the ring count, so one could avoid
     * masking here, and apply the mask while sorting; doing this
     * would require adjusting the 'all_equal' case of [compute_oms_batch_ring_count]. */
    for (ak, k) in scratch_l.iter_mut().zip(keys_l.iter()) {
        *ak = k.wrapping_mul(alpha) & mod_mask;
    }
    for (ak, k) in scratch_r.iter_mut().zip(keys_r.iter()) {
        *ak = k.wrapping_mul(alpha) & mod_mask;
    }
    scratch_l.sort_unstable();
    scratch_r.sort_unstable();

    let gap = 1 << (u_bits - output_bits);

    // TODO: it may be worth it to replace linked scans with scan+binary search
    // when one side is much larger than the other. This should wait until the
    // base interaction count is optimized and/or structure of the level sets is
    // better understood

    let count_lr = compute_oms_batch_ring_count(&scratch_l, &scratch_r, gap, mod_mask);
    let count_rl = compute_oms_batch_ring_count(&scratch_r, &scratch_l, gap, mod_mask);
    let count = count_rl + count_lr;

    return count;
}

/** Sort array by bit-reversed entries. (This is a separate non-inlined function for easier cost tracking
 * and should not be called many times.)
 */
#[inline(never)]
fn bit_rev_sort(arr: &mut [u64]) {
    arr.sort_unstable_by_key(|x| x.reverse_bits());
}

/** Calculate the initial interaction counts for each difference level of the given key list.
 *
 * `rev_sorted_keys` should be sorted by reversed bits, and shifted by (u64::BITS - u_bits).
 *
 * Note: while this approach can be used to efficiently enumerate all O(n) distinct
 * common-prefix intervals with distinct 0/1 next bits, the sum of all interval lengths
 * ranges from O(n log n) to O(n b), so using it to efficiently iterate over intervals
 * for batch delta computation risks being an average-case optimization. (On the other hand,
 * if the sum of all interval lengths is Ω(n b), then there will be large 0/1 "unbalanced"
 * intervals for which the sum may be efficiently computable; it _may_ be possible to use
 * this to improve runtime to O(n (log n)^2), but this would only be a slight improvement
 * in practice since e.g. log n=20, b=64.)
 *
 * Runtime: O(n + b).
 */
#[inline(never)]
fn oms_base_interaction_count(
    rev_sorted_keys: &[u64],
    _u_bits: u32,
    output_bits: u32,
) -> Vec<u128> {
    let max_bit = u64::BITS - output_bits;
    let mut counters = vec![0u128; max_bit as usize];

    #[derive(Debug)]
    struct Transition {
        bit: u32,   /* index `i` of the least significant bit which is changing from 0 to 1 */
        pos: usize, /* index so that (rev_sorted_keys[j] >> bit) & 1 == 1 */
    }

    let mut stack: Vec<Transition> = Vec::new();
    stack.reserve(max_bit as usize);

    for (i_m_1, pair) in rev_sorted_keys.windows(2).enumerate() {
        let i = i_m_1 + 1;
        let (prev, next) = (pair[0], pair[1]);

        let common_bits = (prev ^ next).trailing_zeros().min(max_bit - 1);
        let prev_bit_value = ((prev >> common_bits) & 1) != 0;
        let next_bit_value = ((next >> common_bits) & 1) != 0;
        if common_bits < max_bit - 1 {
            assert!(!prev_bit_value && next_bit_value);
        } else {
            if prev_bit_value == next_bit_value {
                /* No transition, continue */
                continue;
            }
        }

        let t = Transition {
            bit: common_bits,
            pos: i,
        };

        /* Pop elements from stack, and update the count table */
        loop {
            if let Some(top) = stack.last() {
                /* Cannot transition from 0 to 1 multiple times in a row */
                assert!(top.bit != common_bits);
                if top.bit < common_bits {
                    break;
                }
            } else {
                break;
            }

            let top = stack.pop().unwrap();
            let segment_start = if let Some(s) = stack.last() { s.pos } else { 0 };
            let count0 = top.pos - segment_start;
            let count1 = i - top.pos;
            /* Factor two included because we count _directed_ interactions. */
            let product = 2 * (count0 as u128) * (count1 as u128);
            counters[top.bit as usize] = counters[top.bit as usize].wrapping_add(product);
        }

        stack.push(t);
    }

    /* Final stack cleanup. */
    while let Some(top) = stack.pop() {
        let segment_start = if let Some(s) = stack.last() { s.pos } else { 0 };
        let count0 = top.pos - segment_start;
        let count1 = rev_sorted_keys.len() - top.pos;
        let product = 2 * (count0 as u128) * (count1 as u128);
        counters[top.bit as usize] = counters[top.bit as usize].wrapping_add(product);
    }

    if false {
        let mut reference = vec![0u128; (u64::BITS - output_bits) as usize];
        for i in 0..(u64::BITS - output_bits) {
            let msk = (1 << i) - 1;
            let mut interval_start = 0;
            while interval_start < rev_sorted_keys.len() {
                let common_prefix = rev_sorted_keys[interval_start] & msk;

                let mut count1 = 0;
                let mut interval_end = interval_start;

                while interval_end < rev_sorted_keys.len()
                    && (rev_sorted_keys[interval_end] & msk) == common_prefix
                {
                    count1 += (rev_sorted_keys[interval_end] & (1 << i)) >> i;
                    interval_end += 1;
                }

                let count0 = (interval_end - interval_start) as u64 - count1;
                let pairs = 2 * (count0 as u128) * (count1 as u128);
                reference[i as usize] = reference[i as usize].checked_add(pairs).unwrap();
                interval_start = interval_end;
            }
        }

        println!("fast {:?}", counters);
        println!("slow {:?}", reference);
        assert!(counters == reference);
    }

    counters
}

/* Variant of Raman95 using a pessimistic estimator for 'bad extensions'
 * that may be more amenable to optimizations, with iteration order
 * improvements and batch optimizations, but no 'cross-bit' optimizations.
 *
 * Runtime: O(sort(n) * b (log n)) = O(n * b (log n)^2) with comparison sorts.
 * (It may be possible to improve this to O(n b^2) with radix sort; or
 * by combining sort operations between `i` passes, but `mod_mask` complicates
 * this, and the sorting operation is not clearly a bottleneck in practice;
 * rather, the O(b log n) operations per element are. Further improvements may be
 * possible (e.g: using the structure of the longest common prefix array).)
 */
#[inline(never)]
fn oms_opt_batch_construction(keys: &[u64], u_bits: u32, output_bits: u32) -> MultiplyShiftHash {
    assert!(keys.len() < usize::MAX / 2); /* For wrapping index calculations */

    let u_compensation_shift = u64::BITS - u_bits;

    /* To efficiently iterate over pairs of groups of keys whose lsb `i` bits match
     * (and whose `i+1`th bit differs), it suffices to _sort_ the key set and then
     * scan over the result. */
    let mut rev_sorted_keys_vec = Vec::from_iter(keys.iter().map(|k| k << u_compensation_shift));
    let _ = keys;
    /* Note: if the source key set is _fixed_, then it may be practical to re-sort
     * on every `b` iteration and multiply by `a` in-place. Or alternatively: _first_
     * multiply by 'a', and _then_ sort the result by reversed bits, since
     * `a` multiplication preserves lsb-prefix length for pairs. Although the
     * difference isn't critical. */
    bit_rev_sort(&mut rev_sorted_keys_vec);
    let rev_sorted_keys: &[u64] = &rev_sorted_keys_vec;

    /* Scratch storage table in which to write multiplied/sorted data */
    let mut scratch = vec![0u64; rev_sorted_keys.len()];

    let mut a = 1;
    // let mut last_wt = None;

    let base_counts = oms_base_interaction_count(rev_sorted_keys, u_bits, output_bits);
    let mut last_interaction_counts = base_counts.clone();

    let mut last_wt = None;
    /* Bits in 'a' higher than or at u_bits have no effect on the output */
    for b in 0..u_bits {
        /* Calculate the "weight" (total bad extension mass) of the extension of `a` by a 0-bit. This,
         * plus the weight of the 1-bit extension, should equal the weight from the previous
         * iteration. (Except in first round, where we find the initial weight of `a`.) */
        let mut wt = 0;

        let mut interaction_counts = last_interaction_counts.clone(); // vec![0u128; (u64::BITS - output_bits) as usize];

        let alpha_bits = b + 1;

        /* Can entirely ignore keys matching on the bottom `u-o` keys, because multiplying
         * them by 'a' never causes a collision */
        for i in 0..(u64::BITS - output_bits) {
            let extensions_per_interaction: u128 = if alpha_bits + i <= u64::BITS - output_bits {
                1 << (u64::BITS - output_bits - alpha_bits)
            } else if alpha_bits + i < u64::BITS {
                1 << i
            } else {
                1 << (u64::BITS - alpha_bits)
            };

            if alpha_bits + i <= u64::BITS - output_bits {
                /* These counts do not change since alpha has not grown large enough to
                 * bring level i differences into the top `o` bits. */
                wt += base_counts[i as usize] * extensions_per_interaction;
                continue;
            }

            if alpha_bits + i > u64::BITS + 1 && true {
                /* These counts do not change because further extensions to `alpha` only
                 * differently affect bits that are masked off by mod 2^u? */
                wt += interaction_counts[i as usize] * extensions_per_interaction;
                continue;
            }

            let mut interactions = 0;
            let msk = (1 << i) - 1;
            let mut interval_start = 0;
            while interval_start < rev_sorted_keys.len() {
                let common_prefix = rev_sorted_keys[interval_start] & msk;

                let mut count1 = 0;
                let mut interval_end = interval_start;

                // TODO: use an enumerated &iterator? or sentinel? Or are bounds checks already eliminated
                // TODO: the most practical thing may be to build an index, to avoid scanning too many
                // unpaired or tiny intervals.
                while interval_end < rev_sorted_keys.len()
                    && (rev_sorted_keys[interval_end] & msk) == common_prefix
                {
                    count1 += (rev_sorted_keys[interval_end] & (1 << i)) >> i;
                    interval_end += 1;
                }

                let count0 = (interval_end - interval_start) - count1 as usize;

                let all_keys = &rev_sorted_keys[interval_start..interval_end];
                let (keys0, keys1) = all_keys.split_at(count0);

                /* Possible average-case optimization: use scratch[..count0+count1] for slightly better
                 * physical memory locality. */
                let scratch_region = &mut scratch[interval_start..interval_end];
                let (scratch0, scratch1) = scratch_region.split_at_mut(count0);

                interactions += compute_oms_batch_delta(
                    &keys0,
                    &keys1,
                    scratch0,
                    scratch1,
                    a,
                    alpha_bits,
                    i,
                    u64::BITS,
                    output_bits,
                );

                interval_start = interval_end;
            }

            interaction_counts[i as usize] = interactions;
            if alpha_bits + i <= u64::BITS - output_bits {
                assert!(interaction_counts[i as usize] == base_counts[i as usize]);
            }

            if alpha_bits + i > u64::BITS + 1 {
                assert!(
                    interaction_counts[i as usize] == last_interaction_counts[i as usize],
                    "{} -> {}",
                    last_interaction_counts[i as usize],
                    interaction_counts[i as usize]
                );
            }

            // todo: do interactions per level always decrease? (usually _but not always_)
            if false {
                if interaction_counts[i as usize] > last_interaction_counts[i as usize] {
                    println!("u {} increase in count at i={}; alpha_bits + i={}, u_bits-output_bits={}, from {} to {}", u_bits, i,
                            alpha_bits + i, u_bits - output_bits, last_interaction_counts[i as usize], interaction_counts[i as usize]);
                    // Note: since there are O(n) intervals, could record change in interaction counts per-interval
                }
            }

            wt += interaction_counts[i as usize] * extensions_per_interaction;
        }
        // println!("interaction counts: {:?}", interaction_counts);
        last_interaction_counts = interaction_counts;

        if let Some(lwt) = last_wt {
            /* Chose the sub-average branch. */
            // wt  = lwt - wt
            if lwt - wt < wt {
                a |= 1 << b;
                last_wt = Some(lwt - wt);
            } else {
                last_wt = Some(wt);
            }
        } else {
            last_wt = Some(wt);
        }
    }
    assert!(util::saturating_shr(a, u_bits) == 0);
    MultiplyShiftHash {
        shift: u64::BITS - output_bits,
        a: a << u_compensation_shift,
    }
}

/* ---------------------------------------------------------------------------- */

/**
 * Universe reduction using the OMS hash construction, followed by HMPQuadHash.
 */
pub struct OMSxHMP01Hash {
    reduction: FastOMS,
    hash: HMPQuadHash,
}
pub type OMSxHMP01Dict = HDict<OMSxHMP01Hash>;

impl PerfectHash<u64, u64> for OMSxHMP01Hash {
    fn new(keys: &[u64], u_bits: u32) -> OMSxHMP01Hash {
        let reduction = FastOMS::new(keys, u_bits);
        let half_bits = reduction.output_bits().div_ceil(2);
        let nbits = next_log_power_of_two(keys.len());
        let r_bits = half_bits.max(nbits + 1);
        let reduced_keys: Vec<(u64, u64)> = keys
            .iter()
            .map(|k| {
                let v = reduction.query(*k);
                (v >> r_bits, v & ((1 << r_bits) - 1))
            })
            .collect();
        let hash = HMPQuadHash::new(&reduced_keys, r_bits);
        OMSxHMP01Hash { reduction, hash }
    }
    fn query(&self, key: u64) -> u64 {
        let v = self.reduction.query(key);
        self.hash
            .query(v >> self.hash.r_bits, v & ((1 << self.hash.r_bits) - 1))
    }
    fn output_bits(&self) -> u32 {
        self.hash.r_bits
    }
}

#[test]
fn test_omsxhmp01_hash() {
    test_perfect_hash::<OMSxHMP01Hash>();
}

/* ---------------------------------------------------------------------------- */

/** The FKS84 hash construction using the fast OMS construction for the primary hash, and Raman95
 * for the secondary hash (because while the latter takes quadratic time, it is a bit faster
 * than the batch construction on small inputs.)
 *
 * NOTE: it may be worth switching between quadratic/near-linear constructions on large inputs,
 * and/or special casing O(1) sized inputs to save space.
 */
pub struct OMSxFKSHash {
    primary_hash: MultiplyShiftHash,
    // secondary hashes + base offsets. Note: the u_bits values for these are all the same
    secondary_hashes: Vec<(MultiplyShiftHash, u64)>,
    output_bits: u32,
}
pub type OMSxFKSDict = HDict<OMSxFKSHash>;

impl PerfectHash<u64, u64> for OMSxFKSHash {
    fn new(keys: &[u64], u_bits: u32) -> Self {
        /* Optimal number of bits in the secondary table. See analysis.pdf
         * for the calculation. This assumes size=1 buckets are efficiently packed
         * into gaps. */
        let bytes_per_leaf = std::mem::size_of::<(MultiplyShiftHash, u64)>();
        let bytes_per_output = std::mem::size_of::<(u64, u64)>();
        let lognm1 = next_log_power_of_two(keys.len().max(1) - 1);
        let num = 2
            * (bytes_per_output as u128)
            * (keys.len() as u128)
            * ((keys.len().max(1) - 1) as u128);
        let denom = bytes_per_leaf as u128;
        // todo: implement a simple integer square root
        let logratio = next_log_power_of_two(((num / denom) as f64).sqrt() as usize);
        let s_bits = lognm1.min(logratio).max(2); // ensure output_bits >= 1

        let max_collisions = (keys.len() as u128) * ((keys.len().max(1) - 1) as u128) >> s_bits;
        let output_bits = next_log_power_of_two(keys.len())
            .max(next_log_power_of_two(4 * max_collisions as usize));

        let primary_hash = oms_opt_batch_construction(keys, u_bits, s_bits);
        let mut buckets: Vec<Vec<u64>> = Vec::new();
        buckets.resize_with(1 << s_bits, Vec::new);
        for k in keys {
            buckets[primary_hash.query(*k) as usize].push(*k);
        }

        /* First place size >= 2 buckets, and then place size 1 buckets in the gaps */
        let null_hash = Raman95Hash::new(&[], 0).0;
        let mut secondary_hashes = vec![(null_hash, 0); 1 << s_bits];
        let mut outputs_taken = vec![false; 1 << output_bits]; // todo: use bitset

        let mut bucket_offset = 0;
        for (i, list) in buckets.iter().enumerate() {
            if list.len() <= 1 {
                continue;
            }
            let hash = Raman95Hash::new(&list, u_bits).0;
            let hash_size = 1u64 << hash.output_bits();
            for key in list.iter() {
                outputs_taken[(bucket_offset + hash.query(*key)) as usize] = true;
            }
            secondary_hashes[i] = (hash, bucket_offset);
            bucket_offset += hash_size;
        }

        let mut free_offset = 0;
        for (i, list) in buckets.iter().enumerate() {
            if list.len() != 1 {
                continue;
            }
            let hash = Raman95Hash::new(&list, u_bits).0;
            assert!(hash.output_bits() == 0);
            while outputs_taken[free_offset] {
                free_offset += 1;
            }
            outputs_taken[free_offset] = true;
            secondary_hashes[i] = (hash, free_offset as u64);
            free_offset += 1;
        }

        OMSxFKSHash {
            primary_hash,
            secondary_hashes,
            output_bits,
        }
    }
    fn output_bits(&self) -> u32 {
        self.output_bits
    }

    fn query(&self, key: u64) -> u64 {
        let index = self.primary_hash.query(key);
        let (hash2, offset) = &self.secondary_hashes[index as usize];
        /* Note: could also use the lower bit portion of the primary hash
         * as the secondary key. This would have fewer bits and be slightly
         * faster to construct, but may slow down evaluation. */
        let index2 = hash2.query(key);
        index2 + offset
    }
}

#[test]
fn test_omsxfks_hash() {
    test_perfect_hash::<OMSxFKSHash>();
}

/* ---------------------------------------------------------------------------- */

/** A perfect hash using two multiplications followed by a table lookup.
 *
 * The double displacement perfect hashing stage of [OMSxHMP01Hash] uses two displacement steps
 * (each with a large lookup table), where the first reduces the number of collisions and the
 * second completes the perfect hashing. The OMS hash can be used again in place of the first table,
 * because it also is collision reducing, and it has the nice property that the map `ax mod 2^u`
 * is 1:1, so the top `log n` bits (which are low-collision hashed) and the bottom `log n` bits
 * together uniquely identify a key.
 *
 * Open question: can this design be made to work with a single multiplication?
 */
pub struct OMSxDispHash {
    a1: u64,
    mask: u64,
    a2: u64,
    shift_hi: u32,
    shift_lo: u32,
    displacements: Vec<u64>,
    output_bits: u32,
}
pub type OMSxDispDict = HDict<OMSxDispHash>;

impl PerfectHash<u64, u64> for OMSxDispHash {
    fn new(keys: &[u64], u_bits: u32) -> Self {
        /* First multiplication parameters are straightforward: reduce the space as far as possible
         * without collisions */
        let s_bits =
            next_log_power_of_two(keys.len().wrapping_mul(keys.len().wrapping_sub(1)) + 1).max(1);
        assert!(s_bits < u64::BITS);
        let primary_hash = oms_opt_batch_construction(keys, u_bits, s_bits);
        let mask = u64::MAX ^ ((1 << primary_hash.shift) - 1);
        let reduced_keys = Vec::from_iter(keys.iter().map(|k| primary_hash.query(*k)));

        /* Compute hash and table parameters to minimize total space
         * when used to implement a dictionary. */
        let bytes_per_output = std::mem::size_of::<(u64, u64)>();
        let bytes_per_table_entry = std::mem::size_of::<u64>();
        let w = next_log_power_of_two(4 * keys.len() * keys.len().saturating_sub(1));
        let n_bits = next_log_power_of_two(keys.len());
        let log_ratio = next_log_power_of_two(bytes_per_table_entry.div_ceil(2 * bytes_per_output));
        let ideal_r = (log_ratio + w).div_ceil(2);
        /* Number of output bits */
        let r_bits = ideal_r.clamp(n_bits, w);
        /* Number of table index bits */
        let t_bits = w - r_bits;
        /* Number of bits to XOR with (may be equal to or less than `r`) */
        let lo_bits = s_bits.checked_sub(t_bits).unwrap();

        let secondary_hash = oms_opt_batch_construction(&reduced_keys, s_bits, t_bits);

        if t_bits == 0 {
            /* Special case: the displacement table is not needed.
             *
             * However, the specific shifts used to query can not shift the high bit region
             * to zero, so instead zero the low bits and use an identity displacement map;
             * this special case only occurs for tiny `n` so this is not expensive.
             */
            return OMSxDispHash {
                a1: primary_hash.a,
                mask,
                a2: 1,
                shift_hi: u64::BITS - s_bits,
                shift_lo: u64::BITS - s_bits,
                displacements: (0..(1 << s_bits)).collect(),
                output_bits: s_bits,
            };
        }

        let split_keys: Vec<(u64, u64)> = reduced_keys
            .iter()
            .map(|k| {
                let d = (secondary_hash.a.wrapping_mul(*k)) >> (u64::BITS - s_bits);

                assert!(*k >> s_bits == 0 && d >> s_bits == 0);
                let hi = d >> lo_bits;
                let lo = d & ((1 << lo_bits) - 1);
                assert!(hi >> r_bits == 0);
                assert!(lo >> r_bits == 0);
                (hi, lo)
            })
            .collect();

        // Test variant: explicitly verify that collision counts are as expected?

        let displacements = compute_displacement(&split_keys, t_bits, r_bits);

        let ret = OMSxDispHash {
            a1: primary_hash.a,
            mask,
            a2: secondary_hash.a >> (u64::BITS - s_bits),
            shift_hi: u64::BITS - t_bits,
            shift_lo: u64::BITS - s_bits,
            displacements,
            output_bits: r_bits,
        };

        if false {
            for key in keys.iter() {
                let reduced = key.wrapping_mul(ret.a1) & ret.mask;
                let decollided = reduced.wrapping_mul(ret.a2);
                let hi = decollided >> ret.shift_hi;
                let lo = (decollided >> ret.shift_lo) & ((1 << (ret.shift_hi - ret.shift_lo)) - 1);

                let red = primary_hash.query(*key);
                let decol = (secondary_hash.a.wrapping_mul(red)) >> (u64::BITS - s_bits);
                let hi2 = decol >> lo_bits;
                let lo2 = decol & ((1 << lo_bits) - 1);
                assert!(
                    hi == hi2 && lo == lo2,
                    "{:64b} {} {} | {:64b} {} {}",
                    decollided >> ret.shift_lo,
                    hi,
                    lo,
                    decol,
                    hi2,
                    lo2
                );
            }

            let mut check: BTreeSet<u64> = BTreeSet::new();
            for k in split_keys.iter() {
                check.insert(k.0 | (k.1 << r_bits));
            }
            assert!(check.len() == split_keys.len());
        }

        ret
    }
    fn output_bits(&self) -> u32 {
        self.output_bits
    }

    fn query(&self, key: u64) -> u64 {
        let reduced = key.wrapping_mul(self.a1) & self.mask;
        let decollided = reduced.wrapping_mul(self.a2);
        let hi = decollided >> self.shift_hi;
        let lo = (decollided >> self.shift_lo) & ((1 << (self.shift_hi - self.shift_lo)) - 1);
        self.displacements[hi as usize] ^ lo
    }
}

#[test]
fn test_omsxdisp_hash() {
    test_perfect_hash::<OMSxDispHash>();
}
