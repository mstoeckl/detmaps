/* SPDX-License-Identifier: MPL-2.0 */
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::hash::Hash;

pub mod util;

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

/* ---------------------------------------------------------------------------- */

struct Ruzic09BReduction {
    // Multipliers; with each mask-multiply-shift-add step, domain size is reduced by half,
    // from 64 to 32 to 16 bits.
    steps: Vec<(u64, u32)>,
    // Number of bits of the final output
    end_bits: u32,
}

fn find_good_multiplier_fast(keys: &[u64], s: u32, max_mult_bits: u32) -> u64 {
    // binary search to find multiplier
    let mut low = 1u64; // skip 0, it makes collisions
    let mut high = 1 << max_mult_bits;

    fn est_bad_parameters(keys: &[u64], s: u32, low: u64, high: u64) -> u64 {
        // TODO: this may be oversimplified and thus incorrect in some cases
        // (e.g.: how should duplicates be handled?)

        let mut phi_indices: Vec<usize> = (0..keys.len()).collect();
        let mut f_indices: Vec<usize> = (0..keys.len()).collect();
        // So: does unstable sorting just overestimate?
        phi_indices.sort_unstable_by_key(|x| (keys[*x] >> s) + low * (keys[*x] & ((1 << s) - 1)));
        f_indices.sort_unstable_by_key(|x| (keys[*x] >> s) + high * (keys[*x] & ((1 << s) - 1)));

        // This overestimates inversions by up to a factor of 2
        let mut est_invs = 0;
        for (i_phi, i_f) in phi_indices.iter().zip(f_indices.iter()) {
            est_invs += i_phi.abs_diff(*i_f) as u64;
        }
        2 * est_invs
    }

    // Does using n(n-1)/2 help in any case?
    let mut bad_param_est = (keys.len().checked_mul(keys.len()).unwrap() / 2) as u64;
    while low < high {
        let mid = low + (high - low) / 2;

        let m1est = est_bad_parameters(keys, s, low, mid);
        /* Each iteration should reduce the estimate to <= 2/3 of the previous value */
        if m1est <= (2 * bad_param_est) / 3 {
            high = mid;
        } else {
            low = mid;
        }
        bad_param_est = (2 * bad_param_est) / 3;
    }
    assert!(bad_param_est == 0);

    low
}

impl Ruzic09BReduction {
    fn new_fast(kvs: &[(u64, u64)]) -> Ruzic09BReduction {
        let mut keys: Vec<u64> = kvs.iter().map(|x| x.0).collect();
        let n_bits = 64 - keys.len().leading_zeros();
        let mut u_bits = 64;
        let max_mult_bits = n_bits * 4; // more precisely, 2/log2(3/2) = 3.419

        // println!("n {} u {} mm {}", n_bits, u_bits, max_mult_bits);

        // Number of bits approaches 4*n_bits
        let mut mults = Vec::new();
        while u_bits > max_mult_bits + n_bits {
            let s = (u_bits - max_mult_bits) / 2;
            // println!("n {} u {} s {}", n_bits, u_bits, s);
            let a = find_good_multiplier_fast(&keys, s, max_mult_bits);
            mults.push((a, s));
            for k in keys.iter_mut() {
                *k = (*k >> s) + a * (*k & ((1 << s) - 1));
            }
            if false {
                // Verify multiplier actually works
                let mut b = BTreeSet::new();
                for k in keys.iter() {
                    assert!(b.insert(k), "duplicate value: {}", k);
                }
            }

            u_bits = s + max_mult_bits;
        }

        Ruzic09BReduction {
            steps: mults,
            end_bits: u_bits,
        }
    }
    #[allow(dead_code)]
    fn new_precise(_kvs: &[(u64, u64)]) -> Ruzic09BReduction {
        // Why isn't O(n^2) possible?

        /* There have been improvements to the problem of counting the number of permutation inversions;
         * since Ružić09 was written. See e.g. Chan & Pǎtraşcu 2010,
         * "Counting Inversions, Offline Orthogonal Range Coutning, and Related Problems" */
        todo!();
    }

    fn apply(&self, mut key: u64) -> u64 {
        for (a, s) in self.steps.iter() {
            key = (key >> s) + a * (key & ((1 << s) - 1));
        }
        key
    }
}

pub struct R09BxHMP01Dict {
    reduction: Ruzic09BReduction,

    r_bits: u32,
    // TODO: use a binary tree merge instead of a sequence of hashes
    hashes: Vec<HMPQuadHash>,
    /** Table of key-value pairs indexed by the perfect hash construction */
    table: Vec<(u64, u64)>,
    /** A value which is not a valid key */
    non_key: u64,
}

impl Dict<u64, u64> for R09BxHMP01Dict {
    fn new(data: &[(u64, u64)]) -> R09BxHMP01Dict {
        let reduction = Ruzic09BReduction::new_fast(data);

        let reduced_data: Vec<(u64, u64)> = data
            .iter()
            .map(|(k, v)| (reduction.apply(*k), *v))
            .collect();

        let n_bits = reduced_data
            .len()
            .checked_next_power_of_two()
            .unwrap()
            .trailing_zeros();
        let r_bits = n_bits + 1;
        let steps = reduction.end_bits.max(1).div_ceil(r_bits) - 1;

        let mut lead: Vec<u64> = reduced_data
            .iter()
            .map(|(k, _v)| *k & ((1 << r_bits) - 1))
            .collect();

        let mut hashes: Vec<HMPQuadHash> = Vec::new();
        for round in 0..steps {
            let mut pairs: Vec<(u64, u64)> = lead
                .iter()
                .zip(reduced_data.iter())
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
                .zip(reduced_data.iter())
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
        let r_bits = hashes[0].r_bits;
        let mut table: Vec<(u64, u64)> = vec![(non_key, 0); 1 << r_bits];
        for (i, v) in lead.iter().enumerate() {
            table[*v as usize] = data[i];
        }

        R09BxHMP01Dict {
            reduction,
            r_bits,
            hashes,
            table,
            non_key,
        }
    }
    fn query(&self, key: u64) -> Option<u64> {
        let d = self.reduction.apply(key);
        let mut val = d & ((1 << self.r_bits) - 1);
        // TODO: better construction than iterative
        for (i, h) in self.hashes.iter().enumerate() {
            let nxt = (d >> ((i as u32 + 1) * (self.r_bits))) & ((1 << self.r_bits) - 1);
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
fn test_r09xhmp01_dict() {
    let x: Vec<(u64, u64)> = (0..(u8::MAX as u64)).map(|x| (x.pow(8), x)).collect();
    let hash = R09BxHMP01Dict::new(&x);
    for (k, v) in x {
        assert!(Some(v) == hash.query(k));
    }
    let x: Vec<(u64, u64)> = (0..(1u64 << 10)).map(|x| (x.pow(6), x)).collect();
    let hash = R09BxHMP01Dict::new(&x);
    for (k, v) in x {
        assert!(Some(v) == hash.query(k));
    }
}

/* ---------------------------------------------------------------------------- */

/** Deterministic variant of tabulation hashing from [N]^k to [N^2], followed
 * by double displacement hashing for the last [N^2] -> [N] step.
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

fn next_log_power_of_two(v: usize) -> u32 {
    usize::BITS - v.leading_zeros()
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

    // outputs_by_rows ; want outputs_by_values....
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
    fn new(data: &[(u64, u64)]) -> Self {
        let n_bits = next_log_power_of_two(data.len());
        let r_bits = n_bits + 1;

        let mut xor_table = Vec::new();
        let mask_kp01 = if 2 * r_bits < u64::BITS {
            (1 << (2 * r_bits)) - 1
        } else {
            u64::MAX
        };
        let mut comp_keys: Vec<u128> = data.iter().map(|x| (x.0 & mask_kp01) as u128).collect();

        for i in 2..u64::BITS.div_ceil(r_bits) {
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
    let hash = XorReducedDict::new(&x);
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
    let hash = XorReducedDict::new(&x);
    for (k, v) in x {
        assert!(Some(v) == hash.query(k));
    }
}
