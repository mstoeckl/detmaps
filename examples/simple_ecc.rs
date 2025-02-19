/* SPDX-License-Identifier: MPL-2.0 */
/*! Test program: find a simple error correcting code for e.g. u32->u128 */

use detmaps::util::clmul;
use rayon::prelude::*;

fn xorshift(mut x: u64) -> u64 {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

fn main() {
    let mut key = 0x1;

    /* Current best output, after running for a ~10 hours:
     *
     * (got 26-28 at trials 0/1/2, got 29 at trial 27, )
     *
     * trial 987: record (4989641109508346571, 3213968901338125373) 30/128; current: (674191574824813287, 14832925326557861010) 28/128
     *
     * Best result up to trial=10000 is still 30/128.
     *
     * Running another factor of ~10 longer may get up to 31/128 (= outputs in [n^5])
     */

    let mut best = (0, 0);
    let mut best_dist = 0;
    for trial in 0..1000000 {
        // Use xorshift* generator
        key = xorshift(key);
        let k1 = key * 0x2545F4914F6CDD1D;
        key = xorshift(key);
        let k2 = key * 0x2545F4914F6CDD1D;

        if trial < 900 {
            // skip already checked keys
            continue;
        }

        // let mut min_dist = u32::MAX;
        // (z ★ i) is bilinear w.r.t. xor, so (a ★ i) ^ (b ★ i) = (a^b) ★ i
        // and thus the minimum distance equals the minimum codeword weight
        let min_dist = if true {
            (1..=u32::MAX)
                .into_par_iter()
                .map(|i| {
                    let e = clmul(i as u64, k1) ^ (clmul(i as u64, k2) << 64);
                    e.count_ones()
                })
                .reduce(|| u32::MAX, |a, b| std::cmp::min(a, b))
        } else {
            let mut min_dist = u32::MAX;
            for i in 0..=u32::MAX {
                if i == 0 {
                    continue;
                }
                let e = clmul(i as u64, k1) ^ (clmul(i as u64, k2) << 64);
                min_dist = min_dist.min(e.count_ones());
            }
            min_dist
        };

        if min_dist > best_dist {
            best = (k1, k2);
            best_dist = min_dist;
        }

        let out_bits = 128;
        println!(
            "trial {}: record {:?} {}/{}; current: {:?} {}/{}",
            trial,
            best,
            best_dist,
            out_bits,
            (k1, k2),
            min_dist,
            out_bits
        );
    }
}
