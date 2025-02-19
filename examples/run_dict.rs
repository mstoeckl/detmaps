/* SPDX-License-Identifier: MPL-2.0 */
/*! Test program: build and query a dictionary */

use detmaps::*;
use std::time::Instant;

/** Generate n pseudo-random samples in 0..=u32::MAX _without_ repetition. */
fn make_random_sampled_32(n: u64, seed: u64) -> Vec<(u64, u64)> {
    assert!(n.is_power_of_two());
    let p = (seed.wrapping_mul(0x9e3779b97f4a7c16)) | 0x1;
    /* Now n and p are coprime, so (x*p)%n is a permutation. */
    (0..n)
        .map(|x| (((x as u128 * p as u128) % (1u128 << 32)) as u64, x))
        .collect()
}

/** Generate n pseudo-random samples from the first n entries of `make_random_sampled_32`, _without_ repetition. */
fn make_random_order_32(n: u64, seed_vals: u64, seed_order: u64) -> Vec<u64> {
    let vals = make_random_sampled_32(n, seed_vals);

    assert!(n.is_power_of_two());
    let p = (seed_order.wrapping_mul(0x9e3779b97f4a7c16)) | 0x1;
    /* Now n and p are coprime, so (x*p)%n is a permutation. */
    (0..n)
        .map(|x| vals[((x as u128 * p as u128) % (n as u128)) as usize].0)
        .collect()
}

#[inline(never)]
fn do_par_queries<D: Dict<u64, u64>>(dict: &D, queries: &[u64]) {
    let mut x = 0;
    for q in queries.iter() {
        x += dict.query(*q).unwrap();
    }
    std::hint::black_box(x);
}

fn main() {
    // TODO: control dictionary type, size, pattern through argv

    let sz = 1 << 18;
    let data = &make_random_sampled_32(sz, 1);
    let queries = std::hint::black_box(make_random_order_32(sz, 1, 1111));
    let t0 = Instant::now();

    let dict = std::hint::black_box(HagerupMP01Dict::new(&data));
    let t1 = Instant::now();
    do_par_queries(&dict, &queries);
    let t2 = Instant::now();
    println!(
        "size: {}; construction time: {} secs; query time: {} secs",
        sz,
        t1.duration_since(t0).as_secs_f64(),
        t2.duration_since(t1).as_secs_f64()
    )
}
