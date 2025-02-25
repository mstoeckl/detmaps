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
fn do_virt_queries(dict: &GenericDict, queries: &[u64]) {
    let mut x = 0;
    for q in queries.iter() {
        x += dict.query(*q).unwrap();
    }
    std::hint::black_box(x);
}

enum GenericDict {
    Binsearch(BinSearchDict<u64, u64>),
    BTree(BTreeDict<u64, u64>),
    Hash(HashDict<u64, u64>),
    HMP01(HagerupMP01Dict),
    HMP01u(HMP01UnreducedDict),
    R09BxHMP01(R09BxHMP01Dict),
    XorxHMP01(XorReducedDict),
}
impl GenericDict {
    fn new_typed(data: &[(u64, u64)], tp: &str) -> Self {
        match tp {
            "binsearch" => Self::Binsearch(BinSearchDict::new(data)),
            "btree" => Self::BTree(BTreeDict::new(data)),
            "hash" => Self::Hash(HashDict::new(data)),
            "hmp01" => Self::HMP01(HagerupMP01Dict::new(data)),
            "hmp01u" => Self::HMP01u(HMP01UnreducedDict::new(data)),
            "r09b+hmp01" => Self::R09BxHMP01(R09BxHMP01Dict::new(data)),
            "xor+hmp01" => Self::XorxHMP01(XorReducedDict::new(data)),
            _ => panic!("Unknown dict type: {}", tp),
        }
    }
}
impl Dict<u64, u64> for GenericDict {
    fn new(_data: &[(u64, u64)]) -> Self {
        unimplemented!();
    }
    // note: Box<dyn > may be more useful as optimizers might lift unnecessary calculations
    #[inline(never)]
    fn query(&self, key: u64) -> Option<u64> {
        match self {
            Self::Binsearch(ref v) => v.query(key),
            Self::BTree(ref v) => v.query(key),
            Self::Hash(ref v) => v.query(key),
            Self::HMP01(ref v) => v.query(key),
            Self::HMP01u(ref v) => v.query(key),
            Self::R09BxHMP01(ref v) => v.query(key),
            Self::XorxHMP01(ref v) => v.query(key),
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dict_type: &str = args.get(1).map(|x| x.as_str()).unwrap_or("hmp01");
    let sz = args
        .get(2)
        .map(|x| u64::from_str_radix(&x, 10).unwrap())
        .unwrap_or(1 << 18);

    let pattern = args.get(3).map(|x| x.as_str()).unwrap_or("rand32");
    let data = match pattern {
        "rand32" => util::make_random_chain_32(sz, 0x1, 0x1111),
        "rand64" => util::make_random_chain_64(sz, 0x1, 0x1111),
        _ => unimplemented!(),
    };
    let queries = std::hint::black_box(make_random_order_32(sz, 1, 1111));
    let t0 = Instant::now();

    let dict = std::hint::black_box(GenericDict::new_typed(&data, dict_type));
    let t1 = Instant::now();
    do_virt_queries(&dict, &queries);
    let t2 = Instant::now();
    println!(
        "type: {} size: {}; construction time: {} secs; query time: {} secs",
        dict_type,
        sz,
        t1.duration_since(t0).as_secs_f64(),
        t2.duration_since(t1).as_secs_f64()
    )
}
