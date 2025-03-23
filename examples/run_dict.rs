/* SPDX-License-Identifier: MPL-2.0 */
/*! Test program: build and query a dictionary */

use detmaps::*;
use std::time::Instant;

struct DictTrialImpl<H> {
    dict: H,
}
trait DictTrial {
    fn new(data: &[(u64, u64)], u_bits: u32) -> Box<dyn DictTrial>
    where
        Self: Sized;
    fn test_par(&self, data: &[u64], result: u64);
    // todo: add test_chain, with number of steps to run for
}
impl<H> DictTrial for DictTrialImpl<H>
where
    H: Dict<u64, u64> + 'static,
{
    fn new(data: &[(u64, u64)], u_bits: u32) -> Box<dyn DictTrial>
    where
        Self: Sized,
    {
        Box::new(DictTrialImpl {
            dict: H::new(data, u_bits),
        })
    }
    #[inline(never)]
    fn test_par(&self, queries: &[u64], result: u64) {
        let mut x = 0;
        for q in queries.iter() {
            x += self.dict.query(*q).unwrap();
        }
        assert!(std::hint::black_box(x) == result);
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
        "rand32" => util::make_random_chain(sz, 32, 0x1, 0x1111),
        "rand64" => util::make_random_chain(sz, 64, 0x1, 0x1111),
        _ => unimplemented!(),
    };
    let queries: Vec<u64> = std::hint::black_box(data.iter().map(|x| x.0).collect());
    let mut result: u64 = 0;
    for (_k, v) in data.iter() {
        result = result.wrapping_add(*v);
    }
    let t0 = Instant::now();

    let u_bits = u64::BITS;

    let dict = match dict_type {
        "binsearch" => DictTrialImpl::<BinSearchDict<u64, u64>>::new(&data, u_bits),
        "btree" => DictTrialImpl::<BTreeDict<u64, u64>>::new(&data, u_bits),
        "hash" => DictTrialImpl::<HashDict<u64, u64>>::new(&data, u_bits),
        "hmp01" => DictTrialImpl::<HagerupMP01Dict>::new(&data, u_bits),
        "iter+hmp01" => DictTrialImpl::<HMP01UnreducedDict>::new(&data, u_bits),
        "r09f+hmp01" => DictTrialImpl::<R09BfxHMP01Dict>::new(&data, u_bits),
        "r09p+hmp01" => DictTrialImpl::<R09BpxHMP01Dict>::new(&data, u_bits),
        "xor+hmp01" => DictTrialImpl::<XorReducedDict>::new(&data, u_bits),
        "oms+hmp01" => DictTrialImpl::<OMSxHMP01Dict>::new(&data, u_bits),
        "oms+fks" => DictTrialImpl::<OMSxFKSDict>::new(&data, u_bits),
        "raman95" => DictTrialImpl::<Raman95Dict>::new(&data, u_bits),
        "oma+slow" => DictTrialImpl::<OMADict>::new(&data, u_bits),
        "oma+fast" => DictTrialImpl::<FastOMADict>::new(&data, u_bits),
        "oma+disp" => DictTrialImpl::<OMAxDispDict>::new(&data, u_bits),
        "r09a" => DictTrialImpl::<Ruzic09Dict>::new(&data, u_bits),
        _ => panic!(),
    };

    let t1 = Instant::now();

    dict.test_par(&queries, result);

    let t2 = Instant::now();
    println!(
        "type: {} size: {}; construction time: {} secs; (par)query time: {} secs",
        dict_type,
        sz,
        t1.duration_since(t0).as_secs_f64(),
        t2.duration_since(t1).as_secs_f64()
    )
}
