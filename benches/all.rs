/* SPDX-License-Identifier: MPL-2.0 */
use criterion::{
    black_box, criterion_group, criterion_main, measurement::WallTime, AxisScale, BatchSize,
    BenchmarkGroup, BenchmarkId, Criterion, PlotConfiguration, Throughput,
};
use detmaps::*;
use std::time::Duration;

/** Generate a sorted sequence of the first n integers */
fn make_sequential(n: u64) -> Vec<(u64, u64)> {
    (0..n).map(|x| (x, x)).collect()
}
fn ith_random_sampled_32(i: u64, seed: u64) -> u64 {
    /* Now 2^32 and p are coprime, so (x*p)%n is a permutation. */
    let p = (seed.wrapping_mul(0x9e3779b97f4a7c16)) | 0x1;
    ((i as u128 * p as u128) % (1u128 << 32)) as u64
}

/** Generate n pseudo-random samples in 0..=u32::MAX _without_ repetition. */
fn make_random_sampled_32(n: u64, seed: u64) -> Vec<(u64, u64)> {
    (0..n)
        .map(|x| (ith_random_sampled_32(x, seed), x))
        .collect()
}

/** Generate n pseudo-random samples from the first n entries of `make_random_sampled_32`, _without_ repetition.
 * TODO: make this an iterator, to reduce space overhead */
fn make_random_order_32(n: u64, seed_vals: u64, seed_order: u64) -> Vec<u64> {
    assert!(n.is_power_of_two());
    /* Now n and p are coprime, so (x*p)%n is a permutation. */
    (0..n)
        .map(|x| {
            let y = ith_random_sampled_32(x, seed_order) % n;
            ith_random_sampled_32(y, seed_vals)
        })
        .collect()
}

/** Setup load for sequ */
fn time_setup<T: Dict<u64, u64>>(data: &[(u64, u64)]) -> u64 {
    std::hint::black_box(T::new(&data));
    0
}
fn get_setup<T: Dict<u64, u64>>(data: &[(u64, u64)]) -> T {
    T::new(&data)
}
/** Query load: every element in 0..n. Access pattern is very friendly to some dictionaries, but not others.
 * TODO: is there a standard 'high discrepancy' permutation, or do e.g. `n` random samples in [0..n] with XorShift suffice?
 */
fn sequential_time_query<T: Dict<u64, u64>>(d: &T, n: u64) -> u64 {
    let mut x = 0;
    for i in 0..n {
        x += d.query(i).unwrap_or(0);
    }
    x
}

fn ordered_time_query<T: Dict<u64, u64>>(d: &T, queries: &[u64]) -> u64 {
    let mut x = 0;
    for q in queries {
        x += d.query(*q).unwrap_or(0);
    }
    x
}

fn bench_rand32<T: Dict<u64, u64>>(bits: u32, group: &mut BenchmarkGroup<WallTime>, name: &str) {
    let sz: u64 = 1 << bits;
    let seed_vals = 1;
    let seed_order = 1111;

    group.bench_with_input(
        BenchmarkId::new(format!("rand32_{}_setup", name), bits),
        &sz,
        |b, &sz| {
            b.iter_batched(
                || make_random_sampled_32(sz, seed_vals),
                |x| time_setup::<T>(&x),
                BatchSize::PerIteration,
            );
        },
    );
    group.bench_with_input(
        BenchmarkId::new(format!("rand32_{}_query", name), bits),
        &sz,
        |b, &sz| {
            // TODO: setup is always the same for deterministic case, and expensive; can deduplicate?
            b.iter_batched(
                || {
                    (
                        get_setup::<T>(&make_random_sampled_32(sz, seed_vals)),
                        make_random_order_32(sz, seed_vals, seed_order),
                    )
                },
                |x| ordered_time_query(&x.0, &x.1),
                BatchSize::PerIteration,
            );
        },
    );
}

fn criterion_benchmark(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group("pseudorand setup and query");
    group.plot_config(plot_config);
    group.warm_up_time(Duration::from_millis(20));
    group.measurement_time(Duration::from_millis(100));
    group.sample_size(10);
    /* Note: tiny sizes are affected by measurement overhead; sizes >20 take hours for hmp01... */
    for bits in 0..=32 {
        let sz: u64 = 1 << bits;
        group.throughput(Throughput::Bytes(std::mem::size_of::<u64>() as u64 * sz));

        // TODO: add 'hard_timeout' option either in criterion or with workaround
        if bits <= 25 {
            bench_rand32::<BinSearchDict<u64, u64>>(bits, &mut group, "binsearch");
        }
        if bits <= 28 {
            bench_rand32::<BTreeDict<u64, u64>>(bits, &mut group, "btree");
        }
        if bits <= 27 {
            bench_rand32::<HashDict<u64, u64>>(bits, &mut group, "hash");
        }
        if bits <= 20 {
            bench_rand32::<HagerupMP01Dict>(bits, &mut group, "hmp01");
        }
        if bits <= 22 {
            bench_rand32::<HMP01UnreducedDict>(bits, &mut group, "hmp01u");
        }
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
