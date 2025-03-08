/* SPDX-License-Identifier: MPL-2.0 */
use criterion::{
    criterion_group, criterion_main, measurement::WallTime, AxisScale, BatchSize, BenchmarkGroup,
    BenchmarkId, Criterion, PlotConfiguration, Throughput,
};
use detmaps::*;
use std::time::Duration;

/** Setup load for sequ */
fn time_setup<T: Dict<u64, u64>>(data: &[(u64, u64)]) -> u64 {
    std::hint::black_box(T::new(&data));
    0
}
fn get_setup<T: Dict<u64, u64>>(data: &[(u64, u64)]) -> T {
    T::new(&data)
}

fn bench_group<T: Dict<u64, u64>>(
    bits: u32,
    group: &mut BenchmarkGroup<WallTime>,
    name: &str,
    pattern: &str,
    make_random_chain: fn(u64, u64, u64) -> Vec<(u64, u64)>,
) {
    let sz: u64 = 1 << bits;
    let seed_vals = 1;
    let seed_order = 1111;

    group.bench_with_input(
        BenchmarkId::new(format!("{}_{}_setup", pattern, name), bits),
        &sz,
        |b, &sz| {
            b.iter_batched(
                || make_random_chain(sz, seed_vals, seed_order),
                |x| time_setup::<T>(&x),
                BatchSize::PerIteration,
            );
        },
    );
    // TODO: setup is always the same for deterministic case, and expensive; can deduplicate?

    // TODO: a) properly randomize initial seeds, for every run
    group.bench_with_input(
        BenchmarkId::new(format!("{}_{}_chainq", pattern, name), bits),
        &sz,
        |b, &sz| {
            b.iter_batched(
                || {
                    let chain = make_random_chain(sz, seed_vals, seed_order);
                    let d = get_setup::<T>(&chain);
                    (d, chain)
                },
                |x| util::chain_query(&x.0, &x.1),
                BatchSize::PerIteration,
            );
        },
    );
    group.bench_with_input(
        BenchmarkId::new(format!("{}_{}_parq", pattern, name), bits),
        &sz,
        |b, &sz| {
            b.iter_batched(
                || {
                    let chain = make_random_chain(sz, seed_vals, seed_order);
                    let d = get_setup::<T>(&chain);
                    (d, chain)
                },
                |x| util::parallel_query(&x.0, &x.1),
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
        let patopts: [(&str, fn(u64, u64, u64) -> Vec<(u64, u64)>); 2] = [
            ("rand32", util::make_random_chain_32),
            ("rand64", util::make_random_chain_64),
        ];
        for (pattern, patfn) in patopts.iter() {
            if bits <= 25 {
                bench_group::<BinSearchDict<u64, u64>>(
                    bits,
                    &mut group,
                    "binsearch",
                    pattern,
                    *patfn,
                );
            }
            if bits <= 28 {
                bench_group::<BTreeDict<u64, u64>>(bits, &mut group, "btree", pattern, *patfn);
            }
            if bits <= 27 {
                bench_group::<HashDict<u64, u64>>(bits, &mut group, "hash", pattern, *patfn);
            }
            if bits <= 20 && *pattern == "rand32" {
                bench_group::<HagerupMP01Dict>(bits, &mut group, "hmp01", pattern, *patfn);
            }
            if bits <= 25 {
                bench_group::<HMP01UnreducedDict>(bits, &mut group, "hmp01u", pattern, *patfn);
            }
            if bits <= 22 {
                bench_group::<R09BxHMP01Dict>(bits, &mut group, "r09bxhmp01", pattern, *patfn);
            }
            if bits <= 22 {
                bench_group::<XorReducedDict>(bits, &mut group, "xor+hmp01", pattern, *patfn);
            }
            if bits <= 22 {
                bench_group::<Ruzic09Dict>(bits, &mut group, "r09a", pattern, *patfn);
            }
            if bits <= 12 {
                /* This construction has _quadratic_ output size and is only suitable for small inputs */
                bench_group::<Raman96Dict>(bits, &mut group, "raman96", pattern, *patfn);
            }
        }
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
