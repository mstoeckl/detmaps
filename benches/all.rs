/* SPDX-License-Identifier: MPL-2.0 */
/* Benchmarking code / mini-framework for _possibly_ long running tasks with setup and processing phases.
 *
 * (The main alternatives, `criterion` and `divan`, have no good way to handle time limits (or crashes due to
 * hitting memory limit). This approach runs a subprocess per benchmark size class.)
 */

use clap::{Arg, ArgAction};
use detmaps::*;
use rustix::{io, pipe, process};
use std::{
    collections::BTreeSet,
    io::Write,
    os::fd::{AsRawFd, FromRawFd, OwnedFd},
};

struct BenchConfig<'a> {
    name: &'a str,
    dict_name: &'a str,
    min_time_sec: f64,
    soft_max_time_sec: f64,
    output_pipe: OwnedFd,
    u_bits: u32,
    sz: usize,
}

// todo: replace with plain 'fn'
type BenchFn = Box<dyn Fn(&BenchConfig)>;

struct DictionaryCycleResults {
    average_time_per_setup: f64,
    average_time_per_chainq: f64,
    average_time_per_parq: f64,
}

struct ParameterInfo {
    samples: u64,
    min: f64,
    q1: f64,
    median: f64,
    q3: f64,
    max: f64,
    systematic_error: f64,
}

impl ParameterInfo {
    fn from_iter<I: Iterator<Item = f64>>(x: I, systematic_error: f64) -> ParameterInfo {
        let mut v: Vec<f64> = x.collect();
        v.sort_by(|x, y| f64::partial_cmp(x, y).unwrap());
        let median = (v[v.len() / 2] + v[v.len() - 1 - v.len() / 2]) / 2.0;
        ParameterInfo {
            samples: v.len() as u64,
            min: *v.first().unwrap(),
            q1: v[v.len() / 4], // note: quartiles use the next most extreme value and are not interpolated like the median
            median,
            q3: v[v.len() - 1 - v.len() / 4],
            max: *v.last().unwrap(),
            systematic_error,
        }
    }
    fn to_string(&self) -> String {
        format!("{{ \"samples\": {}, \"min\": {}, \"q1\": {}, \"median\": {}, \"q3\": {}, \"max\": {}, \"systematic_error\": {} }}", self.samples, self.min, self.q1, self.median, self.q3, self.max, self.systematic_error)
    }
}

fn run_dictionary_cycle<H: Dict<u64, u64>>(
    iterations: u64,
    u_bits: u32,
    sz: usize,
    seed: u64,
) -> DictionaryCycleResults {
    let data: Vec<(u64, u64)> = std::hint::black_box(if u_bits == 32 {
        util::make_random_chain_32(sz as u64, seed, seed + 1)
    } else if u_bits == 64 {
        util::make_random_chain_64(sz as u64, seed, seed + 1)
    } else {
        panic!();
    });

    let start_time = std::time::Instant::now();
    let mut dict = None;
    for _ in 0..iterations {
        dict = Some(std::hint::black_box(H::new(&data, u_bits)))
    }
    let setup_time = std::time::Instant::now();
    /* Note: for sparse dictionaries like Raman95, the query performance critically
     * depends on whether the value are already loaded in the cache. It is possible
     * that during dictionary construction large temporaries are dropped or memory
     * for unqueried cells is touched, making the cache drop all rows with key data,
     * or not. To ensure all measured code has approximately the same set of initial
     * cached data, load values now. Access times may be significantly worse if
     * caches are emptied; however, that case is hard to measure. */
    let dict = dict.unwrap();
    util::parallel_query(&dict, &data);
    let warmup_time = std::time::Instant::now();
    for _ in 0..iterations {
        util::chain_query(&dict, &data);
    }
    let chain_query_time = std::time::Instant::now();
    // TODO: can simply follow the chain for 'n*iterations' steps;
    // and check the final end point is the starting point; no need for outer/inner loops
    for _ in 0..iterations {
        util::parallel_query(&dict, &data);
    }
    let parallel_query_time = std::time::Instant::now();

    DictionaryCycleResults {
        average_time_per_setup: setup_time.duration_since(start_time).as_secs_f64()
            / (iterations as f64),
        average_time_per_chainq: chain_query_time.duration_since(warmup_time).as_secs_f64()
            / (iterations as f64),
        average_time_per_parq: parallel_query_time
            .duration_since(chain_query_time)
            .as_secs_f64()
            / (iterations as f64),
    }
}

fn run_bench<H: Dict<u64, u64>>(cfg: &BenchConfig) {
    println!("Running benchmark: {}", cfg.name);

    /* The very first cycle may take significantly longer than usual due to the
     * cost of loading all the code and bringing it into caches. Run the instance
     * on a smaller input to correct for this, to get more accurate first-cycle
     * measurements. (This _may_ miss code which only runs at smaller or
     * larger instance sizes, or load code which is not used */
    run_dictionary_cycle::<H>(1, cfg.u_bits, 64, 0x1);

    let mut seed = 0x1234;
    let seed_start = std::time::Instant::now();
    let seed_results = run_dictionary_cycle::<H>(1, cfg.u_bits, cfg.sz, seed);
    let seed_end = std::time::Instant::now();
    let seed_elapsed = seed_end.duration_since(seed_start).as_secs_f64();

    /* This extrapolation can still strongly underestimate the required time
     * on small input sizes, but is unlikely to _overestimate_ the time. */
    let suggested_iteration_count = (cfg.min_time_sec / (10.0 * seed_elapsed)).ceil() as u64;
    assert!(suggested_iteration_count >= 1);
    /* Assuming 1us time measurement overhead.
     * This systematic error should be an _upper bound_ on the systematic
     * deviation of the measured time from the time we wanted to measure,
     * but it ignores the cost of the iteration loop itself and however
     * the compiler and CPU together optimize or pessimize std::black_box.
     */
    let systematic_error = 1e-6 / (suggested_iteration_count as f64);

    let mut measurements = Vec::new();
    if suggested_iteration_count == 1 {
        /* Test is slow, run it at most 3 times or until soft time limit */
        measurements.push(seed_results);

        let remaining_iters = ((cfg.soft_max_time_sec / seed_elapsed).ceil() as u64).min(3) - 1;
        seed += 2;
        for _ in 0..remaining_iters {
            measurements.push(run_dictionary_cycle::<H>(1, cfg.u_bits, cfg.sz, seed));
        }
    } else {
        /* Repeat test 10 times */
        for _ in 0..10 {
            measurements.push(run_dictionary_cycle::<H>(
                suggested_iteration_count,
                cfg.u_bits,
                cfg.sz,
                seed,
            ));
            seed += 2;
        }
    }

    /* Compute basic statistics.
     *
     * Note: lag spikes may correlate between the different parts measured in these runs.
     */
    let info_setup = ParameterInfo::from_iter(
        measurements.iter().map(|x| x.average_time_per_setup),
        systematic_error,
    );
    let info_chainq = ParameterInfo::from_iter(
        measurements.iter().map(|x| x.average_time_per_chainq),
        systematic_error,
    );
    let info_parq = ParameterInfo::from_iter(
        measurements.iter().map(|x| x.average_time_per_parq),
        systematic_error,
    );

    let mut message = String::from("{");
    message += &format!(
        "\"dictionary\": \"{}\", \"u_bits\": {}, \"n_elements\": {}, ",
        cfg.dict_name, cfg.u_bits, cfg.sz
    );

    message += "\"measurements\": {\n";

    message += "  \"setup\": ";
    message += &info_setup.to_string();
    message += ",\n";

    message += "  \"chainq\": ";
    message += &info_chainq.to_string();
    message += ",\n";

    message += "  \"parq\": ";
    message += &info_parq.to_string();

    message.push_str("\n} }\n");

    let b = message.as_bytes();
    let length_field = u32::to_le_bytes(message.len() as u32);
    assert!(io::write(&cfg.output_pipe, &length_field) == Ok(4));
    assert!(io::write(&cfg.output_pipe, b) == Ok(b.len()));
}

fn make_bench_fn<H: Dict<u64, u64>>() -> Box<dyn Fn(&BenchConfig)> {
    Box::new(|cfg| {
        run_bench::<H>(cfg);
    })
}

fn main() {
    let cmd = clap::Command::new("bench-all")
        .about("Set of benchmarks for deterministic dictionary testing")
        .arg(
            Arg::new("bench")
                .long("bench")
                .action(ArgAction::SetTrue)
                .default_value("false"),
        )
        .arg(
            Arg::new("exact")
                .long("exact")
                .action(ArgAction::SetTrue)
                .default_value("false"),
        )
        .arg(
            Arg::new("list")
                .long("list")
                .action(ArgAction::SetTrue)
                .default_value("false"),
        )
        .arg(
            Arg::new("new")
                .long("new")
                .action(ArgAction::SetTrue)
                .default_value("false")
                .help("Only run benchmarks for which no output has been recorded"),
        )
        .arg(Arg::new("run").long("run").action(ArgAction::Set).help(
            "Internal use: run the listed test and report outputs. Argument is output pipe fd.",
        ))
        .arg(Arg::new("filters").action(ArgAction::Append));

    let matches = cmd.get_matches();

    let memory_limit = 10000000000; /* bytes */
    let time_hard_max = 100; /* seconds */
    let time_soft_max = 3.0; /* seconds */
    let time_min = 0.1; /* seconds */

    // benchmarks: one per dictionary type, has 3 outputs; setup/parq/chainq
    let benchmarks: &[(&'static str, BenchFn)] = &[
        ("binsearch", make_bench_fn::<BinSearchDict<u64, u64>>()),
        ("btree", make_bench_fn::<BTreeDict<u64, u64>>()),
        ("hash", make_bench_fn::<HashDict<u64, u64>>()),
        ("hmp01", make_bench_fn::<HagerupMP01Dict>()),
        ("iter+hmp01", make_bench_fn::<HMP01UnreducedDict>()),
        ("r09p+hmp01", make_bench_fn::<R09BpxHMP01Dict>()),
        ("r09f+hmp01", make_bench_fn::<R09BfxHMP01Dict>()),
        ("xor+hmp01", make_bench_fn::<XorReducedDict>()),
        ("oms+hmp01", make_bench_fn::<OMSxHMP01Dict>()),
        ("oms+fks", make_bench_fn::<OMSxFKSDict>()),
        ("oms+disp", make_bench_fn::<OMSxDispDict>()),
        ("r09a", make_bench_fn::<Ruzic09Dict>()),
        ("raman95", make_bench_fn::<Raman95Dict>()),
    ];

    if let Some(fdstr) = matches.get_one::<String>("run") {
        let fd_no = i32::from_str_radix(&fdstr, 10).unwrap();
        let output_pipe = unsafe { OwnedFd::from_raw_fd(fd_no) };

        let mut names: Vec<&String> = matches.get_many::<String>("filters").unwrap().collect();
        let full_name = names.pop().unwrap();
        assert!(names.is_empty());

        let parts: Vec<&str> = full_name.split("_").collect();
        assert!(parts.len() == 3);
        let bench_name = parts[0];
        let u_bits = u32::from_str_radix(&parts[1], 10).unwrap();
        let sz_bits = u32::from_str_radix(&parts[2], 10).unwrap();
        let sz = 1 << sz_bits;

        let mem_limit = process::getrlimit(process::Resource::As);
        let time_limit = process::getrlimit(process::Resource::Cpu);

        let new_memory_limit = mem_limit.current.unwrap_or(memory_limit).min(memory_limit);
        process::setrlimit(
            process::Resource::As,
            process::Rlimit {
                current: Some(new_memory_limit),
                maximum: Some(new_memory_limit),
            },
        )
        .unwrap();

        /* Note: this imposes a CPU time limit inside the process, but the
         * timeout granularity is limited, and (in general) time limits should
         * in terms of wall clock time and be enforced by the parent process,
         * to ensure the benchmark subprocess cannot hang while ignoring signals.
         *
         * Also: SIGXCPU dumps core by default; a parent SIGKILL would avoid this.
         */
        let new_time_limit = time_limit
            .current
            .unwrap_or(time_hard_max)
            .min(time_hard_max);
        process::setrlimit(
            process::Resource::Cpu,
            process::Rlimit {
                current: Some(new_time_limit),
                maximum: Some(new_time_limit + 1),
            },
        )
        .unwrap();

        let cfg = BenchConfig {
            name: &full_name,
            dict_name: bench_name,
            min_time_sec: time_min,
            soft_max_time_sec: time_soft_max,
            output_pipe,
            u_bits,
            sz,
        };

        let Some(b) = benchmarks.iter().find(|x| x.0 == bench_name) else {
            panic!("Unknown benchmark {}", bench_name);
        };
        b.1(&cfg);
        return;
    }

    let size_classes: Vec<u32> = (0..=33).collect();
    let is_list = matches.get_flag("list");
    if is_list {
        for c in size_classes.iter() {
            for u_bits in [32u32, 64u32] {
                for entry in benchmarks {
                    println!("{}_{}_{}", entry.0, u_bits, c);
                }
            }
        }
        return;
    }

    let is_bench = matches.get_flag("bench");
    if !is_bench {
        eprintln!("Running in test mode (without --bench) not yet supported");
        return;
    }

    let new_only = matches.get_flag("new");

    let exact = matches.get_flag("exact");
    let filters: Vec<&String> = matches
        .get_many::<String>("filters")
        .unwrap_or_default()
        .collect();

    let mut chosen_benches: Vec<(&'static str, u32, u32)> = Vec::new();
    for c in size_classes.iter() {
        for u_bits in [32u32, 64u32] {
            if *c > u_bits {
                continue;
            }

            for entry in benchmarks {
                let name = format!("{}_{}_{}", entry.0, u_bits, c);
                let keep = if filters.is_empty() {
                    true
                } else {
                    if exact {
                        /* exact match with a filter */
                        filters.iter().any(|x| x.as_str() == name)
                    } else {
                        /* Substring match with a filter */
                        filters.iter().any(|x| name.contains(*x))
                    }
                };
                if keep {
                    chosen_benches.push((entry.0, u_bits, *c));
                }
            }
        }
    }

    let executable = std::env::current_exe().unwrap();

    let mut failed_benches = BTreeSet::new();

    let bench_time: String = chrono::Local::now().format("%Y-%m-%d-%H:%M:%S").to_string();
    let folder = "target/bench/";
    let archive_folder = String::from(folder) + &bench_time + "/";
    let output_folder = String::from(folder) + "main/";
    std::fs::create_dir_all(&archive_folder).unwrap();
    std::fs::create_dir_all(&output_folder).unwrap();

    for (name, u_bits, sz_class) in chosen_benches {
        if failed_benches.contains(&(name, u_bits)) {
            continue;
        }

        let full_name = format!("{}_{}_{}", name, u_bits, sz_class);
        let file_name = full_name.clone() + ".json";
        let current_path = std::path::PathBuf::from(output_folder.clone()).join(&file_name);
        if new_only && std::fs::exists(&current_path).unwrap() {
            continue;
        }

        let (pipe_r, pipe_w) = pipe::pipe_with(pipe::PipeFlags::CLOEXEC).unwrap();
        io::fcntl_setfd(&pipe_w, io::FdFlags::empty()).unwrap();
        let shared_fd = format!("{}", pipe_w.as_raw_fd());

        let bench_start_time = std::time::Instant::now();

        let mut handle = std::process::Command::new(&executable)
            .args(&["--run", &shared_fd, &full_name])
            .spawn()
            .unwrap();
        drop(pipe_w);

        /* todo: move time enforcement to this process */

        let mut output: Vec<u8> = Vec::new();
        let mut buf = [0; 4096];
        let mut clean_exit;
        loop {
            match io::read(&pipe_r, &mut buf) {
                Err(err) => match err {
                    io::Errno::AGAIN | io::Errno::INTR => {
                        continue;
                    }
                    io::Errno::CONNRESET | io::Errno::NOTCONN => {
                        clean_exit = true;
                        break;
                    }
                    _ => {
                        clean_exit = false;
                        break;
                    }
                },
                Ok(b) => {
                    if b == 0 {
                        clean_exit = true;
                        break;
                    }
                    output.extend_from_slice(&buf[..b]);
                }
            }
        }
        handle.wait().unwrap();
        drop(pipe_r);

        if let Some(l) = output.first_chunk::<4>() {
            let length = u32::from_le_bytes(*l) as usize;
            if output.len() != length + 4 {
                clean_exit = false;
            }
        } else {
            clean_exit = false;
        }

        let bench_end_time = std::time::Instant::now();

        if !clean_exit {
            println!(
                "Benchmark '{}' crashed or timed out after {} secs, skipping all others in the series",
                full_name,
                bench_end_time.duration_since(bench_start_time).as_secs_f64(),
            );
            failed_benches.insert((name, u_bits));
            // TODO: formalize and store timeout/crash results as JSON,
            // ideally distinguishing which failure mode occurred
            continue;
        }
        let validated_output = &output[4..];

        println!(
            "Benchmark '{}' completed after {} secs",
            full_name,
            bench_end_time
                .duration_since(bench_start_time)
                .as_secs_f64()
        );

        let archive_path = std::path::PathBuf::from(archive_folder.clone()).join(&file_name);
        let rel_archive_path = std::path::PathBuf::from("../")
            .join(&bench_time)
            .join(&file_name);

        let mut f = std::fs::File::create(&archive_path).unwrap();
        f.write(validated_output).unwrap();
        drop(f);

        let _ = std::fs::remove_file(&current_path);
        std::os::unix::fs::symlink(&rel_archive_path, &current_path).unwrap();
    }

    println!("Done.");
}
