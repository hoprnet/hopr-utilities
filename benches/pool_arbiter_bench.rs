//! Single-node benchmark for the encode/decode pool arbiter.
//!
//! Reproduces the #8246 contention on ONE shared Rayon pool (unlike the hoprnet cluster tests, which
//! share one process-global pool across every node and so cannot isolate a single role): a sustained
//! DECODE flood (relay forwarding / exit peel) competes with latency-critical ENCODE (SURB
//! generation). We measure how long a fixed batch of encode tasks takes to complete while decode
//! floods the pool, with the arbiter on vs off. A second group measures pure-decode throughput to
//! confirm the arbiter adds ~no overhead when there is no encode to protect.
//!
//! Run: `cargo bench --features parallelize-rayon --bench pool_arbiter_bench`

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use criterion::{Criterion, criterion_group, criterion_main};
use hopr_utilities::parallelize::cpu;

/// Simulate a CPU-bound SPHINX operation by spinning for `us` microseconds.
fn spin(us: u64) {
    let end = Instant::now() + Duration::from_micros(us);
    while Instant::now() < end {
        std::hint::spin_loop();
    }
}

const DECODE_US: u64 = 300; // SPHINX peel (relay forward / exit terminate)
const ENCODE_US: u64 = 120; // SPHINX wrap + SURB generation
const ENCODE_BATCH: usize = 200; // encode tasks timed per iteration
const FLOOD_SUBMITTERS: usize = 16; // concurrent decode submitters

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap()
}

fn prod_pool_size() -> usize {
    std::thread::available_parallelism()
        .map(|n| (n.get() / 2).max(1))
        .unwrap_or(2)
}

/// Key metric: latency of a batch of encode tasks while decode floods the pool.
fn encode_under_decode_flood(c: &mut Criterion) {
    let rt = runtime();
    let _ = cpu::init_thread_pool(prod_pool_size());

    let mut group = c.benchmark_group("encode_under_decode_flood");
    group.sample_size(20);
    for enabled in [true, false] {
        cpu::configure_arbitration(enabled, 75, 50);
        group.bench_function(if enabled { "arbiter_on" } else { "arbiter_off" }, |b| {
            b.to_async(&rt).iter_custom(|iters| async move {
                let stop = Arc::new(AtomicBool::new(false));
                let floods: Vec<_> = (0..FLOOD_SUBMITTERS)
                    .map(|_| {
                        let stop = stop.clone();
                        tokio::spawn(async move {
                            while !stop.load(Ordering::Relaxed) {
                                let _ = cpu::spawn_decode_blocking(|| spin(DECODE_US), "bench_decode").await;
                            }
                        })
                    })
                    .collect();

                // Let the flood saturate the pool before timing.
                tokio::time::sleep(Duration::from_millis(50)).await;

                let start = Instant::now();
                for _ in 0..iters {
                    for _ in 0..ENCODE_BATCH {
                        let _ = cpu::spawn_encode_blocking(|| spin(ENCODE_US), "bench_encode").await;
                    }
                }
                let elapsed = start.elapsed();

                stop.store(true, Ordering::Relaxed);
                for f in floods {
                    let _ = f.await;
                }
                elapsed
            });
        });
    }
    group.finish();
}

/// Overhead check: pure decode throughput with no encode present — the arbiter must not engage
/// (encode == 0), so on and off should be within noise.
fn decode_throughput_no_encode(c: &mut Criterion) {
    let rt = runtime();
    let _ = cpu::init_thread_pool(prod_pool_size());

    let mut group = c.benchmark_group("decode_throughput_no_encode");
    group.sample_size(20);
    for enabled in [true, false] {
        cpu::configure_arbitration(enabled, 75, 50);
        group.bench_function(if enabled { "arbiter_on" } else { "arbiter_off" }, |b| {
            b.to_async(&rt).iter(|| async {
                let tasks: Vec<_> = (0..64)
                    .map(|_| cpu::spawn_decode_blocking(|| spin(DECODE_US), "bench_decode"))
                    .collect();
                for t in tasks {
                    let _ = t.await;
                }
            });
        });
    }
    group.finish();
}

/// Reverse check (north-star): a batch of DECODE tasks while ENCODE floods the pool. Decode must
/// keep at least its guaranteed share (`100 - encode_reserve_pct` = 50%) and not be starved — the
/// arbiter caps decode only *down to* that floor, and never throttles encode.
fn decode_under_encode_flood(c: &mut Criterion) {
    let rt = runtime();
    let _ = cpu::init_thread_pool(prod_pool_size());

    let mut group = c.benchmark_group("decode_under_encode_flood");
    group.sample_size(20);
    for enabled in [true, false] {
        cpu::configure_arbitration(enabled, 75, 50);
        group.bench_function(if enabled { "arbiter_on" } else { "arbiter_off" }, |b| {
            b.to_async(&rt).iter_custom(|iters| async move {
                let stop = Arc::new(AtomicBool::new(false));
                let floods: Vec<_> = (0..FLOOD_SUBMITTERS)
                    .map(|_| {
                        let stop = stop.clone();
                        tokio::spawn(async move {
                            while !stop.load(Ordering::Relaxed) {
                                let _ = cpu::spawn_encode_blocking(|| spin(ENCODE_US), "bench_encode").await;
                            }
                        })
                    })
                    .collect();
                tokio::time::sleep(Duration::from_millis(50)).await;

                let start = Instant::now();
                for _ in 0..iters {
                    for _ in 0..ENCODE_BATCH {
                        let _ = cpu::spawn_decode_blocking(|| spin(DECODE_US), "bench_decode").await;
                    }
                }
                let elapsed = start.elapsed();

                stop.store(true, Ordering::Relaxed);
                for f in floods {
                    let _ = f.await;
                }
                elapsed
            });
        });
    }
    group.finish();
}

/// Exit under download: encode (download data) and decode (incoming acks/SURBs) both heavy at once.
/// Measures aggregate completion of an interleaved encode+decode batch — neither side should
/// collapse; the arbiter should keep the pool fully utilised near a 50/50 split.
fn exit_mixed_encode_decode(c: &mut Criterion) {
    let rt = runtime();
    let _ = cpu::init_thread_pool(prod_pool_size());

    let mut group = c.benchmark_group("exit_mixed_encode_decode");
    group.sample_size(20);
    for enabled in [true, false] {
        cpu::configure_arbitration(enabled, 75, 50);
        group.bench_function(if enabled { "arbiter_on" } else { "arbiter_off" }, |b| {
            b.to_async(&rt).iter(|| async {
                // Fire an interleaved batch of encode and decode concurrently, await all.
                let enc: Vec<_> = (0..64)
                    .map(|_| cpu::spawn_encode_blocking(|| spin(ENCODE_US), "bench_encode"))
                    .collect();
                let dec: Vec<_> = (0..64)
                    .map(|_| cpu::spawn_decode_blocking(|| spin(DECODE_US), "bench_decode"))
                    .collect();
                for t in enc {
                    let _ = t.await;
                }
                for t in dec {
                    let _ = t.await;
                }
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    encode_under_decode_flood,
    decode_under_encode_flood,
    exit_mixed_encode_decode,
    decode_throughput_no_encode,
);
criterion_main!(benches);
