//! Single-node benchmark proving the encode/decode pool arbiter fixes the #8246 starvation.
//!
//! The Rayon pool is pinned to a small fixed size so the decode flood genuinely saturates it and
//! starves SURB encode — reproducing the production contention that a well-resourced test box hides.
//! We measure how long a batch of encode tasks takes while decode floods, arbiter on vs off; the
//! reverse (decode under encode flood, must not be disadvantaged); the exit's mixed encode+decode;
//! and pure decode (no encode) to confirm zero overhead for forwarding.
//!
//! Run: `cargo bench --features parallelize-rayon --bench pool_arbiter_bench`

#[path = "bench_common/mod.rs"]
mod common;

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use common::{DECODE_US, ENCODE_US_3HOP_2SURB as ENCODE_US, spin};
use criterion::{Criterion, criterion_group, criterion_main};
use hopr_utilities::parallelize::cpu;

const ENCODE_BATCH: usize = 200;
const FLOOD_SUBMITTERS: usize = 16;
const PINNED_POOL: usize = 2; // constrain resources so decode-flood actually saturates the pool

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap()
}

fn flood_then_measure(c: &mut Criterion, group: &str, flood_encode: bool) {
    let rt = runtime();
    let _ = cpu::init_thread_pool(PINNED_POOL);
    let mut g = c.benchmark_group(group);
    g.sample_size(20);
    for enabled in [true, false] {
        cpu::with_arbitration(common::arbitration(enabled));
        g.bench_function(if enabled { "arbiter_on" } else { "arbiter_off" }, |b| {
            b.to_async(&rt).iter_custom(|iters| async move {
                let stop = Arc::new(AtomicBool::new(false));
                let floods: Vec<_> = (0..FLOOD_SUBMITTERS)
                    .map(|_| {
                        let stop = stop.clone();
                        tokio::spawn(async move {
                            while !stop.load(Ordering::Relaxed) {
                                if flood_encode {
                                    let _ = cpu::spawn_encode_blocking(|| spin(ENCODE_US), "b_enc").await;
                                } else {
                                    let _ = cpu::spawn_decode_blocking(|| spin(DECODE_US), "b_dec").await;
                                }
                            }
                        })
                    })
                    .collect();
                tokio::time::sleep(Duration::from_millis(50)).await;
                let start = Instant::now();
                for _ in 0..iters {
                    for _ in 0..ENCODE_BATCH {
                        if flood_encode {
                            let _ = cpu::spawn_decode_blocking(|| spin(DECODE_US), "b_dec").await;
                        } else {
                            let _ = cpu::spawn_encode_blocking(|| spin(ENCODE_US), "b_enc").await;
                        }
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
    g.finish();
}

/// The core proof: SURB encode latency while decode floods — arbiter_on must be far lower.
fn encode_under_decode_flood(c: &mut Criterion) {
    flood_then_measure(c, "encode_under_decode_flood", false);
}
/// Reverse: decode while encode floods — must not be disadvantaged.
fn decode_under_encode_flood(c: &mut Criterion) {
    flood_then_measure(c, "decode_under_encode_flood", true);
}

/// Pure decode (no encode present): the arbiter must not engage — zero overhead for forwarding.
fn decode_only_no_encode(c: &mut Criterion) {
    let rt = runtime();
    let _ = cpu::init_thread_pool(PINNED_POOL);
    let mut g = c.benchmark_group("decode_only_no_encode");
    g.sample_size(20);
    for enabled in [true, false] {
        cpu::with_arbitration(common::arbitration(enabled));
        g.bench_function(if enabled { "arbiter_on" } else { "arbiter_off" }, |b| {
            b.to_async(&rt).iter(|| async {
                let tasks: Vec<_> = (0..64)
                    .map(|_| cpu::spawn_decode_blocking(|| spin(DECODE_US), "b_dec"))
                    .collect();
                for t in tasks {
                    let _ = t.await;
                }
            });
        });
    }
    g.finish();
}

/// Exit: encode and decode both heavy at once — neither should collapse.
fn exit_mixed_encode_decode(c: &mut Criterion) {
    let rt = runtime();
    let _ = cpu::init_thread_pool(PINNED_POOL);
    let mut g = c.benchmark_group("exit_mixed_encode_decode");
    g.sample_size(20);
    for enabled in [true, false] {
        cpu::with_arbitration(common::arbitration(enabled));
        g.bench_function(if enabled { "arbiter_on" } else { "arbiter_off" }, |b| {
            b.to_async(&rt).iter(|| async {
                let enc: Vec<_> = (0..64)
                    .map(|_| cpu::spawn_encode_blocking(|| spin(ENCODE_US), "b_enc"))
                    .collect();
                let dec: Vec<_> = (0..64)
                    .map(|_| cpu::spawn_decode_blocking(|| spin(DECODE_US), "b_dec"))
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
    g.finish();
}

criterion_group!(
    benches,
    encode_under_decode_flood,
    decode_under_encode_flood,
    exit_mixed_encode_decode,
    decode_only_no_encode,
);
criterion_main!(benches);
