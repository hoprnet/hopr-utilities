//! Sustained-throughput + latency matrix for the encode/decode pool arbiter (harness = false).
//!
//! Busy-weights are the **real** SPHINX op costs measured on this machine via `packet_bench`, and
//! throughput is reported as delivered MB/s at SESSION_MTU (1020 B/pkt):
//!   decode (forwarding/peel, any hop)                 = ~98 µs
//!   encode (full, 3-hop/2-SURB incl. SURB generation) = ~471 µs   (encode is the *expensive* op)
//! so a delivered download packet costs ~98 µs decode + ~471 µs encode. We sweep the demand ratio
//! across decode-dominant (download flood), balanced-count, and encode-dominant regimes on a small
//! pinned pool, and report per-class MB/s + P50/P99 latency, arbiter ON vs OFF. Goal: the arbiter
//! must protect encode under a decode-volume flood while never cutting delivery in the other regimes
//! (max throughput everywhere).
//!
//! Representative result on this machine (enc=471µs, dec=98µs, 1020 B/pkt, pool=2 threads):
//!   scenario             | enc MB/s ON→OFF | enc P99 ON→OFF | dec MB/s ON→OFF
//!   relay (decode-only)  |       —         |      —         | 19.71 → 19.78  (zero overhead)
//!   download flood 1:16  |  1.70 → 0.80    | 0.6 → 1.3 ms   | 11.47 → 15.85  (encode 2.1× protected)
//!   download heavy 1:8   |  1.71 → 1.20    | 0.6 → 0.9 ms   | 11.56 → 13.95
//!   balanced 4:4         |  3.41 → 3.42    | 1.2 → 1.2 ms   |  3.43 →  3.42  (neutral)
//!   encode-dominant 8:2  |  3.94 → 3.92    | 2.0 → 2.1 ms   |  0.98 →  1.01  (neutral)
//! The flood-gate keeps every non-flood regime neutral while protecting SURB encode under a genuine
//! decode-volume flood (the #8246 case), so download SURB production never starves.

#[path = "bench_common/mod.rs"]
mod common;

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use common::{DECODE_US, ENCODE_US_3HOP_2SURB as ENCODE_US, PAYLOAD_BYTES, spin};
use hopr_utilities::parallelize::cpu;

const PINNED_POOL: usize = 2;
const WINDOW: Duration = Duration::from_millis(2000);

#[derive(Default)]
struct Class {
    ops: u64,
    p50_ms: f64,
    p99_ms: f64,
}

struct Row {
    enc: Class,
    dec: Class,
}

fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        0.0
    } else {
        sorted[((sorted.len() as f64 * p) as usize).min(sorted.len() - 1)]
    }
}

async fn measure(enc_submitters: usize, dec_submitters: usize) -> Row {
    let stop = Arc::new(AtomicBool::new(false));
    let (enc_ops, dec_ops) = (Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)));
    let enc_lat = Arc::new(std::sync::Mutex::new(Vec::<f64>::new()));
    let dec_lat = Arc::new(std::sync::Mutex::new(Vec::<f64>::new()));

    let mut tasks = Vec::new();
    let spawn_loop =
        |is_enc: bool, ops: Arc<AtomicU64>, lat: Arc<std::sync::Mutex<Vec<f64>>>, stop: Arc<AtomicBool>| {
            tokio::spawn(async move {
                while !stop.load(Ordering::Relaxed) {
                    let t0 = Instant::now();
                    if is_enc {
                        let _ = cpu::spawn_encode_blocking(|| spin(ENCODE_US), "enc").await;
                    } else {
                        let _ = cpu::spawn_decode_blocking(|| spin(DECODE_US), "dec").await;
                    }
                    let ms = t0.elapsed().as_secs_f64() * 1000.0;
                    ops.fetch_add(1, Ordering::Relaxed);
                    if let Ok(mut v) = lat.lock() {
                        if v.len() < 50000 {
                            v.push(ms);
                        }
                    }
                }
            })
        };
    for _ in 0..enc_submitters {
        tasks.push(spawn_loop(true, enc_ops.clone(), enc_lat.clone(), stop.clone()));
    }
    for _ in 0..dec_submitters {
        tasks.push(spawn_loop(false, dec_ops.clone(), dec_lat.clone(), stop.clone()));
    }

    tokio::time::sleep(WINDOW).await;
    stop.store(true, Ordering::Relaxed);
    for t in tasks {
        let _ = t.await;
    }

    let finish = |ops: &AtomicU64, lat: &std::sync::Mutex<Vec<f64>>| {
        let mut v = lat.lock().unwrap().clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Class {
            ops: ops.load(Ordering::Relaxed),
            p50_ms: pct(&v, 0.50),
            p99_ms: pct(&v, 0.99),
        }
    };
    Row {
        enc: finish(&enc_ops, &enc_lat),
        dec: finish(&dec_ops, &dec_lat),
    }
}

fn main() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(6)
        .enable_all()
        .build()
        .unwrap();
    let _ = cpu::init_thread_pool(PINNED_POOL);

    // (label, encode submitters, decode submitters)
    let scenarios: &[(&str, usize, usize)] = &[
        ("relay: decode-only", 0, 8),
        ("download flood (enc:dec 1:16)", 1, 16),
        ("download heavy (1:8)", 1, 8),
        ("download moderate (2:8)", 2, 8),
        ("balanced count (4:4)", 4, 4),
        ("exit enc-heavy (6:4)", 6, 4),
        ("encode-dominant (8:2)", 8, 2),
    ];

    println!(
        "\n## Real-weight throughput/latency matrix (enc={ENCODE_US}µs, dec={DECODE_US}µs, {PAYLOAD_BYTES}B/pkt, \
         pool={PINNED_POOL}, {}ms)\n",
        WINDOW.as_millis()
    );
    println!("| scenario | arb | enc MB/s | dec MB/s | total MB/s | enc P50/P99 ms | dec P50/P99 ms |");
    println!("|---|---|--:|--:|--:|--:|--:|");
    let mb = |ops: u64, s: f64| ops as f64 * PAYLOAD_BYTES as f64 / (1024.0 * 1024.0) / s;
    for (label, enc, dec) in scenarios {
        for enabled in [true, false] {
            cpu::with_arbitration(common::arbitration(enabled));
            let r = rt.block_on(measure(*enc, *dec));
            let s = WINDOW.as_secs_f64();
            let (enc_mb, dec_mb) = (mb(r.enc.ops, s), mb(r.dec.ops, s));
            println!(
                "| {label} | {} | {:.2} | {:.2} | {:.2} | {:.1}/{:.1} | {:.1}/{:.1} |",
                if enabled { "ON " } else { "off" },
                enc_mb,
                dec_mb,
                enc_mb + dec_mb,
                r.enc.p50_ms,
                r.enc.p99_ms,
                r.dec.p50_ms,
                r.dec.p99_ms,
            );
        }
    }
    println!();
}
