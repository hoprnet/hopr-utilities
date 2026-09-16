//! Per-hop pipeline throughput on a fully-engaged pool, busy-emulated with the **real** SPHINX op
//! costs measured on this machine (see `bench_common`), harness = false. Throughput is delivered
//! goodput in MB/s at SESSION_MTU (1020 B/pkt; onion-padded, so constant across hop counts).
//!
//! Two framings are reported, because they answer different questions:
//!
//!  1. **End-to-end session ceiling** (the headline): a delivered N-hop packet is encoded once at the
//!     source and peeled once at each of the N relays + the destination — but those N+2 ops run on
//!     *independent pools on different machines* that pipeline, so the achievable session throughput
//!     is the **slowest single role**, `min(source-encode, relay-forward, dest-decode)`, NOT their
//!     sum. Encode (SURB generation) is the expensive op, so ≥1-hop sessions are source-encode-bound
//!     while relays forward with >2× headroom.
//!  2. **Single-pool aggregate** (reference only): all `1 encode + (N+1) decodes` charged to one pool
//!     — i.e. one machine doing the whole path's crypto. This is a lower bound, not session
//!     throughput; it is the natural place to show the arbiter adds zero overhead (ON ≈ OFF).
//!
//! Run: `cargo bench --features parallelize-rayon --bench pool_arbiter_hops`

#[path = "bench_common/mod.rs"]
mod common;

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Duration,
};

use common::{DECODE_US, ENCODE_US_BY_HOP, pool_size, pps_to_mb, spin};
use hopr_utilities::parallelize::cpu;

const WORKERS: usize = 32;
const WINDOW: Duration = Duration::from_millis(2000);

/// Saturate the pool with `WORKERS` packet-workers; each does `enc_us` encode (skipped if 0) + `decodes`
/// decodes per packet. Returns delivered packets/s.
async fn pipeline_pps(enc_us: u64, decodes: usize) -> f64 {
    let stop = Arc::new(AtomicBool::new(false));
    let pkts = Arc::new(AtomicU64::new(0));
    let mut tasks = Vec::new();
    for _ in 0..WORKERS {
        let (stop, pkts) = (stop.clone(), pkts.clone());
        tasks.push(tokio::spawn(async move {
            while !stop.load(Ordering::Relaxed) {
                if enc_us > 0 {
                    let _ = cpu::spawn_encode_blocking(move || spin(enc_us), "enc").await;
                }
                for _ in 0..decodes {
                    let _ = cpu::spawn_decode_blocking(|| spin(DECODE_US), "dec").await;
                }
                pkts.fetch_add(1, Ordering::Relaxed);
            }
        }));
    }
    tokio::time::sleep(WINDOW).await;
    stop.store(true, Ordering::Relaxed);
    for t in tasks {
        let _ = t.await;
    }
    pkts.load(Ordering::Relaxed) as f64 / WINDOW.as_secs_f64()
}

fn main() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(8)
        .enable_all()
        .build()
        .unwrap();
    let pool = pool_size();
    let _ = cpu::init_thread_pool(pool);
    cpu::with_arbitration(common::arbitration(true));

    // ── Isolated per-role capacities (arbiter on; each role is a distinct machine's pool) ──────────
    let decode_mb = pps_to_mb(rt.block_on(pipeline_pps(0, 1))); // relay forward / dest receive: one peel
    let encode_mb: Vec<f64> = (0..=3)
        .map(|h| pps_to_mb(rt.block_on(pipeline_pps(ENCODE_US_BY_HOP[h], 0))))
        .collect();

    println!(
        "\n## Fully-engaged pool = {pool} threads, real busy-emulated SPHINX ops (dec={DECODE_US}µs, enc 0/1/2/3-hop = {:?}µs, 1020 B/pkt)\n",
        ENCODE_US_BY_HOP
    );

    println!("### Isolated per-role capacity (each role runs on its own pool)");
    println!("| role | pkts/s | MB/s |");
    println!("|---|--:|--:|");
    let dec_pps = decode_mb * 1024.0 * 1024.0 / common::PAYLOAD_BYTES as f64;
    println!("| relay forward / dest decode (any hop) | {dec_pps:.0} | {decode_mb:.2} |");
    for (h, &mb) in encode_mb.iter().enumerate() {
        let pps = mb * 1024.0 * 1024.0 / common::PAYLOAD_BYTES as f64;
        println!("| source encode {h}-hop | {pps:.0} | {mb:.2} |");
    }

    // ── End-to-end session ceiling = min over the roles on the path (headline) ─────────────────────
    println!("\n### End-to-end session throughput = min(source encode, relay/dest decode)  ← real ceiling");
    println!("| hops | source encode MB/s | relay/dest decode MB/s | **end-to-end MB/s** | bound by |");
    println!("|---|--:|--:|--:|---|");
    for h in 0..=3 {
        let enc = encode_mb[h];
        let e2e = enc.min(decode_mb);
        let bound = if enc <= decode_mb {
            "source encode"
        } else {
            "dest decode"
        };
        // 0-hop has no relays; ≥1-hop relays forward at `decode_mb` each (never the bottleneck here).
        let relay = if h == 0 {
            "—".to_string()
        } else {
            format!("{decode_mb:.2}")
        };
        println!("| {h} | {enc:.2} | {relay} | **{e2e:.2}** | {bound} |");
    }

    // ── Single-pool aggregate (reference; NOT session throughput) — shows arbiter overhead is zero ──
    println!("\n### Single-pool aggregate: whole-path crypto on ONE pool (reference lower bound; arbiter ON vs OFF)");
    println!("| hops | work/pkt (enc + (n+1)×dec) | arbiter | pkts/s | MB/s |");
    println!("|---|---|---|--:|--:|");
    for hops in 0usize..=3 {
        let enc = ENCODE_US_BY_HOP[hops];
        let decodes = hops + 1;
        for enabled in [true, false] {
            cpu::with_arbitration(common::arbitration(enabled));
            let pps = rt.block_on(pipeline_pps(enc, decodes));
            println!(
                "| {hops} | {enc}µs + {decodes}×{DECODE_US}µs | {} | {pps:.0} | {:.2} |",
                if enabled { "ON " } else { "off" },
                pps_to_mb(pps),
            );
        }
    }
    println!();
}
