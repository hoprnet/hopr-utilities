//! Shared scaffolding for the `pool_arbiter_*` benches: the busy-spin emulator, the **measured**
//! SPHINX op costs, the delivered-payload size, and the production pool sizing. Kept in one place so the
//! op-cost numbers — the load-bearing input to every scenario — have a single source of truth.
//!
//! Measured on this machine via `hopr-crypto-packet`'s `packet_bench` (criterion medians):
//!   encode (`packet_sending_no_precomputation`, 1 SURB) = 80 / 228 / 336 / 436 µs for 0/1/2/3 hop
//!   encode 3-hop / 2-SURB (worst case)                  = 471 µs
//!   decode (`packet_forwarding`, any hop, one peel)     = 98 µs
//!
//! Included by each bench with `#[path = "bench_common/mod.rs"] mod common;`. Not every bench uses
//! every item, so individual `dead_code` is expected per compilation unit.
#![allow(dead_code)]

use std::time::{Duration, Instant};

/// Full SPHINX encode (wrap + SURB generation) cost in µs, indexed by hop count, 1 SURB.
pub const ENCODE_US_BY_HOP: [u64; 4] = [80, 228, 336, 436];
/// Full SPHINX encode cost for the 3-hop / 2-SURB worst case (µs).
pub const ENCODE_US_3HOP_2SURB: u64 = 471;
/// SPHINX decode (peel / forward) cost in µs — hop-independent (one peel per node).
pub const DECODE_US: u64 = 98;
/// App bytes delivered per packet (`SESSION_MTU`; the SPHINX wire payload is 1038 B and, being
/// onion-padded, is constant across hop counts).
pub const PAYLOAD_BYTES: u64 = 1020;

/// Busy-emulate a CPU op of `us` microseconds by spinning (models a pool thread blocked in crypto).
#[inline]
pub fn spin(us: u64) {
    let end = Instant::now() + Duration::from_micros(us);
    while Instant::now() < end {
        std::hint::spin_loop();
    }
}

/// Production pool sizing: `available_parallelism / 2`, floored so the pool is never degenerate.
#[inline]
pub fn pool_size() -> usize {
    std::thread::available_parallelism()
        .map(|n| (n.get() / 2).max(2))
        .unwrap_or(4)
}

/// Convert a packet rate (packets/s) to delivered MB/s at [`PAYLOAD_BYTES`].
#[inline]
pub fn pps_to_mb(pps: f64) -> f64 {
    pps * PAYLOAD_BYTES as f64 / (1024.0 * 1024.0)
}

/// The arbiter config a bench uses to toggle arbitration on/off at the default 75/50 tuning.
#[inline]
pub fn arbitration(enabled: bool) -> hopr_utilities::parallelize::cpu::ArbitrationConfig {
    use hopr_utilities::parallelize::cpu::ArbitrationConfig;
    if enabled {
        ArbitrationConfig::Enabled {
            occupancy_pct: 75,
            encode_reserve_pct: 50,
        }
    } else {
        ArbitrationConfig::Disabled
    }
}
