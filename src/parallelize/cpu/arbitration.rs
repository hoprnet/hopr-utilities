//! Encode/decode arbitration for the shared Rayon pool (submodule of [`super`]).
//!
//! The encode (SPHINX wrap + SURB generation) and decode (SPHINX peel) paths share the single Rayon
//! pool. Left unmanaged, a heavy download floods decode (forwarding) and starves SURB encode → the
//! SURB ring drains → download collapses (#8246).
//!
//! The arbiter is deliberately **asymmetric and occupancy-gated** so it never harms the common case
//! (measured: a symmetric cap that always engaged halved forwarding):
//!   * Only DECODE is ever throttled; ENCODE never blocks (it is the light, latency-critical, protected class — SURB
//!     generation).
//!   * Decode is throttled ONLY when all of: the pool is genuinely saturated (running threads ≥ `occupancy_pct`), AND
//!     encode work is actually present, AND decode already exceeds its share.
//!   * A pure forwarding relay (encode ≈ 0) is therefore never throttled; nor is any node whose pool is below the
//!     occupancy threshold — so the arbiter adds ~zero overhead until genuinely needed.
//!
//! Occupancy is measured by [`RUNNING`] (tasks actually executing on a pool thread), not the
//! queued+running `*_OUTSTANDING` counters, which the deep pipeline ready-queues keep saturated.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};

use event_listener::IntoNotification;

use super::{DECODE_OUTSTANDING, ENCODE_OUTSTANDING, pool_thread_count};

/// Configuration of the encode/decode pool arbiter. Modelled as an enum so the disabled state cannot
/// carry (and silently ignore) tuning percentages — an illegal combination is unrepresentable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArbitrationConfig {
    /// Decode is never throttled; the pool is shared first-come-first-served.
    Disabled,
    /// Decode yields pool slots to encode under a genuine flood.
    Enabled {
        /// Pool occupancy (percent of threads actually running) at or above which decode admission
        /// may engage. Below it, decode is never throttled.
        occupancy_pct: u32,
        /// Share of the pool (percent) that decode yields to encode when both contend under saturation.
        encode_reserve_pct: u32,
    },
}

/// A percentage in an [`ArbitrationConfig::Enabled`] fell outside `1..=100`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ArbitrationConfigError {
    /// `occupancy_pct` was out of range.
    #[error("occupancy_pct must be in 1..=100, got {0}")]
    OccupancyOutOfRange(u32),
    /// `encode_reserve_pct` was out of range.
    #[error("encode_reserve_pct must be in 1..=100, got {0}")]
    EncodeReserveOutOfRange(u32),
}

impl ArbitrationConfig {
    /// Both percentages must lie in `1..=100` when arbitration is enabled.
    ///
    /// # Examples
    /// ```
    /// # use hopr_utilities::parallelize::cpu::ArbitrationConfig;
    /// assert!(ArbitrationConfig::Disabled.validate().is_ok());
    /// assert!(
    ///     ArbitrationConfig::Enabled {
    ///         occupancy_pct: 75,
    ///         encode_reserve_pct: 50
    ///     }
    ///     .validate()
    ///     .is_ok()
    /// );
    /// assert!(
    ///     ArbitrationConfig::Enabled {
    ///         occupancy_pct: 0,
    ///         encode_reserve_pct: 50
    ///     }
    ///     .validate()
    ///     .is_err()
    /// );
    /// ```
    pub fn validate(&self) -> Result<(), ArbitrationConfigError> {
        if let ArbitrationConfig::Enabled {
            occupancy_pct,
            encode_reserve_pct,
        } = self
        {
            if !(1..=100).contains(occupancy_pct) {
                return Err(ArbitrationConfigError::OccupancyOutOfRange(*occupancy_pct));
            }
            if !(1..=100).contains(encode_reserve_pct) {
                return Err(ArbitrationConfigError::EncodeReserveOutOfRange(*encode_reserve_pct));
            }
        }
        Ok(())
    }
}

/// When `false`, [`super::spawn_decode_blocking`] skips admission entirely.
static ARBITRATION_ENABLED: AtomicBool = AtomicBool::new(true);
/// Pool-occupancy threshold (percent) at or above which decode admission may engage.
static ARBITRATION_OCCUPANCY_PCT: AtomicU32 = AtomicU32::new(75);
/// Fraction (percent) of the pool that decode yields to encode when both contend under saturation.
static ARBITRATION_ENCODE_RESERVE_PCT: AtomicU32 = AtomicU32::new(50);
/// Tasks currently executing on a pool thread (true occupancy, `0..=pool_thread_count`).
static RUNNING: AtomicUsize = AtomicUsize::new(0);
/// Notified when a running task finishes (a pool slot frees) so a blocked decode admitter retries.
static ARBITRATION_EVENT: event_listener::Event = event_listener::Event::new();
/// Set once the arbiter has been configured, so [`with_arbitration_once`] is first-wins. The arbiter
/// is process-global; in a multi-node-per-process host (tests, the cluster example) this stops a
/// later node's pipeline startup from clobbering the arbiter for already-running pipelines.
static ARBITRATION_CONFIGURED: AtomicBool = AtomicBool::new(false);

/// Returns the number of tasks currently executing on a pool thread.
#[inline]
pub fn running_tasks() -> usize {
    RUNNING.load(Ordering::Relaxed)
}

/// Applies `config` to the arbiter's statics; percentages are clamped defensively to `1..=100`.
fn apply(config: ArbitrationConfig) {
    match config {
        ArbitrationConfig::Disabled => ARBITRATION_ENABLED.store(false, Ordering::Relaxed),
        ArbitrationConfig::Enabled {
            occupancy_pct,
            encode_reserve_pct,
        } => {
            ARBITRATION_OCCUPANCY_PCT.store(occupancy_pct.clamp(1, 100), Ordering::Relaxed);
            ARBITRATION_ENCODE_RESERVE_PCT.store(encode_reserve_pct.clamp(1, 100), Ordering::Relaxed);
            ARBITRATION_ENABLED.store(true, Ordering::Relaxed);
        }
    }
    // A live policy change (e.g. an explicit `with_arbitration(Disabled)` override) must wake every
    // blocked decode admitter: they re-evaluate the gate on wake, so a disable or a loosened cap
    // takes effect immediately instead of stranding them under the previous policy.
    ARBITRATION_EVENT.notify(usize::MAX);
}

/// Configure the arbiter. Call once at startup, next to [`super::init_thread_pool`].
///
/// Applies unconditionally (last write wins) and marks the arbiter configured — use this for an
/// explicit process-level override (e.g. a benchmark toggling on/off). Node startup should prefer
/// [`with_arbitration_once`] so it cannot clobber such an override or a peer pipeline.
pub fn with_arbitration(config: ArbitrationConfig) {
    apply(config);
    ARBITRATION_CONFIGURED.store(true, Ordering::Release);
}

/// Configure the arbiter only if it has not been configured yet, returning `true` if this call
/// applied the settings. Idempotent and first-wins: the arbiter is a single process-global, so a
/// per-node pipeline startup uses this to configure it once without overwriting an earlier explicit
/// [`with_arbitration`] or another node's already-applied settings.
pub fn with_arbitration_once(config: ArbitrationConfig) -> bool {
    if ARBITRATION_CONFIGURED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return false; // already configured by an earlier call — leave it untouched
    }
    apply(config);
    true
}

/// Awaits until the decode path may submit without starving encode. Passthrough when arbitration is
/// disabled or the pool is uninitialised; see the module policy note above.
pub(super) async fn admit_decode() {
    let pool_threads = pool_thread_count();
    // Disabled, or pool not yet initialised → reserve the slot and pass straight through.
    if !ARBITRATION_ENABLED.load(Ordering::Relaxed) || pool_threads == 0 {
        DECODE_OUTSTANDING.fetch_add(1, Ordering::Relaxed);
        return;
    }
    let occupancy_pct = ARBITRATION_OCCUPANCY_PCT.load(Ordering::Relaxed);
    let reserve_pct = ARBITRATION_ENCODE_RESERVE_PCT.load(Ordering::Relaxed);
    loop {
        // Re-check on every iteration: `with_arbitration(Disabled)` can turn the arbiter off while
        // this admitter is blocked, and `apply` wakes us to observe it here (pass straight through).
        if !ARBITRATION_ENABLED.load(Ordering::Relaxed) {
            DECODE_OUTSTANDING.fetch_add(1, Ordering::Relaxed);
            return;
        }
        if try_reserve_decode(pool_threads, occupancy_pct, reserve_pct) {
            return;
        }
        // Register before re-checking so a slot freed — or a disable — in between is not missed.
        let listener = ARBITRATION_EVENT.listen();
        if !ARBITRATION_ENABLED.load(Ordering::Relaxed) {
            DECODE_OUTSTANDING.fetch_add(1, Ordering::Relaxed);
            return;
        }
        if try_reserve_decode(pool_threads, occupancy_pct, reserve_pct) {
            return;
        }
        listener.await;
    }
}

/// Atomically reserves one decode slot — increments `DECODE_OUTSTANDING` iff decode is currently
/// under its admission cap — and returns `true`. Returns `false` without reserving when the cap is
/// full. The compare-exchange makes the cap a hard limit: concurrent admitters cannot all pass a
/// stale `dec < cap` check and overshoot the encode reservation.
fn try_reserve_decode(pool_threads: usize, occupancy_pct: u32, reserve_pct: u32) -> bool {
    loop {
        // `Acquire` on the predicate loads pairs with the `Release` decrements in the drop of the
        // slot guards: it closes the register-then-recheck window so a slot freed just before
        // `listen()` is observed here rather than parking the waiter until the next task completes.
        let dec = DECODE_OUTSTANDING.load(Ordering::Acquire);
        let limit = decode_admit_cap(
            pool_threads,
            RUNNING.load(Ordering::Acquire),
            dec,
            ENCODE_OUTSTANDING.load(Ordering::Acquire),
            occupancy_pct,
            reserve_pct,
        )
        .unwrap_or(usize::MAX); // `None` = unconstrained
        if dec >= limit {
            return false;
        }
        if DECODE_OUTSTANDING
            .compare_exchange_weak(dec, dec + 1, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
        {
            return true;
        }
    }
}

/// Decode must outnumber encode outstanding by at least this factor to count as a "flood" worth
/// throttling. Below it the two demands are comparable, FIFO already serves them fairly, and capping
/// decode would needlessly cut delivery throughput (a delivered packet needs both a decode and the
/// encode that replenishes its SURB).
///
/// Deliberately a fixed structural invariant, not an [`ArbitrationConfig`] knob: unlike
/// `occupancy_pct`/`encode_reserve_pct` (which tune how hard the gate reserves), the flood ratio
/// defines *what counts as a flood at all* and changing it risks either never engaging or throttling
/// balanced load — not something an operator should tune blind.
const DECODE_FLOOD_FACTOR: usize = 2;

/// The decode-admission policy, as a pure function so it is trivially testable.
///
/// Returns `None` when decode is unconstrained — the pool is below the occupancy threshold
/// (`running < occupancy_pct% of pool`), there is no encode work to protect (`enc == 0`, the
/// pure-forwarding-relay case), or decode is not flooding relative to encode. Otherwise returns
/// `Some(cap)`, the maximum decode tasks allowed outstanding so that up to
/// `min(enc, encode_reserve_pct% of pool)` threads are left for encode; floored at 1 for liveness.
fn decode_admit_cap(
    pool_threads: usize,
    running: usize,
    decode_outstanding: usize,
    enc_outstanding: usize,
    occupancy_pct: u32,
    encode_reserve_pct: u32,
) -> Option<usize> {
    // `.max(1)` so tiny pools (1–2 threads) don't floor the threshold to 0, which would treat an idle
    // pool as "saturated" and defeat the gate on exactly the small hosts #8246 targeted.
    let occupancy_threshold = ((pool_threads as u64 * occupancy_pct as u64 / 100) as usize).max(1);
    if running < occupancy_threshold {
        return None; // pool not saturated → never throttle
    }
    if enc_outstanding == 0 {
        return None; // nothing to protect (pure forwarding relay) → never throttle
    }
    if decode_outstanding <= enc_outstanding.saturating_mul(DECODE_FLOOD_FACTOR) {
        return None; // decode is not flooding relative to encode → FIFO is fair, leave it alone
    }
    let reserve = ((pool_threads as u64 * encode_reserve_pct as u64).div_ceil(100) as usize).min(enc_outstanding);
    Some(pool_threads.saturating_sub(reserve).max(1))
}

/// RAII guard that decrements a tagged outstanding counter when dropped, waking a blocked decode
/// admitter. Caller must increment the counter before constructing this guard.
pub(super) struct TaggedGuard(&'static AtomicUsize);

impl TaggedGuard {
    #[inline]
    pub(super) fn new(counter: &'static AtomicUsize) -> Self {
        Self(counter)
    }
}

impl Drop for TaggedGuard {
    #[inline]
    fn drop(&mut self) {
        // `Release` publishes this gate-opening decrement to the `Acquire` predicate loads in
        // `try_reserve_decode`.
        self.0.fetch_sub(1, Ordering::Release);
        // Decrementing DECODE_OUTSTANDING (frees a decode slot) or ENCODE_OUTSTANDING (raises the
        // decode cap / opens the flood-gate) can make a blocked decode admitter admissible; the
        // admission predicate reads these counters, so wake a waiter here. `additional()` makes each
        // freed slot wake a *distinct* waiter (plain `notify(1)` coalesces and would strand waiters
        // when several tasks finish at once).
        ARBITRATION_EVENT.notify(1.additional());
    }
}

/// RAII guard tracking a task that is actually executing on a pool thread. Increments [`RUNNING`] on
/// construction; on drop decrements it and wakes one blocked decode admitter (a slot freed).
pub(super) struct RunningGuard;

impl RunningGuard {
    #[inline]
    pub(super) fn enter() -> Self {
        RUNNING.fetch_add(1, Ordering::Relaxed);
        Self
    }
}

impl Drop for RunningGuard {
    #[inline]
    fn drop(&mut self) {
        // `Release` pairs with the `Acquire` occupancy load in `try_reserve_decode`.
        RUNNING.fetch_sub(1, Ordering::Release);
        // A finishing task (including untagged acks) lowers occupancy, which can open the occupancy
        // gate for a blocked decode admitter. Wake a distinct waiter per freed slot.
        ARBITRATION_EVENT.notify(1.additional());
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use serial_test::serial;

    use super::{
        ARBITRATION_CONFIGURED, ARBITRATION_ENABLED, ArbitrationConfig, DECODE_OUTSTANDING, ENCODE_OUTSTANDING,
        RUNNING, admit_decode, decode_admit_cap, try_reserve_decode, with_arbitration, with_arbitration_once,
    };

    const N: usize = 8;
    const OCC: u32 = 75; // occupancy threshold percent
    const RES: u32 = 50; // encode reserve percent
    const FLOOD: usize = N * 4; // decode_outstanding well above enc*FLOOD_FACTOR

    const ENABLED: ArbitrationConfig = ArbitrationConfig::Enabled {
        occupancy_pct: OCC,
        encode_reserve_pct: RES,
    };

    /// Resets every process-global arbiter static to its fresh-process default, so a serial test
    /// never inherits state a prior one left behind.
    fn reset_arbitration() {
        ARBITRATION_ENABLED.store(true, Ordering::Relaxed);
        ARBITRATION_CONFIGURED.store(false, Ordering::Release);
        RUNNING.store(0, Ordering::Release);
        ENCODE_OUTSTANDING.store(0, Ordering::Release);
        DECODE_OUTSTANDING.store(0, Ordering::Release);
    }

    #[test]
    fn below_occupancy_threshold_decode_is_never_throttled() {
        // 75% of 8 == 6 running threads; anything below leaves decode unconstrained.
        for running in 0..(N * OCC as usize / 100) {
            assert_eq!(
                decode_admit_cap(N, running, FLOOD, 4, OCC, RES),
                None,
                "decode must be free below the occupancy threshold (running={running})",
            );
        }
    }

    #[test]
    fn saturated_but_no_encode_is_never_throttled() {
        // Pure forwarding relay: pool full of decode, zero encode → nothing to protect.
        assert_eq!(decode_admit_cap(N, N, FLOOD, 0, OCC, RES), None);
    }

    #[test]
    fn balanced_load_is_not_throttled() {
        // Saturated with encode present, but decode is comparable to encode (not a flood) → FIFO is
        // already fair, throttling would only cut delivery. Only a genuine flood engages.
        assert_eq!(decode_admit_cap(N, N, N, N, OCC, RES), None); // dec == enc
        assert_eq!(decode_admit_cap(N, N, 2 * N, N, OCC, RES), None); // dec == enc * FLOOD_FACTOR
        assert!(decode_admit_cap(N, N, 2 * N + 1, N, OCC, RES).is_some()); // just over the threshold
    }

    #[test]
    fn decode_flood_reserves_threads_for_encode() {
        // Decode floods (>> encode) → decode capped so 50% is left for encode.
        assert_eq!(decode_admit_cap(N, N, FLOOD, N, OCC, RES), Some(N / 2));
        // Reserve never exceeds actual encode demand: only 1 encode task → reserve just 1 thread.
        assert_eq!(decode_admit_cap(N, N, FLOOD, 1, OCC, RES), Some(N - 1));
    }

    #[test]
    fn cap_is_never_zero_liveness() {
        // A full encode reserve on a tiny pool still admits at least one decode task.
        assert_eq!(decode_admit_cap(2, 2, 50, 10, OCC, 100), Some(1));
        for running in 0..=N {
            for enc in 1..=N {
                if let Some(cap) = decode_admit_cap(N, running, FLOOD, enc, OCC, RES) {
                    assert!(cap >= 1, "cap must never be zero (running={running}, enc={enc})");
                }
            }
        }
    }

    #[test]
    fn config_validation_rejects_out_of_range_percentages() {
        assert!(ENABLED.validate().is_ok());
        assert!(ArbitrationConfig::Disabled.validate().is_ok());
        assert!(
            ArbitrationConfig::Enabled {
                occupancy_pct: 0,
                encode_reserve_pct: 50
            }
            .validate()
            .is_err()
        );
        assert!(
            ArbitrationConfig::Enabled {
                occupancy_pct: 75,
                encode_reserve_pct: 101
            }
            .validate()
            .is_err()
        );
    }

    /// Drives `try_reserve_decode` directly — the atomic CAS reservation wrapper around
    /// `decode_admit_cap` that the async `admit_decode` loop delegates to. Unit tests run with an
    /// uninitialised pool, so `admit_decode` short-circuits and never exercises this path; this test
    /// simulates a saturated pool by setting the counters directly.
    #[test]
    #[serial]
    fn try_reserve_decode_refuses_under_flood_and_admits_otherwise() {
        reset_arbitration();

        // Saturated pool + encode present + decode flooding → the cap engages and decode is already
        // over it, so a fresh reservation is refused and the counter is left untouched.
        RUNNING.store(N, Ordering::Release);
        ENCODE_OUTSTANDING.store(N, Ordering::Release);
        DECODE_OUTSTANDING.store(FLOOD, Ordering::Release);
        assert!(!try_reserve_decode(N, OCC, RES), "must refuse a decode flood");
        assert_eq!(
            DECODE_OUTSTANDING.load(Ordering::Relaxed),
            FLOOD,
            "a refused call must not reserve"
        );

        // Same saturation, but decode is only comparable to encode (not a flood) → unconstrained, so
        // the reservation succeeds and increments the counter by exactly one.
        DECODE_OUTSTANDING.store(N, Ordering::Release);
        assert!(
            try_reserve_decode(N, OCC, RES),
            "must admit when decode is not flooding"
        );
        assert_eq!(
            DECODE_OUTSTANDING.load(Ordering::Relaxed),
            N + 1,
            "an admitted call reserves one slot"
        );

        // Below the occupancy threshold decode is never throttled, even while flooding.
        RUNNING.store(0, Ordering::Release);
        DECODE_OUTSTANDING.store(FLOOD, Ordering::Release);
        assert!(
            try_reserve_decode(N, OCC, RES),
            "must admit below the occupancy threshold"
        );
        assert_eq!(DECODE_OUTSTANDING.load(Ordering::Relaxed), FLOOD + 1);

        reset_arbitration();
    }

    /// `with_arbitration_once` is first-wins: the first caller applies, later ones no-op, so a
    /// per-node pipeline startup can never clobber the process-global arbiter of a peer.
    #[test]
    #[serial]
    fn with_arbitration_once_is_first_wins() {
        reset_arbitration();

        assert!(
            with_arbitration_once(ArbitrationConfig::Disabled),
            "first call must apply"
        );
        assert!(
            !ARBITRATION_ENABLED.load(Ordering::Relaxed),
            "first call's value must stick"
        );

        // A later differing call is ignored — the first configuration wins.
        assert!(!with_arbitration_once(ENABLED), "second call must be a no-op");
        assert!(
            !ARBITRATION_ENABLED.load(Ordering::Relaxed),
            "second call must not overwrite"
        );

        // An explicit `with_arbitration` still overrides unconditionally (last-write-wins).
        with_arbitration(ENABLED);
        assert!(
            ARBITRATION_ENABLED.load(Ordering::Relaxed),
            "explicit config must override"
        );

        reset_arbitration();
    }

    /// Disabled arbitration and an uninitialised pool are both pure passthrough (never block).
    /// `admit_decode` reserves the `DECODE_OUTSTANDING` slot, so we release it after each call.
    #[tokio::test]
    #[serial]
    async fn admit_decode_is_passthrough_when_disabled_or_pool_uninitialised() {
        reset_arbitration();

        // Pool is uninitialised in unit tests (pool_thread_count() == 0) → immediate return.
        tokio::time::timeout(std::time::Duration::from_secs(5), admit_decode())
            .await
            .expect("admit_decode must not block with an uninitialised pool");
        DECODE_OUTSTANDING.fetch_sub(1, Ordering::Relaxed);

        with_arbitration(ArbitrationConfig::Disabled);
        tokio::time::timeout(std::time::Duration::from_secs(5), admit_decode())
            .await
            .expect("admit_decode must not block when disabled");
        DECODE_OUTSTANDING.fetch_sub(1, Ordering::Relaxed);

        reset_arbitration();
    }
}
