//! Parallelization utilities for CPU-heavy blocking workloads.
//!
//! This crate provides async-friendly wrappers around Rayon's thread pool for offloading
//! CPU-intensive operations (EC multiplication, ECDSA signing, MAC verification) from
//! async executor threads.
//!
//! For background on async executors and blocking, see
//! [Async: What is blocking?](https://ryhl.io/blog/async-what-is-blocking/).
//!
//! See the [`cpu`] module for the primary API.

/// Factor of `pool_thread_count` above which the encode pool is considered to be
/// under high pressure.
///
/// With 0.5: pressure is considered high once encode tasks occupy more than half
/// the pool's thread count (as outstanding = queued + running).
const ENCODE_PRESSURE_HIGH_FACTOR: f64 = 0.5;

/// Returns `true` when the encode pool has headroom for another encode task.
///
/// Reads the current [`cpu::ENCODE_OUTSTANDING`] and [`cpu::pool_thread_count`]
/// at call time, avoiding the stale-flag problem that a cached `AtomicBool`
/// introduces.  When the `parallelize-rayon` feature is not enabled (e.g. in
/// unit tests) this always returns `true`.
#[inline]
pub fn encode_pool_has_headroom() -> bool {
    #[cfg(feature = "parallelize-rayon")]
    {
        let threads = cpu::pool_thread_count();
        if threads == 0 {
            return true; // pool not initialised yet — don't block
        }
        let outstanding = cpu::ENCODE_OUTSTANDING.load(std::sync::atomic::Ordering::Relaxed);
        (outstanding as f64) < threads as f64 * ENCODE_PRESSURE_HIGH_FACTOR
    }
    #[cfg(not(feature = "parallelize-rayon"))]
    true
}

/// Module for thread pool-based parallelization of CPU-heavy blocking workloads.
///
/// ## Zombie Task Prevention
///
/// The Rayon thread pool is sized to CPU cores for crypto operations. Callers wrap
/// tasks with timeouts (e.g., 150ms for packet decoding). When a timeout fires, the
/// async receiver is dropped, but Rayon has no native cancellation—the closure
/// continues as a "zombie" task whose result is discarded.
///
/// Under sustained load, zombie accumulation can starve the pool: timed-out tasks
/// continue occupying threads, causing subsequent tasks to also time out. To break
/// this cycle, each spawned closure checks `tx.is_canceled()` before executing.
/// If the receiver was dropped while queued, the closure returns immediately.
///
/// ## Queue Depth Limiting
///
/// To prevent unbounded queue growth, the module tracks outstanding tasks (queued +
/// running). Use [`cpu::spawn_blocking`] or [`cpu::spawn_fifo_blocking`] which return
/// [`cpu::SpawnError::QueueFull`] when the configured limit is reached.
///
/// Set `HOPR_CPU_TASK_QUEUE_LIMIT` environment variable to enable limiting.
///
/// ## Observability
///
/// Prometheus metrics (behind the `telemetry` feature) track:
/// - **submitted**: total tasks entering the queue
/// - **completed**: tasks that delivered results to a live receiver
/// - **cancelled**: tasks skipped via cooperative cancellation
/// - **orphaned**: tasks that ran but whose receiver was dropped during execution
/// - **rejected**: tasks rejected due to queue being full
/// - **queue_wait**: histogram of queue wait time
/// - **execution_time**: histogram of task execution duration
/// - **outstanding_tasks**: current queued + running tasks
/// - **queue_limit**: configured maximum (for comparison)
#[cfg(feature = "parallelize-rayon")]
pub mod cpu {
    use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};

    use futures::channel::oneshot;
    pub use rayon;

    /// Histogram buckets for timing metrics (seconds).
    #[cfg(all(feature = "parallelize", feature = "telemetry", not(test)))]
    const TIMING_BUCKETS: &[f64] = &[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0];

    mod metrics {
        #[cfg(any(not(all(feature = "parallelize", feature = "telemetry")), test))]
        pub use noop::*;
        #[cfg(all(feature = "parallelize", feature = "telemetry", not(test)))]
        pub use real::*;

        #[cfg(all(feature = "parallelize", feature = "telemetry", not(test)))]
        mod real {
            use lazy_static::lazy_static;

            lazy_static! {
                static ref TASKS_SUBMITTED: hopr_types::telemetry::SimpleCounter =
                    hopr_types::telemetry::SimpleCounter::new(
                        "hopr_rayon_tasks_submitted_total",
                        "Total number of tasks submitted to the Rayon thread pool",
                    )
                    .unwrap();
                static ref TASKS_COMPLETED: hopr_types::telemetry::SimpleCounter =
                    hopr_types::telemetry::SimpleCounter::new(
                        "hopr_rayon_tasks_completed_total",
                        "Total number of Rayon tasks that completed and delivered results",
                    )
                    .unwrap();
                static ref TASKS_CANCELLED: hopr_types::telemetry::SimpleCounter =
                    hopr_types::telemetry::SimpleCounter::new(
                        "hopr_rayon_tasks_cancelled_total",
                        "Total number of Rayon tasks skipped because receiver was already dropped",
                    )
                    .unwrap();
                static ref TASKS_ORPHANED: hopr_types::telemetry::SimpleCounter =
                    hopr_types::telemetry::SimpleCounter::new(
                        "hopr_rayon_tasks_orphaned_total",
                        "Total number of Rayon tasks whose results were discarded after completion",
                    )
                    .unwrap();
                static ref TASKS_REJECTED: hopr_types::telemetry::SimpleCounter =
                    hopr_types::telemetry::SimpleCounter::new(
                        "hopr_rayon_tasks_rejected_total",
                        "Total number of tasks rejected due to queue being full",
                    )
                    .unwrap();
                static ref QUEUE_WAIT: hopr_types::telemetry::SimpleHistogram =
                    hopr_types::telemetry::SimpleHistogram::new(
                        "hopr_rayon_queue_wait_seconds",
                        "Time tasks spend waiting in the Rayon queue before execution starts",
                        super::super::TIMING_BUCKETS.to_vec(),
                    )
                    .unwrap();
                static ref EXECUTION_TIME: hopr_types::telemetry::MultiHistogram =
                    hopr_types::telemetry::MultiHistogram::new(
                        "hopr_rayon_execution_seconds",
                        "Time tasks spend executing in the Rayon thread pool",
                        super::super::TIMING_BUCKETS.to_vec(),
                        &["operation"],
                    )
                    .unwrap();
                static ref OUTSTANDING_TASKS: hopr_types::telemetry::SimpleGauge =
                    hopr_types::telemetry::SimpleGauge::new(
                        "hopr_rayon_outstanding_tasks",
                        "Current number of tasks queued or running in the Rayon pool",
                    )
                    .unwrap();
                static ref QUEUE_LIMIT: hopr_types::telemetry::SimpleGauge = hopr_types::telemetry::SimpleGauge::new(
                    "hopr_rayon_queue_limit",
                    "Configured maximum outstanding tasks for the Rayon thread pool",
                )
                .unwrap();
            }

            #[inline]
            pub fn submitted() {
                TASKS_SUBMITTED.increment();
            }

            #[inline]
            pub fn completed() {
                TASKS_COMPLETED.increment();
            }

            #[inline]
            pub fn cancelled() {
                TASKS_CANCELLED.increment();
            }

            #[inline]
            pub fn orphaned() {
                TASKS_ORPHANED.increment();
            }

            #[inline]
            pub fn rejected() {
                TASKS_REJECTED.increment();
            }

            #[inline]
            pub fn observe_queue_wait(seconds: f64) {
                QUEUE_WAIT.observe(seconds);
            }

            #[inline]
            pub fn observe_execution(operation: &str, seconds: f64) {
                EXECUTION_TIME.observe(&[operation], seconds);
            }

            #[inline]
            pub fn outstanding_inc() {
                OUTSTANDING_TASKS.increment(1.0);
            }

            #[inline]
            pub fn outstanding_dec() {
                OUTSTANDING_TASKS.decrement(1.0);
            }

            #[inline]
            pub fn set_queue_limit(limit: usize) {
                QUEUE_LIMIT.set(limit as f64);
            }
        }

        #[cfg(any(not(all(feature = "parallelize", feature = "telemetry")), test))]
        mod noop {
            #[inline]
            pub fn submitted() {}
            #[inline]
            pub fn completed() {}
            #[inline]
            pub fn cancelled() {}
            #[inline]
            pub fn orphaned() {}
            #[inline]
            pub fn rejected() {}
            #[inline]
            pub fn observe_queue_wait(_: f64) {}
            #[inline]
            pub fn observe_execution(_: &str, _: f64) {}
            #[inline]
            pub fn outstanding_inc() {}
            #[inline]
            pub fn outstanding_dec() {}
            #[inline]
            pub fn set_queue_limit(_: usize) {}
        }
    }

    /// Current number of outstanding tasks (queued + running).
    static OUTSTANDING: AtomicUsize = AtomicUsize::new(0);

    /// Thread count set by [`init_thread_pool`]; `0` means the pool has not been initialised yet.
    static POOL_THREAD_COUNT: AtomicUsize = AtomicUsize::new(0);

    lazy_static::lazy_static! {
        /// Queue limit from environment. `None` means no limit.
        static ref QUEUE_LIMIT: Option<usize> = {
            let limit = std::env::var("HOPR_CPU_TASK_QUEUE_LIMIT")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&v| v > 0);

            if let Some(l) = limit {
                metrics::set_queue_limit(l);
            }

            limit
        };
    }

    /// Error type for spawn operations.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
    pub enum SpawnError {
        /// The queue is full and cannot accept more tasks.
        #[error("rayon queue full: {current}/{limit} tasks outstanding")]
        QueueFull {
            /// Current outstanding task count when rejection occurred.
            current: usize,
            /// Configured queue limit.
            limit: usize,
        },
    }

    /// Returns the current outstanding task count (queued + running).
    #[inline]
    pub fn outstanding_tasks() -> usize {
        OUTSTANDING.load(Ordering::Relaxed)
    }

    /// Returns the configured queue limit, or `None` if unlimited.
    #[inline]
    pub fn queue_limit() -> Option<usize> {
        *QUEUE_LIMIT
    }

    /// Guard that acquires a slot on construction and calls releases slot on drop,
    /// even if the task panics or returns early.
    struct SlotGuard;

    impl SlotGuard {
        /// Attempts to acquire a slot for a new task.
        ///
        /// Returns `Ok(())` if no limit or slot acquired, `Err(QueueFull)` if at limit.
        pub fn try_acquire_slot() -> Result<Self, SpawnError> {
            let prev = OUTSTANDING.fetch_add(1, Ordering::AcqRel);
            metrics::outstanding_inc();
            let guard = Self;

            if let Some(limit) = *QUEUE_LIMIT {
                let new = prev + 1;
                if new > limit {
                    metrics::rejected();
                    return Err(SpawnError::QueueFull { current: prev, limit });
                }
            }
            Ok(guard)
        }
    }

    impl Drop for SlotGuard {
        #[inline]
        fn drop(&mut self) {
            let prev = OUTSTANDING.fetch_sub(1, Ordering::AcqRel);
            debug_assert!(prev > 0, "outstanding task count underflow");
            metrics::outstanding_dec();
        }
    }

    /// Initialize the Rayon thread pool with the given number of threads.
    ///
    /// Also initializes the queue limit metric.
    pub fn init_thread_pool(num_threads: usize) -> Result<(), rayon::ThreadPoolBuildError> {
        let builder = rayon::ThreadPoolBuilder::new().num_threads(num_threads);

        let builder = builder.spawn_handler(|thread| {
            let mut thread_builder = std::thread::Builder::new();
            if let Some(name) = thread.name() {
                thread_builder = thread_builder.name(name.to_owned());
            }
            if let Some(stack_size) = thread.stack_size() {
                thread_builder = thread_builder.stack_size(stack_size);
            }
            thread_builder.spawn(|| {
                #[cfg(target_os = "macos")]
                unsafe {
                    // MacOS: Set the QOS class to "user initiated" to allow running on performance cores
                    libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INITIATED, 0);
                }
                thread.run()
            })?;
            Ok(())
        });

        // Store the requested thread count before build_global() so that pipeline code that
        // calls pool_thread_count() after init can read a non-zero value immediately.
        POOL_THREAD_COUNT.store(num_threads, Ordering::Relaxed);
        let result = builder.build_global();
        let _ = *QUEUE_LIMIT; // Initialize limit metric
        result
    }

    /// Returns the thread count the pool was initialised with, or `0` if [`init_thread_pool`]
    /// has not been called yet.
    #[inline]
    pub fn pool_thread_count() -> usize {
        POOL_THREAD_COUNT.load(Ordering::Relaxed)
    }

    /// Outstanding tasks currently attributed to the **encode** path (packet_encode + SURB generation).
    pub static ENCODE_OUTSTANDING: AtomicUsize = AtomicUsize::new(0);

    /// Outstanding tasks currently attributed to the **decode** path (packet_decode).
    pub static DECODE_OUTSTANDING: AtomicUsize = AtomicUsize::new(0);

    /// Cumulative count of packets dropped because the Rayon decode future timed out.
    ///
    /// Incremented unconditionally (not gated on the `telemetry` feature) so the
    /// stress harness can read it in test builds.
    pub static DECODE_TIMEOUT_DROPS: AtomicUsize = AtomicUsize::new(0);

    /// Cumulative count of outgoing packets (data or SURB) dropped because the
    /// Rayon encode future timed out (150 ms budget exceeded).
    ///
    /// A non-zero and rising count indicates the encode path is saturating the pool.
    pub static ENCODE_TIMEOUT_DROPS: AtomicUsize = AtomicUsize::new(0);

    /// Returns the current encode-path outstanding task count.
    #[inline]
    pub fn encode_outstanding_tasks() -> usize {
        ENCODE_OUTSTANDING.load(Ordering::Relaxed)
    }

    /// Returns the current decode-path outstanding task count.
    #[inline]
    pub fn decode_outstanding_tasks() -> usize {
        DECODE_OUTSTANDING.load(Ordering::Relaxed)
    }

    /// Returns the cumulative decode timeout drop count.
    #[inline]
    pub fn decode_timeout_drop_count() -> usize {
        DECODE_TIMEOUT_DROPS.load(Ordering::Relaxed)
    }

    /// Returns the cumulative encode timeout drop count.
    #[inline]
    pub fn encode_timeout_drop_count() -> usize {
        ENCODE_TIMEOUT_DROPS.load(Ordering::Relaxed)
    }

    // ───────────────────────── encode/decode pool arbitration ─────────────────────────
    //
    // The encode (SPHINX wrap + SURB generation) and decode (SPHINX peel) paths share the single
    // Rayon pool. Left unmanaged, a heavy download floods decode (forwarding) and starves SURB
    // encode → the SURB ring drains → download collapses (#8246).
    //
    // The arbiter is deliberately **asymmetric and occupancy-gated** so it never harms the common
    // case (measured: a symmetric cap that always engaged halved forwarding):
    //   * Only DECODE is ever throttled; ENCODE never blocks (it is the light, latency-critical,
    //     protected class — SURB generation).
    //   * Decode is throttled ONLY when all of: the pool is genuinely saturated (running threads ≥
    //     `occupancy_pct`), AND encode work is actually present, AND decode already exceeds its share.
    //   * A pure forwarding relay (encode ≈ 0) is therefore never throttled; nor is any node whose
    //     pool is below the occupancy threshold — so the arbiter adds ~zero overhead until it is
    //     genuinely needed.
    // Occupancy is measured by [`RUNNING`] (tasks actually executing on a pool thread), not the
    // queued+running `*_OUTSTANDING` counters, which the deep pipeline ready-queues keep saturated.

    /// When `false`, [`spawn_decode_blocking`] skips admission entirely.
    static ARBITRATION_ENABLED: AtomicBool = AtomicBool::new(true);
    /// Pool occupancy (percent of [`RUNNING`] threads over pool size) at or above which decode
    /// admission may engage. Below it, decode is never throttled.
    static ARBITRATION_OCCUPANCY_PCT: AtomicU32 = AtomicU32::new(75);
    /// Fraction (percent) of the pool that decode yields to encode when both contend under saturation.
    static ARBITRATION_ENCODE_RESERVE_PCT: AtomicU32 = AtomicU32::new(50);
    /// Tasks currently executing on a pool thread (true occupancy, `0..=pool_thread_count`).
    static RUNNING: AtomicUsize = AtomicUsize::new(0);
    /// Notified when a running task finishes (a pool slot frees) so a blocked decode admitter retries.
    static ARBITRATION_EVENT: event_listener::Event = event_listener::Event::new();

    /// Returns the number of tasks currently executing on a pool thread.
    #[inline]
    pub fn running_tasks() -> usize {
        RUNNING.load(Ordering::Relaxed)
    }

    /// Configure encode/decode pool arbitration. Call once at startup, next to [`init_thread_pool`].
    ///
    /// `enabled` gates the whole mechanism. `occupancy_pct` is the pool-occupancy threshold below
    /// which decode is never throttled. `encode_reserve_pct` is the share of the pool decode yields
    /// to encode when both contend under saturation. Both percentages are clamped to `1..=100`.
    pub fn configure_arbitration(enabled: bool, occupancy_pct: u32, encode_reserve_pct: u32) {
        ARBITRATION_ENABLED.store(enabled, Ordering::Relaxed);
        ARBITRATION_OCCUPANCY_PCT.store(occupancy_pct.clamp(1, 100), Ordering::Relaxed);
        ARBITRATION_ENCODE_RESERVE_PCT.store(encode_reserve_pct.clamp(1, 100), Ordering::Relaxed);
    }

    /// Awaits until the decode path may submit without starving encode. Passthrough when arbitration
    /// is disabled or the pool is uninitialised; see the module policy note above.
    async fn admit_decode() {
        if !ARBITRATION_ENABLED.load(Ordering::Relaxed) {
            return;
        }
        let pool_threads = pool_thread_count();
        if pool_threads == 0 {
            return;
        }
        let occupancy_pct = ARBITRATION_OCCUPANCY_PCT.load(Ordering::Relaxed);
        let reserve_pct = ARBITRATION_ENCODE_RESERVE_PCT.load(Ordering::Relaxed);
        loop {
            match decode_admit_cap(
                pool_threads,
                RUNNING.load(Ordering::Relaxed),
                ENCODE_OUTSTANDING.load(Ordering::Relaxed),
                occupancy_pct,
                reserve_pct,
            ) {
                // Unconstrained (pool not saturated, or no encode to protect) → submit immediately.
                None => return,
                Some(cap) if DECODE_OUTSTANDING.load(Ordering::Relaxed) < cap => return,
                Some(_) => {
                    // Register before re-checking so a slot freed in between is not missed.
                    let listener = ARBITRATION_EVENT.listen();
                    match decode_admit_cap(
                        pool_threads,
                        RUNNING.load(Ordering::Relaxed),
                        ENCODE_OUTSTANDING.load(Ordering::Relaxed),
                        occupancy_pct,
                        reserve_pct,
                    ) {
                        None => return,
                        Some(cap) if DECODE_OUTSTANDING.load(Ordering::Relaxed) < cap => return,
                        Some(_) => listener.await,
                    }
                }
            }
        }
    }

    /// The decode-admission policy, as a pure function so it is trivially testable.
    ///
    /// Returns `None` when decode is unconstrained — either the pool is below the occupancy threshold
    /// (`running < occupancy_pct% of pool`) or there is no encode work to protect (`enc == 0`), which
    /// is exactly the pure-forwarding-relay case. Otherwise returns `Some(cap)`, the maximum decode
    /// tasks allowed outstanding so that up to `min(enc, encode_reserve_pct% of pool)` threads are
    /// left for encode; floored at 1 for liveness.
    fn decode_admit_cap(
        pool_threads: usize,
        running: usize,
        enc_outstanding: usize,
        occupancy_pct: u32,
        encode_reserve_pct: u32,
    ) -> Option<usize> {
        let occupancy_threshold = (pool_threads as u64 * occupancy_pct as u64 / 100) as usize;
        if running < occupancy_threshold {
            return None; // pool not saturated → never throttle
        }
        if enc_outstanding == 0 {
            return None; // nothing to protect (pure forwarding relay) → never throttle
        }
        let reserve = ((pool_threads as u64 * encode_reserve_pct as u64).div_ceil(100) as usize).min(enc_outstanding);
        Some(pool_threads.saturating_sub(reserve).max(1))
    }

    /// RAII guard that decrements a tagged outstanding counter when dropped.
    ///
    /// Caller must increment the counter before constructing this guard.
    struct TaggedGuard(&'static AtomicUsize);

    impl Drop for TaggedGuard {
        #[inline]
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// RAII guard tracking a task that is actually executing on a pool thread. Increments [`RUNNING`]
    /// on construction; on drop decrements it and wakes one blocked decode admitter (a slot freed).
    struct RunningGuard;

    impl RunningGuard {
        #[inline]
        fn enter() -> Self {
            RUNNING.fetch_add(1, Ordering::Relaxed);
            Self
        }
    }

    impl Drop for RunningGuard {
        #[inline]
        fn drop(&mut self) {
            RUNNING.fetch_sub(1, Ordering::Relaxed);
            ARBITRATION_EVENT.notify(1);
        }
    }

    /// Like [`spawn_fifo_blocking`] but also tracks the task in [`ENCODE_OUTSTANDING`].
    ///
    /// Encode is the protected class: it is never throttled by arbitration.
    pub async fn spawn_encode_blocking<R: Send + 'static>(
        f: impl FnOnce() -> R + Send + 'static,
        operation: &'static str,
    ) -> Result<R, SpawnError> {
        ENCODE_OUTSTANDING.fetch_add(1, Ordering::Relaxed);
        let _guard = TaggedGuard(&ENCODE_OUTSTANDING);
        spawn_fifo_blocking(f, operation).await
    }

    /// Like [`spawn_fifo_blocking`] but also tracks the task in [`DECODE_OUTSTANDING`] and yields a
    /// fair-share pool slot to encode first when the pool is saturated (see [`admit_decode`]).
    pub async fn spawn_decode_blocking<R: Send + 'static>(
        f: impl FnOnce() -> R + Send + 'static,
        operation: &'static str,
    ) -> Result<R, SpawnError> {
        admit_decode().await;
        DECODE_OUTSTANDING.fetch_add(1, Ordering::Relaxed);
        let _guard = TaggedGuard(&DECODE_OUTSTANDING);
        spawn_fifo_blocking(f, operation).await
    }

    /// Builds a cancellable task closure and its receiver.
    ///
    /// The closure wraps `f` with cooperative cancellation, panic catching,
    /// timing metrics, and slot tracking via guard.
    ///
    /// Note: Cooperative cancellation only prevents "queued zombies" - tasks whose
    /// receiver was dropped before execution started. If the timeout fires *after*
    /// execution begins, the task will still run to completion (counted as "orphaned").
    fn cancellable_task<R: Send + 'static>(
        f: impl FnOnce() -> R + Send + 'static,
        operation: &'static str,
    ) -> Result<
        (
            impl FnOnce() + Send + 'static,
            oneshot::Receiver<std::thread::Result<R>>,
        ),
        SpawnError,
    > {
        let guard = SlotGuard::try_acquire_slot()?;

        let (tx, rx) = oneshot::channel();
        let submitted_at = std::time::Instant::now();

        metrics::submitted();

        let task = move || {
            // ensures guard is moved inside the closure, and
            // that the slot is released even on panic
            let _g = guard;

            if tx.is_canceled() {
                tracing::debug!(
                    queue_wait_ms = submitted_at.elapsed().as_millis() as u64,
                    "skipping cancelled task (receiver dropped before execution)"
                );
                metrics::cancelled();
                return;
            }

            let wait_duration = submitted_at.elapsed();
            metrics::observe_queue_wait(wait_duration.as_secs_f64());

            // Mark this task as occupying a pool thread for the duration of `f` (drops after it).
            let _running = RunningGuard::enter();
            let execution_start = std::time::Instant::now();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
            metrics::observe_execution(operation, execution_start.elapsed().as_secs_f64());

            match tx.send(result) {
                Ok(()) => metrics::completed(),
                Err(_) => {
                    tracing::debug!(
                        queue_wait_ms = wait_duration.as_millis() as u64,
                        "receiver dropped during execution, result discarded"
                    );
                    metrics::orphaned();
                }
            }
        };

        Ok((task, rx))
    }

    /// Spawn a blocking function on the Rayon thread pool (LIFO scheduling).
    ///
    /// Uses Rayon's default LIFO scheduling for the thread's local queue.
    ///
    /// Includes cooperative cancellation: if the receiver is dropped before the
    /// task starts (e.g., timeout), the task is skipped without executing.
    ///
    /// # Errors
    ///
    /// Returns [`SpawnError::QueueFull`] if the outstanding task count exceeds the limit.
    pub async fn spawn_blocking<R: Send + 'static>(
        f: impl FnOnce() -> R + Send + 'static,
        operation: &'static str,
    ) -> Result<R, SpawnError> {
        let (task, rx) = cancellable_task(f, operation)?;
        rayon::spawn(task);
        Ok(rx
            .await
            .expect("rayon task channel closed unexpectedly")
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic)))
    }

    /// Spawn a blocking function on the Rayon thread pool (FIFO scheduling).
    ///
    /// Uses FIFO scheduling which prevents starvation of older tasks. This is the
    /// preferred variant for packet decoding and similar ordered workloads.
    ///
    /// Includes cooperative cancellation: if the receiver is dropped before the
    /// task starts (e.g., timeout), the task is skipped without executing.
    ///
    /// # Errors
    ///
    /// Returns [`SpawnError::QueueFull`] if the outstanding task count exceeds the limit.
    pub async fn spawn_fifo_blocking<R: Send + 'static>(
        f: impl FnOnce() -> R + Send + 'static,
        operation: &'static str,
    ) -> Result<R, SpawnError> {
        let (task, rx) = cancellable_task(f, operation)?;
        rayon::spawn_fifo(task);
        Ok(rx
            .await
            .expect("rayon task channel closed unexpectedly")
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic)))
    }

    #[cfg(test)]
    mod arbitration_tests {
        use super::{admit_decode, configure_arbitration, decode_admit_cap};

        const N: usize = 8;
        const OCC: u32 = 75; // occupancy threshold percent
        const RES: u32 = 50; // encode reserve percent

        #[test]
        fn below_occupancy_threshold_decode_is_never_throttled() {
            // 75% of 8 == 6 running threads; anything below leaves decode unconstrained.
            for running in 0..(N * OCC as usize / 100) {
                assert_eq!(
                    decode_admit_cap(N, running, 100, OCC, RES),
                    None,
                    "decode must be free below the occupancy threshold (running={running})",
                );
            }
        }

        #[test]
        fn saturated_but_no_encode_is_never_throttled() {
            // Pure forwarding relay: pool full of decode, zero encode → nothing to protect.
            assert_eq!(decode_admit_cap(N, N, 0, OCC, RES), None);
        }

        #[test]
        fn saturated_with_encode_reserves_threads_for_encode() {
            // Full pool + plenty of encode demand → decode capped so 50% is left for encode.
            assert_eq!(decode_admit_cap(N, N, N, OCC, RES), Some(N / 2));
            // Reserve never exceeds actual encode demand: only 1 encode task → reserve just 1 thread.
            assert_eq!(decode_admit_cap(N, N, 1, OCC, RES), Some(N - 1));
        }

        #[test]
        fn cap_is_never_zero_liveness() {
            // A full encode reserve on a tiny pool still admits at least one decode task.
            assert_eq!(decode_admit_cap(2, 2, 10, OCC, 100), Some(1));
            for running in 0..=N {
                for enc in 0..=N {
                    if let Some(cap) = decode_admit_cap(N, running, enc, OCC, RES) {
                        assert!(cap >= 1, "cap must never be zero (running={running}, enc={enc})");
                    }
                }
            }
        }

        /// Disabled arbitration and an uninitialised pool are both pure passthrough (never block).
        #[tokio::test]
        async fn admit_decode_is_passthrough_when_disabled_or_pool_uninitialised() {
            // Pool is uninitialised in unit tests (pool_thread_count() == 0) → immediate return.
            tokio::time::timeout(std::time::Duration::from_secs(5), admit_decode())
                .await
                .expect("admit_decode must not block with an uninitialised pool");

            configure_arbitration(false, OCC, RES);
            tokio::time::timeout(std::time::Duration::from_secs(5), admit_decode())
                .await
                .expect("admit_decode must not block when disabled");
            configure_arbitration(true, OCC, RES); // restore default for other tests
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{
            Arc,
            atomic::{AtomicU32, Ordering},
        },
        time::Duration,
    };

    use futures::FutureExt;
    use serial_test::serial;

    use super::cpu;

    #[tokio::test]
    #[serial]
    async fn spawn_blocking_returns_result() {
        let result = cpu::spawn_blocking(|| 42, "test").await.unwrap();
        assert_eq!(result, 42);
    }

    #[tokio::test]
    #[serial]
    async fn spawn_fifo_blocking_returns_result() {
        let result = cpu::spawn_fifo_blocking(|| "hello", "test").await.unwrap();
        assert_eq!(result, "hello");
    }

    #[cfg(panic = "unwind")]
    #[tokio::test]
    #[serial]
    async fn spawn_blocking_propagates_panic() {
        let result = std::panic::AssertUnwindSafe(async {
            cpu::spawn_blocking(
                || {
                    panic!("test panic");
                },
                "test",
            )
            .await
            .unwrap()
        })
        .catch_unwind()
        .await;
        assert!(result.is_err(), "should propagate panic from Rayon task");
    }

    #[tokio::test]
    #[serial]
    async fn cancelled_tasks_are_skipped_via_cooperative_cancellation() {
        let initial_outstanding = cpu::outstanding_tasks();
        let executed_count = Arc::new(AtomicU32::new(0));

        for _ in 0..100 {
            let count = executed_count.clone();
            let fut = cpu::spawn_fifo_blocking(
                move || {
                    count.fetch_add(1, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(50));
                },
                "test",
            );
            let _ = fut.now_or_never();
        }

        let start = std::time::Instant::now();
        let result = cpu::spawn_fifo_blocking(|| 42, "test").await.unwrap();
        let elapsed = start.elapsed();

        assert_eq!(result, 42);
        assert!(
            elapsed < Duration::from_secs(2),
            "Task took {elapsed:?} - cancelled tasks may not be getting skipped"
        );

        for _ in 0..50 {
            tokio::time::sleep(Duration::from_millis(100)).await;
            if cpu::outstanding_tasks() == initial_outstanding {
                break;
            }
        }

        let executed = executed_count.load(Ordering::SeqCst);
        assert!(
            executed < 50,
            "Expected most tasks to be skipped by cancellation, but {executed}/100 executed"
        );
    }

    #[tokio::test]
    #[serial]
    async fn pool_recovers_after_cancelled_burst() {
        let initial_outstanding = cpu::outstanding_tasks();

        for _ in 0..50 {
            let fut = cpu::spawn_fifo_blocking(
                || {
                    std::thread::sleep(Duration::from_millis(100));
                },
                "test",
            );
            let _ = fut.now_or_never();
        }

        tokio::time::sleep(Duration::from_millis(300)).await;

        for i in 0..10 {
            let start = std::time::Instant::now();
            let result = cpu::spawn_fifo_blocking(move || i * 2, "test").await.unwrap();
            let elapsed = start.elapsed();

            assert_eq!(result, i * 2);
            assert!(
                elapsed < Duration::from_millis(500),
                "Recovery task {i} took {elapsed:?} - pool may still be starved"
            );
        }

        for _ in 0..50 {
            tokio::time::sleep(Duration::from_millis(100)).await;
            if cpu::outstanding_tasks() == initial_outstanding {
                break;
            }
        }
    }

    #[tokio::test]
    #[serial]
    async fn outstanding_tasks_tracking() {
        let initial = cpu::outstanding_tasks();

        let barrier = Arc::new(std::sync::Barrier::new(2));
        let barrier_clone = barrier.clone();

        let handle = tokio::spawn(async move {
            cpu::spawn_fifo_blocking(
                move || {
                    barrier_clone.wait();
                    42
                },
                "test",
            )
            .await
        });

        tokio::time::sleep(Duration::from_millis(50)).await;

        let during = cpu::outstanding_tasks();
        assert!(
            during > initial,
            "Outstanding should increase: initial={initial}, during={during}"
        );

        barrier.wait();

        let result = handle.await.unwrap();
        assert_eq!(result.unwrap(), 42);

        tokio::time::sleep(Duration::from_millis(50)).await;

        let after = cpu::outstanding_tasks();
        assert_eq!(after, initial, "Outstanding should return to initial after completion");
    }

    #[tokio::test]
    #[serial]
    async fn outstanding_decrements_on_cancellation() {
        let initial = cpu::outstanding_tasks();

        for _ in 0..10 {
            let fut = cpu::spawn_fifo_blocking(
                || {
                    std::thread::sleep(Duration::from_millis(100));
                },
                "test",
            );
            let _ = fut.now_or_never();
        }

        tokio::time::sleep(Duration::from_millis(500)).await;

        let after = cpu::outstanding_tasks();
        assert_eq!(
            after, initial,
            "Outstanding should return to initial after cancelled tasks drain"
        );
    }

    #[tokio::test]
    #[serial]
    async fn tagged_encode_decode_spawns_track_and_release() {
        let enc0 = cpu::encode_outstanding_tasks();
        let dec0 = cpu::decode_outstanding_tasks();

        assert_eq!(cpu::spawn_encode_blocking(|| 1, "test").await.unwrap(), 1);
        assert_eq!(cpu::spawn_decode_blocking(|| 2, "test").await.unwrap(), 2);

        // Counters return to their starting values once the tagged guards drop.
        assert_eq!(cpu::encode_outstanding_tasks(), enc0);
        assert_eq!(cpu::decode_outstanding_tasks(), dec0);

        // Timeout-drop counters and the pool accessors are reachable.
        let _ = cpu::encode_timeout_drop_count();
        let _ = cpu::decode_timeout_drop_count();
        let _ = cpu::pool_thread_count();
        let _ = super::encode_pool_has_headroom();
    }
}
