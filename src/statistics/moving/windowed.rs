//! Ratio of observed to expected events over a sliding time window.

use std::time::{Duration, Instant};

/// One time slice of the ring.
///
/// `stamp` is the absolute slice number this slot currently holds, which is what makes reuse
/// detectable without a separate clear pass: a slot whose stamp is older than the window is stale,
/// and its counts are discarded the moment it is claimed.
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
struct Bucket {
    stamp: u64,
    expected: u64,
    observed: u64,
}

/// Fraction of expected events that were actually observed, over a sliding window.
///
/// Answers "how much of what I set in motion came back, lately". Both counters only ever increase
/// within a slice, and whole slices age out — so unlike a decaying average the value can climb
/// again the moment things recover, and unlike a cumulative ratio it is not anchored by ancient
/// history. Idle reads as *no data* rather than as failure, so something merely unused is never
/// mistaken for something broken.
///
/// # Bucketing
///
/// Time is divided into fixed slices; the ring holds the most recent `bucket_count`. A read sums
/// the live slices. Expectations and observations are counted in whichever slice they *occur* in,
/// which is not necessarily the same one — so the ratio is only meaningful when the window is
/// comfortably longer than the delay between an expectation and its observation. Where it is not,
/// a slice can briefly report more observations than expectations, hence the clamp.
///
/// # Storage
///
/// A fixed-size array of plain counters, so the whole type stays `Copy` and can live inside another
/// `Copy` observation without forcing heap allocation or interior mutability on it. Callers that
/// need to accumulate from several threads should do so in their own lock-free counters and fold
/// the totals in periodically — recording here is a `&mut` operation, cheap but not concurrent.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct WindowedRatio<const BUCKETS: usize> {
    buckets: [Bucket; BUCKETS],
    bucket_width: Duration,
    epoch: Instant,
}

impl<const BUCKETS: usize> WindowedRatio<BUCKETS> {
    /// Creates a ratio over `BUCKETS` slices of `bucket_width`, i.e. a window of their product.
    ///
    /// More, narrower slices track change more finely at the cost of memory; fewer, wider ones
    /// smooth more. The width is clamped away from zero, which would divide by zero when locating
    /// a slice.
    pub fn new(bucket_width: Duration, epoch: Instant) -> Self {
        Self {
            buckets: [Bucket::default(); BUCKETS],
            bucket_width: bucket_width.max(Duration::from_millis(1)),
            epoch,
        }
    }

    /// Records that `count` events are expected to be observed later.
    pub fn record_expected(&mut self, count: u64, now: Instant) {
        self.bucket_at(now).expected += count;
    }

    /// Records that `count` expected events were observed.
    pub fn record_observed(&mut self, count: u64, now: Instant) {
        self.bucket_at(now).observed += count;
    }

    /// Observed / expected across the live window, or `None` when nothing was expected in it.
    ///
    /// `None` is not zero: it means the window holds no evidence either way, which callers must
    /// treat as neutral rather than as a failing peer.
    pub fn value(&self, now: Instant) -> Option<f64> {
        let current = self.absolute_bucket(now);
        let oldest = current.saturating_sub(BUCKETS as u64 - 1);

        let (expected, observed) = self
            .buckets
            .iter()
            .filter(|b| b.stamp >= oldest && b.stamp <= current)
            .fold((0u64, 0u64), |(e, o), b| (e + b.expected, o + b.observed));

        (expected > 0).then(|| (observed as f64 / expected as f64).clamp(0.0, 1.0))
    }

    /// Total span covered by the window.
    pub fn window(&self) -> Duration {
        self.bucket_width * BUCKETS as u32
    }

    fn absolute_bucket(&self, now: Instant) -> u64 {
        (now.saturating_duration_since(self.epoch).as_nanos() / self.bucket_width.as_nanos()) as u64
    }

    /// The slot for `now`, cleared first if it still holds an older slice.
    fn bucket_at(&mut self, now: Instant) -> &mut Bucket {
        let absolute = self.absolute_bucket(now);
        let bucket = &mut self.buckets[(absolute % BUCKETS as u64) as usize];

        if bucket.stamp != absolute {
            // Reused for a new slice, so the previous slice's counts go with it.
            *bucket = Bucket {
                stamp: absolute,
                expected: 0,
                observed: 0,
            };
        }

        bucket
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 4 slices of 1 s — a 4 s window, small enough to reason about slice by slice.
    fn ratio(epoch: Instant) -> WindowedRatio<4> {
        WindowedRatio::new(Duration::from_secs(1), epoch)
    }

    fn at(epoch: Instant, secs: u64) -> Instant {
        epoch + Duration::from_secs(secs)
    }

    #[test]
    fn value_should_be_none_until_something_is_expected() {
        let epoch = Instant::now();
        let mut r = ratio(epoch);

        assert_eq!(None, r.value(epoch), "an empty window holds no evidence");

        // An observation with nothing expected still leaves the denominator at zero.
        r.record_observed(1, epoch);
        assert_eq!(None, r.value(epoch));
    }

    #[test]
    fn value_should_be_one_when_everything_expected_is_observed() {
        let epoch = Instant::now();
        let mut r = ratio(epoch);

        for _ in 0..10 {
            r.record_expected(1, epoch);
            r.record_observed(1, epoch);
        }

        assert_eq!(Some(1.0), r.value(epoch));
    }

    #[test]
    fn value_should_be_zero_when_nothing_expected_is_observed() {
        let epoch = Instant::now();
        let mut r = ratio(epoch);

        for _ in 0..10 {
            r.record_expected(1, epoch);
        }

        assert_eq!(Some(0.0), r.value(epoch));
    }

    #[test]
    fn value_should_recover_upward_once_healthy_slices_enter() {
        let epoch = Instant::now();
        let mut r = ratio(epoch);

        // Two bad seconds: everything expected, nothing observed.
        for sec in 0..2 {
            for _ in 0..10 {
                r.record_expected(1, at(epoch, sec));
            }
        }
        assert_eq!(Some(0.0), r.value(at(epoch, 1)));

        // Then two good ones. This is the property a decay-to-zero cannot provide.
        for sec in 2..4 {
            for _ in 0..10 {
                r.record_expected(1, at(epoch, sec));
                r.record_observed(1, at(epoch, sec));
            }
        }
        assert_eq!(Some(0.5), r.value(at(epoch, 3)), "half the window is now healthy");

        // Once the bad slices age out entirely, the value is back to full health.
        for sec in 4..8 {
            for _ in 0..10 {
                r.record_expected(1, at(epoch, sec));
                r.record_observed(1, at(epoch, sec));
            }
        }
        assert_eq!(Some(1.0), r.value(at(epoch, 7)));
    }

    #[test]
    fn stale_slices_should_age_out_of_the_window() {
        let epoch = Instant::now();
        let mut r = ratio(epoch);

        for _ in 0..10 {
            r.record_expected(1, epoch);
        }
        assert_eq!(Some(0.0), r.value(epoch));

        // A full window later, nothing from that slice counts — and with no newer evidence the
        // window is empty rather than bad.
        assert_eq!(None, r.value(at(epoch, 10)));
    }

    #[test]
    fn value_should_be_clamped_when_observations_outrun_expectations() {
        let epoch = Instant::now();
        let mut r = ratio(epoch);

        // An expectation raised just before the window edge, observed after it aged out: the
        // observation is real but its expectation is gone, so the raw ratio would exceed 1.
        r.record_expected(1, epoch);
        r.record_expected(1, at(epoch, 5));
        r.record_observed(2, at(epoch, 5));

        assert_eq!(Some(1.0), r.value(at(epoch, 5)));
    }

    #[test]
    fn window_should_be_width_times_count() {
        let epoch = Instant::now();
        assert_eq!(Duration::from_secs(4), ratio(epoch).window());
    }

    #[test]
    fn should_be_copy_so_it_can_live_inside_a_copy_observation() {
        // Load-bearing: the graph edge weight that holds this is `Copy` and returned by value, so
        // an allocation or interior mutability here would ripple out into every consumer.
        let epoch = Instant::now();
        let mut r = ratio(epoch);
        r.record_expected(4, epoch);
        r.record_observed(2, epoch);

        let snapshot = r;
        r.record_observed(2, epoch);

        assert_eq!(Some(0.5), snapshot.value(epoch), "the copy must not see later writes");
        assert_eq!(Some(1.0), r.value(epoch));
    }

    #[test]
    fn a_zero_width_slice_should_be_clamped_rather_than_divide_by_zero() {
        let epoch = Instant::now();
        let mut r: WindowedRatio<4> = WindowedRatio::new(Duration::ZERO, epoch);

        r.record_expected(1, epoch);
        r.record_observed(1, epoch);
        assert_eq!(Some(1.0), r.value(epoch));
    }

}
