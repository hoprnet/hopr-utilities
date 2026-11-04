//! Ratio of observed to expected events over a sliding time window.

use std::{
    cmp::Ordering,
    time::{Duration, Instant},
};

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
    ///
    /// `BUCKETS` must be at least one; a ring with no slices has nowhere to record into and is
    /// rejected at compile time, rather than dividing by zero at the first write.
    pub fn new(bucket_width: Duration, epoch: Instant) -> Self {
        const { assert!(BUCKETS > 0, "WindowedRatio needs at least one bucket") }

        Self {
            buckets: [Bucket::default(); BUCKETS],
            bucket_width: bucket_width.max(Duration::from_millis(1)),
            epoch,
        }
    }

    /// Records that `count` events are expected to be observed later.
    ///
    /// A record whose slice has already been overwritten by a newer one is dropped; see
    /// [`Self::bucket_at`].
    pub fn record_expected(&mut self, count: u64, now: Instant) {
        if let Some(bucket) = self.bucket_at(now) {
            bucket.expected = bucket.expected.saturating_add(count);
        }
    }

    /// Records that `count` expected events were observed.
    ///
    /// A record whose slice has already been overwritten by a newer one is dropped; see
    /// [`Self::bucket_at`].
    pub fn record_observed(&mut self, count: u64, now: Instant) {
        if let Some(bucket) = self.bucket_at(now) {
            bucket.observed = bucket.observed.saturating_add(count);
        }
    }

    /// Observed / expected across the live window, or `None` when nothing was expected in it.
    ///
    /// `None` is not zero: it means the window holds no evidence either way, which callers must
    /// treat as neutral rather than as a failing peer.
    pub fn value(&self, now: Instant) -> Option<f64> {
        self.recent_value(BUCKETS, now)
    }

    /// Observed / expected across the most recent `slices` only, or `None` when they hold nothing.
    ///
    /// The full-window [`Self::value`] dilutes a sudden change by the history still in the ring: a
    /// path that stops delivering right now is one bad slice against `BUCKETS - 1` healthy ones.
    /// Reading the newest slices alone is what makes a collapse visible while it is still recent,
    /// at the cost of resting on less evidence -- so it is meant to be compared *against* the full
    /// window rather than used as a standalone verdict.
    ///
    /// `slices` is clamped to `1..=BUCKETS`, so a caller cannot ask for a window wider than the
    /// ring or narrower than one slice.
    pub fn recent_value(&self, slices: usize, now: Instant) -> Option<f64> {
        let slices = slices.clamp(1, BUCKETS) as u64;
        let current = self.absolute_bucket(now);
        let oldest = current.saturating_sub(slices - 1);

        // Saturating, because a wrapped total would not merely be imprecise: it would invert the
        // ratio and read as a collapse.
        let (expected, observed) = self
            .buckets
            .iter()
            .filter(|b| b.stamp >= oldest && b.stamp <= current)
            .fold((0u64, 0u64), |(e, o), b| {
                (e.saturating_add(b.expected), o.saturating_add(b.observed))
            });

        (expected > 0).then(|| (observed as f64 / expected as f64).clamp(0.0, 1.0))
    }

    /// Total span covered by the window.
    ///
    /// Saturates at [`Duration::MAX`] rather than panicking, so an absurdly wide slice degrades to
    /// "effectively forever" instead of taking the caller down.
    pub fn window(&self) -> Duration {
        self.bucket_width
            .saturating_mul(u32::try_from(BUCKETS).unwrap_or(u32::MAX))
    }

    fn absolute_bucket(&self, now: Instant) -> u64 {
        (now.saturating_duration_since(self.epoch).as_nanos() / self.bucket_width.as_nanos()) as u64
    }

    /// The slot for `now`, cleared first if it still holds an older slice, or `None` if it already
    /// holds a newer one.
    ///
    /// Every `BUCKETS`-th slice shares a slot, so a record that arrives out of order -- with a
    /// `now` behind one already recorded -- can land on a slot belonging to a newer slice. Clearing
    /// it would trade live evidence for a slice that has since aged out of the window, so the late
    /// record is dropped instead.
    fn bucket_at(&mut self, now: Instant) -> Option<&mut Bucket> {
        let absolute = self.absolute_bucket(now);
        let bucket = &mut self.buckets[(absolute % BUCKETS as u64) as usize];

        match absolute.cmp(&bucket.stamp) {
            Ordering::Less => None,
            Ordering::Equal => Some(bucket),
            Ordering::Greater => {
                // Reused for a newer slice, so the previous slice's counts go with it.
                *bucket = Bucket {
                    stamp: absolute,
                    expected: 0,
                    observed: 0,
                };
                Some(bucket)
            }
        }
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

    /// The whole point: a collapse that the full window dilutes must be visible in the last slices.
    #[test]
    fn recent_value_should_see_a_collapse_the_full_window_still_dilutes() {
        let epoch = Instant::now();
        let mut r = ratio(epoch);

        // Three healthy slices, then one where nothing comes back.
        for sec in 0..3 {
            r.record_expected(100, at(epoch, sec));
            r.record_observed(100, at(epoch, sec));
        }
        r.record_expected(100, at(epoch, 3));

        let now = at(epoch, 3);
        let full = r.value(now).expect("window holds evidence");
        let recent = r.recent_value(1, now).expect("newest slice holds evidence");

        // 300/400 against 0/100 -- the full window is still mostly healthy history.
        assert_eq!(recent, 0.0, "the newest slice saw nothing come back");
        assert!(
            full > 0.5,
            "the full window should still be dominated by healthy history, got {full}"
        );
    }

    #[test]
    fn recent_value_should_match_the_full_window_when_asked_for_every_slice() {
        let epoch = Instant::now();
        let mut r = ratio(epoch);
        for sec in 0..4 {
            r.record_expected(10, at(epoch, sec));
            r.record_observed(5, at(epoch, sec));
        }

        let now = at(epoch, 3);
        assert_eq!(r.recent_value(4, now), r.value(now));
        // Clamped, so over-asking is the same as asking for the whole ring.
        assert_eq!(r.recent_value(999, now), r.value(now));
    }

    #[test]
    fn recent_value_should_be_none_when_the_recent_slices_hold_nothing() {
        let epoch = Instant::now();
        let mut r = ratio(epoch);
        r.record_expected(100, at(epoch, 0));
        r.record_observed(100, at(epoch, 0));

        // Two slices later nothing has been expected, so there is no evidence either way --
        // which must read as "no data", never as a failing peer.
        assert_eq!(r.recent_value(1, at(epoch, 2)), None);
        assert!(
            r.value(at(epoch, 2)).is_some(),
            "the full window still holds the old slice"
        );
    }

    /// Recovery has to be visible promptly too, not only failure.
    #[test]
    fn recent_value_should_climb_back_before_the_full_window_does() {
        let epoch = Instant::now();
        let mut r = ratio(epoch);

        for sec in 0..3 {
            r.record_expected(100, at(epoch, sec));
        }
        r.record_expected(100, at(epoch, 3));
        r.record_observed(100, at(epoch, 3));

        let now = at(epoch, 3);
        let full = r.value(now).expect("window holds evidence");
        let recent = r.recent_value(1, now).expect("newest slice holds evidence");
        assert_eq!(recent, 1.0, "the newest slice is fully recovered");
        assert!(full < 0.5, "the full window still carries the outage, got {full}");
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
    fn window_should_saturate_rather_than_overflow_on_an_absurd_slice_width() {
        let epoch = Instant::now();
        let r: WindowedRatio<4> = WindowedRatio::new(Duration::MAX, epoch);

        // `Duration::MAX * 4` overflows; multiplying it out would panic instead of answering.
        assert_eq!(Duration::MAX, r.window());
    }

    #[test]
    fn a_late_record_should_not_erase_a_newer_slice() {
        let epoch = Instant::now();
        let mut r = ratio(epoch);

        // Slices 0 and 4 share a slot in a 4-slice ring. Recording slice 4 first and then a late
        // record for slice 0 used to clear the slot, throwing away live evidence in exchange for a
        // slice that has already aged out of the window.
        r.record_expected(100, at(epoch, 4));
        r.record_observed(100, at(epoch, 4));

        r.record_expected(50, epoch);
        r.record_observed(0, epoch);

        assert_eq!(
            Some(1.0),
            r.value(at(epoch, 4)),
            "the late record must not displace the newer slice"
        );
    }

    #[test]
    fn a_late_record_should_still_land_in_a_slice_that_is_still_live() {
        let epoch = Instant::now();
        let mut r = ratio(epoch);

        // Out of order, but slice 0 is still the one that slot holds -- so this is a genuine
        // in-window observation and must be counted, not dropped along with the stale ones.
        r.record_expected(10, at(epoch, 1));
        r.record_expected(10, epoch);
        r.record_observed(10, epoch);

        assert_eq!(Some(0.5), r.value(at(epoch, 1)));
    }

    #[test]
    fn counters_should_saturate_rather_than_overflow() {
        let epoch = Instant::now();
        let mut r = ratio(epoch);

        // Wrapping here would not merely lose precision: the denominator would fall below the
        // numerator and a perfectly healthy window would read as a collapse.
        r.record_expected(u64::MAX, epoch);
        r.record_expected(1, epoch);
        r.record_observed(u64::MAX, epoch);
        r.record_observed(1, epoch);

        assert_eq!(Some(1.0), r.value(epoch));
    }

    #[test]
    fn totals_should_saturate_across_slices() {
        let epoch = Instant::now();
        let mut r = ratio(epoch);

        // Each slice fits in a `u64`; their sum does not.
        for sec in 0..4 {
            r.record_expected(u64::MAX, at(epoch, sec));
            r.record_observed(u64::MAX / 2, at(epoch, sec));
        }

        let value = r.value(at(epoch, 3)).expect("the window holds evidence");
        assert!(
            (0.0..=1.0).contains(&value),
            "a saturated total must stay a ratio, got {value}"
        );
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
