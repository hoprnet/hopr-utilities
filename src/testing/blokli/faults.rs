//! Fault injection for [`TestChainConnector`](super::connector::TestChainConnector).

use std::sync::atomic::Ordering;

use dashmap::{DashMap, DashSet};
use hopr_api::chain::ChainEvent;

use super::connector::TestConnectorError;

/// How a chain operation should misbehave.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Fault {
    /// Operate normally.
    #[default]
    None,
    /// Return an error.
    Fail,
    /// Never resolve.  Models an RPC that neither answers nor times out.
    Hang,
}

/// Chain operations that [`ChainFaults`] can perturb, each naming the call it
/// stands for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChainOp {
    /// `ChainReadSafeOperations::safe_info`.
    SafeInfo,
    /// `ChainValues::balance`, for the safe and for per-peer stake.
    Balance,
    /// `ChainValues::minimum_ticket_price`.
    TicketPrice,
    /// `ChainValues::minimum_incoming_ticket_win_prob`.
    WinProb,
    /// `ChainValues::typical_resolution_time`.
    ResolutionTime,
    /// `ChainReadChannelOperations::stream_channels`.
    StreamChannels,
    /// `ChainReadAccountOperations::stream_accounts`.
    StreamAccounts,
    /// `ChainWriteChannelOperations::open_channel`.
    OpenChannel,
    /// `ChainWriteChannelOperations::fund_channel`.
    FundChannel,
    /// `ChainWriteChannelOperations::close_channel`, which both initiates and
    /// finalizes a closure depending on the channel's status.
    CloseChannel,
}

/// Chain event kinds that [`ChainFaults`] can withhold from subscribers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventKind {
    /// `ChainEvent::ChannelBalanceIncreased`, reporting a funded channel.
    BalanceIncreased,
    /// `ChainEvent::ChannelBalanceDecreased`, reporting a drained channel.
    BalanceDecreased,
    /// `ChainEvent::ChannelOpened`.
    ChannelOpened,
    /// `ChainEvent::ChannelClosureInitiated`, the first of the two closure
    /// steps: the channel has entered its notice period.
    ClosureInitiated,
    /// `ChainEvent::ChannelClosed`, the second closure step.
    Closed,
    /// `ChainEvent::TicketRedeemed`.
    TicketRedeemed,
}

impl EventKind {
    fn of(event: &ChainEvent) -> Option<Self> {
        match event {
            ChainEvent::ChannelBalanceIncreased(..) => Some(Self::BalanceIncreased),
            ChainEvent::ChannelBalanceDecreased(..) => Some(Self::BalanceDecreased),
            ChainEvent::ChannelOpened(_) => Some(Self::ChannelOpened),
            ChainEvent::ChannelClosureInitiated(_) => Some(Self::ClosureInitiated),
            ChainEvent::ChannelClosed(_) => Some(Self::Closed),
            ChainEvent::TicketRedeemed(..) => Some(Self::TicketRedeemed),
            _ => None,
        }
    }
}

/// Shared, live-mutable fault configuration for a
/// [`TestChainConnector`](super::connector::TestChainConnector).
///
/// Handed out by [`TestChainConnector::faults`](super::connector::TestChainConnector::faults) so
/// a test can perturb the chain while the strategy is running — which is how these failures
/// actually occur. Empty by default.
///
/// ```text
/// let faults = connector.faults();
///
/// // The funding tx lands, but its outcome never comes back and the event that
/// // would report it is lost — the strategy learns nothing either way.
/// faults.set_confirmation(ChainOp::FundChannel, Fault::Hang);
/// faults.withhold_event(EventKind::BalanceIncreased);
///
/// // ... drive the strategy, then let the chain recover:
/// faults.clear(ChainOp::FundChannel);
/// faults.deliver_event(EventKind::BalanceIncreased);
///
/// // What the strategy did meanwhile:
/// assert_eq!(faults.calls(ChainOp::FundChannel), 1);
/// assert_eq!(faults.peak_in_flight(ChainOp::FundChannel), 1);
/// ```
///
/// `text` because the example needs a connected `TestChainConnector`, which exists only under the
/// `testing-blokli` feature.
#[derive(Debug, Default)]
pub struct ChainFaults {
    ops: DashMap<ChainOp, Fault>,
    /// Faults applied to the confirmation future of a write op, rather than to
    /// its submission.
    confirmations: DashMap<ChainOp, Fault>,
    withheld_events: DashSet<EventKind>,
    calls: DashMap<ChainOp, usize>,
    /// Writes between submission and confirmation, and the most ever outstanding
    /// at once, per kind and in total.  Lets a test observe what the strategy
    /// does in parallel, not just what it eventually achieves.
    in_flight: DashMap<ChainOp, usize>,
    peak_in_flight: DashMap<ChainOp, usize>,
    /// Total outstanding writes across all kinds, tracked directly rather than summed from
    /// `in_flight` on read — a concurrent `leave_in_flight` racing that sum would let two
    /// simultaneously outstanding writes each observe a total of 1, under-reporting the peak.
    in_flight_total: std::sync::atomic::AtomicUsize,
    peak_in_flight_total: std::sync::atomic::AtomicUsize,
}

impl ChainFaults {
    /// Makes `op` misbehave from now on.  For writes this is *submission*; see
    /// [`ChainFaults::set_confirmation`].
    pub fn set(&self, op: ChainOp, fault: Fault) {
        self.ops.insert(op, fault);
    }

    /// Makes `op`'s confirmation misbehave while submission still succeeds: a tx
    /// accepted but whose outcome never arrives (`Hang`) or fails (`Fail`).
    pub fn set_confirmation(&self, op: ChainOp, fault: Fault) {
        self.confirmations.insert(op, fault);
    }

    /// Restores normal behaviour of `op`, both submission and confirmation.
    pub fn clear(&self, op: ChainOp) {
        self.ops.remove(&op);
        self.confirmations.remove(&op);
    }

    /// Stops delivering `kind` to subscribers, as the lossy broadcast does: the
    /// on-chain effect still happens, the notification does not.
    pub fn withhold_event(&self, kind: EventKind) {
        self.withheld_events.insert(kind);
    }

    /// Resumes delivery of `kind`.
    pub fn deliver_event(&self, kind: EventKind) {
        self.withheld_events.remove(&kind);
    }

    /// Times `op` was invoked, counted on entry, before any injected fault.
    pub fn calls(&self, op: ChainOp) -> usize {
        self.calls.get(&op).map(|c| *c).unwrap_or(0)
    }

    /// Most `op` transactions that were ever in flight at the same time.
    pub fn peak_in_flight(&self, op: ChainOp) -> usize {
        self.peak_in_flight.get(&op).map(|c| *c).unwrap_or(0)
    }

    /// Most writes of any kind ever in flight at once — what
    /// `concurrency.max_concurrent_actions` bounds.
    pub fn peak_in_flight_total(&self) -> usize {
        self.peak_in_flight_total.load(Ordering::Relaxed)
    }

    /// Marks a submitted write as outstanding and updates the watermarks.  The
    /// returned guard releases it when dropped.
    #[must_use]
    pub(super) fn enter_in_flight(self: &std::sync::Arc<Self>, op: ChainOp) -> InFlightGuard {
        // Scoped so the entry guard is released before the map is iterated below.
        let outstanding = {
            let mut entry = self.in_flight.entry(op).or_insert(0);
            *entry += 1;
            *entry
        };

        self.peak_in_flight
            .entry(op)
            .and_modify(|peak| *peak = (*peak).max(outstanding))
            .or_insert(outstanding);

        let total = self.in_flight_total.fetch_add(1, Ordering::Relaxed) + 1;
        self.peak_in_flight_total.fetch_max(total, Ordering::Relaxed);

        InFlightGuard {
            faults: std::sync::Arc::clone(self),
            op,
        }
    }

    /// Marks an outstanding write as finished.
    fn leave_in_flight(&self, op: ChainOp) {
        if let Some(mut outstanding) = self.in_flight.get_mut(&op) {
            *outstanding = outstanding.saturating_sub(1);
        }
        let _ = self
            .in_flight_total
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |t| Some(t.saturating_sub(1)));
    }

    fn fault(&self, op: ChainOp) -> Fault {
        self.ops.get(&op).map(|f| *f).unwrap_or_default()
    }

    fn confirmation_fault(&self, op: ChainOp) -> Fault {
        self.confirmations.get(&op).map(|f| *f).unwrap_or_default()
    }

    fn record(&self, op: ChainOp) {
        *self.calls.entry(op).or_insert(0) += 1;
    }

    pub(super) fn is_withheld(&self, event: &ChainEvent) -> bool {
        EventKind::of(event).is_some_and(|kind| self.withheld_events.contains(&kind))
    }

    /// Applies a `Fail`/`Hang` fault to an async operation, if one is set.
    pub(super) async fn gate(&self, op: ChainOp) -> Result<(), TestConnectorError> {
        self.record(op);
        match self.fault(op) {
            Fault::None => Ok(()),
            Fault::Fail => Err(injected_fault(op)),
            Fault::Hang => futures::future::pending().await,
        }
    }

    /// Applies a fault to a stream-returning operation: `Fail` errors from the
    /// call, `Hang` is returned so the caller can yield a pending stream.
    pub(super) fn gate_stream(&self, op: ChainOp) -> Result<Fault, TestConnectorError> {
        self.record(op);
        match self.fault(op) {
            Fault::Fail => Err(injected_fault(op)),
            other => Ok(other),
        }
    }

    /// Resolves the confirmation future of write op `op`.
    pub(super) async fn confirm(&self, op: ChainOp) -> Result<(), TestConnectorError> {
        match self.confirmation_fault(op) {
            Fault::None => Ok(()),
            Fault::Fail => Err(injected_fault(op)),
            Fault::Hang => futures::future::pending().await,
        }
    }
}

/// Holds an operation's in-flight count up for as long as it lives.
///
/// Tied to the confirmation future's lifetime rather than to a call at its end,
/// so a caller that drops the future without polling it to completion — an
/// aborted task, say — cannot leave the count raised and every later watermark
/// reading too high.
pub struct InFlightGuard {
    faults: std::sync::Arc<ChainFaults>,
    op: ChainOp,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.faults.leave_in_flight(self.op);
    }
}

fn injected_fault(op: ChainOp) -> TestConnectorError {
    TestConnectorError::from(anyhow::anyhow!("injected fault on {op:?}"))
}
