//! A task that runs under a scoped dispatcher may be dropped on a thread whose default is a
//! different, global subscriber. Releasing its spans must not panic (hoprnet#8686).
//!
//! The tasks are spawned like the hoprnet packet pipeline does it: with `spawn_as_abortable!`,
//! which wraps the future in `futures::future::Abortable` and spawns it on tokio.

use std::{
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    time::Duration,
};

use hopr_utilities::{runtime::WithDispatchOnDrop, spawn_as_abortable};
use tracing::Instrument;
use tracing_subscriber::{
    Layer, filter::LevelFilter, layer::SubscriberExt, registry::LookupSpan, util::SubscriberInitExt,
};

/// Counts panics on any thread whose message contains the tracing-subscriber "no such span" panic.
static SPAN_PANICS: AtomicUsize = AtomicUsize::new(0);
/// The tests share one global subscriber and one panic hook, so they run one at a time.
static SERIAL: Mutex<()> = Mutex::new(());

fn setup() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        // The global subscriber, as test-log installs it.
        tracing_subscriber::registry().with(LevelFilter::TRACE).init();
        let default_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if info.to_string().contains("no such span exists") {
                SPAN_PANICS.fetch_add(1, Ordering::SeqCst);
            }
            default_hook(info);
        }));
    });
}

/// Reports the name of every span the scoped subscriber closes.
struct ClosedSpans(mpsc::Sender<&'static str>);

impl<S: tracing::Subscriber + for<'a> LookupSpan<'a>> Layer<S> for ClosedSpans {
    fn on_close(&self, id: tracing::span::Id, ctx: tracing_subscriber::layer::Context<'_, S>) {
        if let Some(span) = ctx.span(&id) {
            let _ = self.0.send(span.name());
        }
    }
}

enum End {
    /// Fire the abort handle that `spawn_as_abortable!` returns.
    Abort,
    /// Let the future finish on its own.
    Complete,
    /// Drop the runtime while the task is still pending.
    Shutdown,
}

/// Spawns a task that holds `child` (whose parent is `parent`) under a scoped dispatcher, ends it
/// as `end` says, and checks that both spans were closed in the scoped subscriber without a panic.
fn spawned_task_releases_its_spans_in_its_own_subscriber(end: End) {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    setup();
    let panics_before = SPAN_PANICS.load(Ordering::SeqCst);

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .build()
        .expect("runtime must build");
    let (closed_tx, closed) = mpsc::channel();
    let scoped = tracing::Dispatch::new(
        tracing_subscriber::registry()
            .with(LevelFilter::TRACE)
            .with(ClosedSpans(closed_tx)),
    );
    let (polled_tx, polled) = mpsc::channel::<()>();
    let (finish_tx, finish) = tokio::sync::oneshot::channel::<()>();
    let abort = tracing::dispatcher::with_default(&scoped, || {
        let parent = tracing::info_span!(parent: None, "parent");
        let child = tracing::info_span!(parent: &parent, "child");
        let future = async move {
            polled_tx.send(()).expect("test must be waiting");
            let _ = finish.await;
        }
        .instrument(child);
        let _guard = runtime.enter();
        spawn_as_abortable!(WithDispatchOnDrop::new(future))
    });
    // `parent` is held only by `child`, and `child` only by the task.
    polled
        .recv_timeout(Duration::from_secs(5))
        .expect("task must be polled");
    assert!(closed.try_recv().is_err(), "no span may close while the task runs");

    match end {
        End::Abort => abort.abort(),
        End::Complete => finish_tx.send(()).expect("task must be waiting"),
        End::Shutdown => drop(runtime),
    }

    let mut names = Vec::new();
    while !names.contains(&"parent") {
        names.push(
            closed
                .recv_timeout(Duration::from_secs(5))
                .expect("ending the task must close `child` and then `parent` in the scoped subscriber"),
        );
    }
    assert_eq!(names, ["child", "parent"]);
    assert_eq!(
        SPAN_PANICS.load(Ordering::SeqCst),
        panics_before,
        "no thread may panic with `no such span`"
    );
}

#[test]
fn an_aborted_task_releases_its_spans_in_its_own_subscriber() {
    spawned_task_releases_its_spans_in_its_own_subscriber(End::Abort);
}

#[test]
fn a_completed_task_releases_its_spans_in_its_own_subscriber() {
    spawned_task_releases_its_spans_in_its_own_subscriber(End::Complete);
}

#[test]
fn a_task_dropped_at_runtime_shutdown_releases_its_spans_in_its_own_subscriber() {
    spawned_task_releases_its_spans_in_its_own_subscriber(End::Shutdown);
}
