//! A pool job may hold the last reference to a span of a scoped dispatcher while the pool thread's
//! default is a different, global subscriber. Releasing that reference must not panic (hoprnet#8686).

use std::{
    future::Future,
    pin::pin,
    sync::mpsc,
    task::{Context, Poll, Waker},
    time::Duration,
};

use hopr_utilities::parallelize::cpu::{self, rayon};
use tracing_subscriber::{
    Layer, filter::LevelFilter, layer::SubscriberExt, registry::LookupSpan, util::SubscriberInitExt,
};

/// Reports the name of every span the scoped subscriber closes.
struct ClosedSpans(mpsc::Sender<&'static str>);

impl<S: tracing::Subscriber + for<'a> LookupSpan<'a>> Layer<S> for ClosedSpans {
    fn on_close(&self, id: tracing::span::Id, ctx: tracing_subscriber::layer::Context<'_, S>) {
        if let Some(span) = ctx.span(&id) {
            let _ = self.0.send(span.name());
        }
    }
}

#[test]
fn a_pool_job_releasing_the_last_span_reference_does_not_abort() {
    tracing_subscriber::registry().with(LevelFilter::TRACE).init();
    cpu::init_thread_pool(1).expect("global pool must be built once in this binary");

    // Occupy the only pool thread so the task remains queued until its future is dropped.
    let (started, blocker_started) = mpsc::channel::<()>();
    let (release, blocked) = mpsc::channel::<()>();
    rayon::spawn(move || {
        started.send(()).expect("test must be waiting");
        let _ = blocked.recv();
    });
    blocker_started
        .recv_timeout(Duration::from_secs(5))
        .expect("blocking job must start");

    let (closed_tx, closed) = mpsc::channel();
    let scoped = tracing::Dispatch::new(
        tracing_subscriber::registry()
            .with(LevelFilter::TRACE)
            .with(ClosedSpans(closed_tx)),
    );
    tracing::dispatcher::with_default(&scoped, || {
        let parent = tracing::info_span!(parent: None, "parent");
        let child = tracing::info_span!(parent: &parent, "child");
        let _entered = child.enter();

        // The first poll captures the child span and queues the job.
        let mut job = pin!(cpu::spawn_fifo_blocking(|| (), "test"));
        assert!(matches!(
            job.as_mut().poll(&mut Context::from_waker(Waker::noop())),
            Poll::Pending
        ));
        // Dropping the future cancels the queued job.
    });
    // The queued job holds the final reference to child.
    assert!(closed.try_recv().is_err(), "no span may close before the job runs");

    release.send(()).expect("blocking job must be waiting");
    let mut names = Vec::new();
    while !names.contains(&"parent") {
        names.push(
            closed
                .recv_timeout(Duration::from_secs(5))
                .expect("the job must close `child` and then `parent` in the scoped subscriber"),
        );
    }
    assert_eq!(names, ["child", "parent"]);
}
