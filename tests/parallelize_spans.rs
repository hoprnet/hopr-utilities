//! Log lines written by blocking tasks keep the span of the code that started them.
//!
//! Its own test binary: the subscriber must be global, because the tasks run on rayon threads.
#![cfg(feature = "parallelize-rayon")]

use std::sync::{Arc, Mutex, OnceLock};

use hopr_utilities::parallelize::cpu;
use tracing::{Instrument, Subscriber, info_span};
use tracing_subscriber::{
    Layer,
    layer::{Context, SubscriberExt},
    registry::LookupSpan,
};

/// For every event: its message and the names of its spans, outermost first.
type Recorded = Arc<Mutex<Vec<(String, Vec<String>)>>>;

#[derive(Clone, Default)]
struct SpanRecorder(Recorded);

impl<S> Layer<S> for SpanRecorder
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_event(&self, event: &tracing::Event<'_>, ctx: Context<'_, S>) {
        let mut message = String::new();
        event.record(&mut |field: &tracing::field::Field, value: &dyn std::fmt::Debug| {
            if field.name() == "message" {
                message = format!("{value:?}");
            }
        });
        let spans = ctx
            .event_scope(event)
            .map(|scope| scope.from_root().map(|span| span.name().to_owned()).collect())
            .unwrap_or_default();
        if let Ok(mut recorded) = self.0.lock() {
            recorded.push((message, spans));
        }
    }
}

fn recorded() -> &'static Recorded {
    static RECORDED: OnceLock<Recorded> = OnceLock::new();
    RECORDED.get_or_init(|| {
        let recorder = SpanRecorder::default();
        let recorded = recorder.0.clone();
        tracing::subscriber::set_global_default(tracing_subscriber::registry().with(recorder))
            .expect("no other subscriber is installed in this test binary");
        recorded
    })
}

/// The span names of the event whose message is `message`.
fn spans_of(message: &str) -> Vec<String> {
    let recorded = recorded().lock().expect("recorder lock");
    let matching: Vec<_> = recorded.iter().filter(|(text, _)| text == message).collect();
    assert_eq!(matching.len(), 1, "expected one event {message:?}, got {recorded:?}");
    matching[0].1.clone()
}

#[tokio::test]
async fn a_task_logs_inside_the_span_that_started_it() {
    recorded();

    cpu::spawn_fifo_blocking(|| tracing::info!("fifo task ran"), "test")
        .instrument(info_span!("session", session_id = 7))
        .await
        .expect("task runs");
    cpu::spawn_blocking(|| tracing::info!("lifo task ran"), "test")
        .instrument(info_span!("session", session_id = 8))
        .await
        .expect("task runs");

    assert_eq!(spans_of("fifo task ran"), ["session"]);
    assert_eq!(spans_of("lifo task ran"), ["session"]);
}

#[tokio::test]
async fn encode_and_decode_tasks_keep_the_whole_span_chain() {
    recorded();

    let transport = info_span!("transport");
    async {
        cpu::spawn_encode_blocking(|| tracing::info!("encode task ran"), "test")
            .await
            .expect("task runs");
        cpu::spawn_decode_blocking(|| tracing::info!("decode task ran"), "test")
            .await
            .expect("task runs");
    }
    .instrument(info_span!(parent: &transport, "packet"))
    .await;

    assert_eq!(spans_of("encode task ran"), ["transport", "packet"]);
    assert_eq!(spans_of("decode task ran"), ["transport", "packet"]);
}

#[tokio::test]
async fn a_task_started_outside_any_span_logs_without_one() {
    recorded();

    cpu::spawn_fifo_blocking(|| tracing::info!("task without a span ran"), "test")
        .await
        .expect("task runs");

    assert!(spans_of("task without a span ran").is_empty());
}
