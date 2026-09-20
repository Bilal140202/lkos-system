//! In-process event bus.
//!
//! Applications (Tauri frontends, agent runtimes) subscribe to knowledge
//! changes instead of polling. Events are delivered synchronously to all
//! subscribers via std channels; slow subscribers only affect their own queue.

use crate::types::Event;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Mutex;

/// Broadcast bus for knowledge-change events.
#[derive(Default)]
pub struct EventBus {
    senders: Mutex<Vec<Sender<Event>>>,
}

impl EventBus {
    /// Create an empty bus.
    pub fn new() -> Self {
        Self::default()
    }

    /// Subscribe; returns a receiver of future events.
    pub fn subscribe(&self) -> Receiver<Event> {
        let (tx, rx) = channel();
        self.senders.lock().expect("event bus lock").push(tx);
        rx
    }

    /// Publish an event to all subscribers (drop-quietly if nobody listens).
    pub fn publish(&self, name: &str, subject_id: i64, payload: &str) {
        let event = Event {
            name: name.to_string(),
            subject_id,
            payload: payload.to_string(),
            at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
        };
        let mut senders = self.senders.lock().expect("event bus lock");
        // Drop dead receivers so the bus doesn't leak.
        senders.retain(|s| s.send(event.clone()).is_ok());
    }
}
