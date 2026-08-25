use gpui::{PlatformDispatcher, Priority, RunnableVariant};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex, mpsc},
    thread::ThreadId,
    time::{Duration, Instant},
};

struct QueuedRunnable {
    ready_at: Instant,
    runnable: RunnableVariant,
}

/// The work and deadline result returned by an explicit host polling phase.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PollOutcome {
    /// Number of foreground tasks run by the poll.
    pub tasks_run: usize,
    /// Earliest queued deadline, if one remains pending.
    pub next_deadline: Option<Instant>,
}

/// A main-thread dispatcher driven by [`crate::EmbeddedGpui::poll`].
pub(crate) struct EmbeddedDispatcher {
    main_thread: ThreadId,
    queue: Mutex<VecDeque<QueuedRunnable>>,
    background_sender: mpsc::Sender<RunnableVariant>,
    wake: Option<Arc<dyn Fn() + Send + Sync>>,
}

impl EmbeddedDispatcher {
    pub(crate) fn new(wake: Option<Arc<dyn Fn() + Send + Sync>>) -> Arc<Self> {
        let (background_sender, background_receiver) = mpsc::channel::<RunnableVariant>();
        let background_receiver = Arc::new(Mutex::new(background_receiver));
        let worker_count = std::thread::available_parallelism().map_or(1, |count| count.get());

        for _ in 0..worker_count {
            let background_receiver = background_receiver.clone();
            std::thread::spawn(move || {
                loop {
                    let runnable = {
                        let receiver = background_receiver
                            .lock()
                            .expect("embedded background receiver poisoned");
                        receiver.recv()
                    };
                    let Ok(runnable) = runnable else { break };
                    runnable.run();
                }
            });
        }

        Arc::new(Self {
            main_thread: std::thread::current().id(),
            queue: Mutex::new(VecDeque::new()),
            background_sender,
            wake,
        })
    }

    pub(crate) fn poll(&self) -> PollOutcome {
        let mut ran = 0;
        loop {
            let now = Instant::now();
            let runnable = {
                let mut queue = self
                    .queue
                    .lock()
                    .expect("embedded dispatcher queue poisoned");
                queue
                    .iter()
                    .position(|item| item.ready_at <= now)
                    .and_then(|index| queue.remove(index))
            };
            let Some(item) = runnable else { break };
            item.runnable.run();
            ran += 1;
        }
        let next_deadline = self
            .queue
            .lock()
            .expect("embedded dispatcher queue poisoned")
            .iter()
            .map(|item| item.ready_at)
            .min();
        PollOutcome {
            tasks_run: ran,
            next_deadline,
        }
    }

    fn enqueue(&self, runnable: RunnableVariant, ready_at: Instant) {
        self.queue
            .lock()
            .expect("embedded dispatcher queue poisoned")
            .push_back(QueuedRunnable { ready_at, runnable });
        self.wake_now();
    }

    fn wake_now(&self) {
        if let Some(wake) = &self.wake {
            wake();
        }
    }
}

impl PlatformDispatcher for EmbeddedDispatcher {
    fn is_main_thread(&self) -> bool {
        std::thread::current().id() == self.main_thread
    }

    fn dispatch(&self, runnable: RunnableVariant, _priority: Priority) {
        if self.background_sender.send(runnable).is_ok() {
            self.wake_now();
        }
    }

    fn dispatch_on_main_thread(&self, runnable: RunnableVariant, _priority: Priority) {
        self.enqueue(runnable, Instant::now());
    }

    fn dispatch_after(&self, duration: Duration, runnable: RunnableVariant) {
        let deadline = Instant::now() + duration;
        self.enqueue(runnable, deadline);
    }

    fn spawn_realtime(&self, f: Box<dyn FnOnce() + Send>) {
        std::thread::spawn(f);
    }
}
