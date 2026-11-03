//! Tiny persistent worker pool for per-token parallel matvecs.
//!
//! Spawning threads per matvec (~210 matvecs per token) would cost more than the math, so the
//! helpers stay alive, spin briefly waiting for the next job (low latency between layers),
//! then `park()` so idle chat time (waiting for `You > ` input) burns no CPU.

use std::{
    cell::UnsafeCell,
    hint::spin_loop,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    thread::{self, JoinHandle},
};

/// A job is called once per thread as `f(thread_idx, n_threads)`.
type JobFn = dyn Fn(usize, usize) + Sync;

/// How many spin iterations a helper waits before parking.
const SPIN_BEFORE_PARK: usize = 200_000;

struct Shared {
    n_threads: usize,
    generation: AtomicUsize,
    remaining: AtomicUsize,
    shutdown: AtomicBool,
    parked_mask: AtomicUsize,
    job: UnsafeCell<Option<*const JobFn>>,
}

// SAFETY: `job` is written only by the owner thread while no helper is reading it
// (between `remaining == 0` and the next `generation` bump), and read by helpers only after
// observing the new generation with Acquire ordering.
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

pub struct WorkerPool {
    shared: Arc<Shared>,
    handles: Vec<JoinHandle<()>>,
}

impl WorkerPool {
    /// Creates a pool of `n_threads` total (the calling thread counts as thread 0, so
    /// `n_threads - 1` helpers are spawned). `pin(i)` runs first on helper `i` (1-based).
    pub fn new<P: Fn(usize) + Send + Sync + 'static>(n_threads: usize, pin: P) -> Self {
        let n_threads = n_threads.max(1);
        let shared = Arc::new(Shared {
            n_threads,
            generation: AtomicUsize::new(0),
            remaining: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            parked_mask: AtomicUsize::new(0),
            job: UnsafeCell::new(None),
        });
        let pin = Arc::new(pin);
        let handles = (1..n_threads)
            .map(|i| {
                let shared = Arc::clone(&shared);
                let pin = Arc::clone(&pin);
                thread::Builder::new()
                    .name(format!("matvec-{i}"))
                    .spawn(move || {
                        pin(i);
                        worker_loop(&shared, i);
                    })
                    .expect("failed to spawn matvec worker")
            })
            .collect();
        Self { shared, handles }
    }

    pub fn n_threads(&self) -> usize {
        self.shared.n_threads
    }

    /// Runs `f(t, n)` on every thread `t in 0..n` (thread 0 = caller) and waits for all.
    pub fn run(&self, f: &(dyn Fn(usize, usize) + Sync)) {
        let s = &*self.shared;
        let n = s.n_threads;
        if n == 1 {
            f(0, 1);
            return;
        }
        // SAFETY: we block below until every helper finished calling `f`, so erasing the
        // lifetime never lets a helper observe a dangling closure.
        unsafe {
            let p: *const JobFn = std::mem::transmute(f as *const (dyn Fn(usize, usize) + Sync + '_));
            *s.job.get() = Some(p);
        }
        s.remaining.store(n - 1, Ordering::Relaxed);
        s.generation.fetch_add(1, Ordering::Release);
        let parked = s.parked_mask.load(Ordering::Relaxed);
        if parked != 0 {
            for (i, h) in self.handles.iter().enumerate() {
                if (parked & (1 << (i + 1))) != 0 {
                    h.thread().unpark();
                }
            }
        }
        f(0, n);
        while s.remaining.load(Ordering::Acquire) != 0 {
            spin_loop();
        }
    }
}

fn worker_loop(s: &Shared, idx: usize) {
    let mut seen = 0usize;
    let mut spins = 0usize;
    loop {
        let g = s.generation.load(Ordering::Acquire);
        if g != seen {
            seen = g;
            if s.shutdown.load(Ordering::Acquire) {
                return;
            }
            // SAFETY: see `Shared`, the job pointer is stable for this generation.
            let f = unsafe { &*(*s.job.get()).unwrap() };
            f(idx, s.n_threads);
            s.remaining.fetch_sub(1, Ordering::Release);
            spins = 0;
            continue;
        }
        if spins < SPIN_BEFORE_PARK {
            spins += 1;
            spin_loop();
        } else {
            s.parked_mask.fetch_or(1 << idx, Ordering::Release);
            if s.generation.load(Ordering::Acquire) != seen {
                s.parked_mask.fetch_and(!(1 << idx), Ordering::Release);
                continue;
            }
            thread::park();
            s.parked_mask.fetch_and(!(1 << idx), Ordering::Release);
            spins = 0;
        }
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        self.shared.generation.fetch_add(1, Ordering::Release);
        for h in self.handles.drain(..) {
            h.thread().unpark();
            let _ = h.join();
        }
    }
}
