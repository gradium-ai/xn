//! The pool is a process-wide singleton with a single job slot, and a dispatch that finds it
//! busy deliberately runs serially instead of queueing. Asserting on fan-out therefore only
//! means anything when nothing else in the process is dispatching -- hence a dedicated test
//! binary, and a mutex to keep these tests out of each other's way inside it.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};
use xn::threadpool::{dispatch, size};

static EXCLUSIVE: Mutex<()> = Mutex::new(());

fn exclusive() -> std::sync::MutexGuard<'static, ()> {
    EXCLUSIVE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Mirrors the private `threadpool::enabled()`; some guarantees below are the pool's.
fn pool_enabled() -> bool {
    !matches!(std::env::var("XN_THREADPOOL").as_deref(), Ok("0"))
}

#[test]
fn every_participant_runs_exactly_once() {
    let _x = exclusive();
    // Repeat: a worker that lags into the *next* job would show up as a miscount, and only
    // sometimes.
    for round in 0..2000 {
        let seen: Vec<AtomicU32> = (0..size()).map(|_| AtomicU32::new(0)).collect();
        dispatch(|ith, n| {
            assert_eq!(n, size());
            seen[ith].fetch_add(1, Ordering::Relaxed);
        });
        for (i, c) in seen.iter().enumerate() {
            assert_eq!(c.load(Ordering::Relaxed), 1, "round={round} ith={i}");
        }
    }
}

#[test]
fn results_are_visible_to_the_caller() {
    let _x = exclusive();
    const N: usize = 4096;
    for _ in 0..2000 {
        let mut out = vec![0u64; N];
        let ptr = out.as_mut_ptr() as usize;
        dispatch(|ith, nth| {
            let p = ptr as *mut u64;
            let mut i = ith;
            while i < N {
                // SAFETY: stripes by `ith` are disjoint.
                unsafe { *p.add(i) = (i as u64) + 1 };
                i += nth;
            }
        });
        assert!(out.iter().enumerate().all(|(i, v)| *v == i as u64 + 1));
    }
}

#[test]
fn nested_dispatch_does_not_deadlock() {
    let _x = exclusive();
    let inner = AtomicU32::new(0);
    dispatch(|_, _| {
        dispatch(|ith, nth| {
            // The pool avoids the deadlock by running serially; rayon does by nesting.
            if pool_enabled() {
                assert_eq!((ith, nth), (0, 1), "a nested dispatch must not fan out");
            }
            assert!(ith < nth, "participant {ith} of {nth} is out of range");
            inner.fetch_add(1, Ordering::Relaxed);
        });
    });
    assert!(inner.load(Ordering::Relaxed) >= size() as u32);
}

/// Callers size work with [`size`]; a `dispatch` wider than that leaves some unclaimed.
#[test]
fn dispatch_width_matches_size() {
    let _x = exclusive();
    let seen = AtomicU32::new(0);
    dispatch(|_, nth| {
        assert_eq!(nth, size(), "dispatch fanned out to a width `size()` does not report");
        seen.fetch_add(1, Ordering::Relaxed);
    });
    assert_eq!(seen.load(Ordering::Relaxed), size() as u32);
}

#[test]
fn a_worker_panic_reaches_the_caller() {
    let _x = exclusive();
    if size() < 2 {
        return;
    }
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let r = std::panic::catch_unwind(|| {
        dispatch(|ith, _| {
            if ith != 0 {
                panic!("boom");
            }
        });
    });
    std::panic::set_hook(hook);
    assert!(r.is_err(), "a panicking worker must not be swallowed");

    // The pool must still be usable, and must not report the old panic again.
    let n = AtomicU32::new(0);
    dispatch(|_, _| {
        n.fetch_add(1, Ordering::Relaxed);
    });
    assert_eq!(n.load(Ordering::Relaxed), size() as u32);
}

/// Several threads driving the pool at once.
///
/// Only one may publish; the rest fall back to running on their own thread. Getting that
/// handover wrong does not deadlock -- it silently lets two publishers share one job slot, so
/// workers run the wrong closure or skip their share, and the damage shows up as wrong numbers
/// in whichever kernel happened to be running. This is the shape of test that catches it.
#[test]
fn concurrent_publishers_each_get_a_complete_result() {
    // Contention is the point, but it comes from the threads spawned below -- the other tests
    // in this file assert on fan-out and must not be caught in it.
    let _x = exclusive();
    const N: usize = 1024;
    let threads = 4;
    let bad = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let mut handles = Vec::new();
    for t in 0..threads {
        let bad = bad.clone();
        handles.push(std::thread::spawn(move || {
            for round in 0..4000 {
                // A value unique to this thread and round, so a stripe written by someone
                // else's job is detectable rather than coincidentally right.
                let tag = ((t * 4000 + round) as u64) << 20;
                let mut out = vec![0u64; N];
                let ptr = out.as_mut_ptr() as usize;
                dispatch(|ith, nth| {
                    let p = ptr as *mut u64;
                    let mut i = ith;
                    while i < N {
                        // SAFETY: stripes by `ith` are disjoint within this job, and each job
                        // owns its own `out`.
                        unsafe { *p.add(i) = tag | (i as u64) };
                        i += nth;
                    }
                });
                if out.iter().enumerate().any(|(i, v)| *v != tag | (i as u64)) {
                    bad.fetch_add(1, Ordering::Relaxed);
                }
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    assert_eq!(bad.load(Ordering::Relaxed), 0, "a concurrent dispatch produced a partial result");
}
