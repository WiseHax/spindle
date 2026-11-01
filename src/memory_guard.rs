//! Keeps spindle from taking the laptop down when the model is bigger than free RAM.
//!
//! Model weights are memory-mapped, so on their own they never fail an allocation. The
//! danger is different: as pages fault in, our working set grows without limit and Windows
//! starts evicting *other* programs' memory. The desktop freezes, which to the user is the
//! same as a crash.
//!
//! The governor puts a hard cap on our working set (`SetProcessWorkingSetSizeEx` with
//! `QUOTA_LIMITS_HARDWS_MAX_ENABLE`) and moves it every 250 ms so a fixed amount of RAM
//! always stays free for everything else. Above the cap, Windows recycles our own model
//! pages instead of someone else's: spindle gets slower, the laptop stays usable.
//! Evicted model pages land on the standby list, so re-touching them is usually a soft fault.

use std::{
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use windows::Win32::System::{
    Memory::{SetProcessWorkingSetSizeEx, QUOTA_LIMITS_HARDWS_MAX_DISABLE, QUOTA_LIMITS_HARDWS_MAX_ENABLE, QUOTA_LIMITS_HARDWS_MIN_DISABLE},
    ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS},
    Threading::GetCurrentProcess,
};

use crate::mmap_file::available_ram_bytes;

pub const MIB: u64 = 1024 * 1024;

/// RAM left free for Windows and other apps unless `--reserve` says otherwise. "Available"
/// already includes the standby cache, which other apps can take instantly, so this doesn't
/// need to be large. 1 GiB squeezed a 610 MiB model on a busy 8 GB laptop for no benefit.
pub const DEFAULT_RESERVE: u64 = 512 * MIB;
/// Ignore cap changes smaller than this to avoid a syscall every tick.
const HYSTERESIS: u64 = 16 * MIB;

#[derive(Clone, Copy, Debug)]
pub struct Budget {
    /// System RAM that must stay available to others.
    pub reserve: u64,
    /// Never cap below this (our private memory plus a little room for weights).
    pub floor: u64,
    /// User limit from `--ram-budget`, if any.
    pub ceiling: Option<u64>,
    /// Minimum working set. Must cover pinned (`VirtualLock`ed) weights, since Windows only
    /// lets a process lock about as much as its minimum working set.
    pub min_ws: u64,
}

/// Next working-set cap: we may grow into whatever free RAM exceeds the reserve, and must
/// give back whatever the reserve is short by. Clamped to `[floor, ceiling]`.
pub fn next_cap(working_set: u64, available: u64, b: &Budget) -> u64 {
    let cap = (working_set + available).saturating_sub(b.reserve).max(b.floor);
    b.ceiling.map_or(cap, |c| cap.min(c.max(b.floor)))
}

/// `(working set, private bytes, page faults so far)` of this process.
pub fn process_counters() -> (u64, u64, u64) {
    unsafe {
        let mut pmc = PROCESS_MEMORY_COUNTERS {
            cb: std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
            ..Default::default()
        };
        if !K32GetProcessMemoryInfo(GetCurrentProcess(), &mut pmc, pmc.cb).as_bool() {
            return (0, 0, 0);
        }
        (pmc.WorkingSetSize as u64, pmc.PagefileUsage as u64, pmc.PageFaultCount as u64)
    }
}

fn apply_cap(min_ws: u64, cap: u64) -> bool {
    unsafe {
        SetProcessWorkingSetSizeEx(
            GetCurrentProcess(),
            min_ws.min(cap) as usize,
            cap as usize,
            QUOTA_LIMITS_HARDWS_MAX_ENABLE | QUOTA_LIMITS_HARDWS_MIN_DISABLE,
        )
        .is_ok()
    }
}

/// Raises the minimum working set to `bytes` (soft minimum, no hard max) so that much can be
/// locked. Returns false if Windows refuses.
pub fn allow_locking(bytes: u64) -> bool {
    unsafe {
        SetProcessWorkingSetSizeEx(
            GetCurrentProcess(),
            bytes as usize,
            (bytes + 256 * MIB) as usize,
            QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE,
        )
        .is_ok()
    }
}

/// Live numbers the chat can print.
#[derive(Default)]
pub struct GovernorStatus {
    pub cap: AtomicU64,
    pub available: AtomicU64,
    /// Ticks where free RAM was below the reserve and we had to shrink.
    pub squeezes: AtomicU64,
}

pub struct Governor {
    pub status: Arc<GovernorStatus>,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl Governor {
    /// Applies the first cap right away (so startup is protected too), then keeps adjusting
    /// it from a background thread. Returns `None` if Windows refuses the hard limit.
    pub fn start(budget: Budget) -> Option<Self> {
        let (ws, _, _) = process_counters();
        let avail = available_ram_bytes();
        let cap = next_cap(ws, avail, &budget);
        if !apply_cap(budget.min_ws, cap) {
            return None;
        }
        let status = Arc::new(GovernorStatus::default());
        status.cap.store(cap, Ordering::Relaxed);
        status.available.store(avail, Ordering::Relaxed);
        let stop = Arc::new(AtomicBool::new(false));

        let (st, sp) = (Arc::clone(&status), Arc::clone(&stop));
        let handle = thread::Builder::new()
            .name("mem-governor".into())
            .spawn(move || {
                let mut last = cap;
                while !sp.load(Ordering::Relaxed) {
                    thread::sleep(Duration::from_millis(250));
                    let (ws, _, _) = process_counters();
                    let avail = available_ram_bytes();
                    st.available.store(avail, Ordering::Relaxed);
                    if avail < budget.reserve {
                        st.squeezes.fetch_add(1, Ordering::Relaxed);
                    }
                    let cap = next_cap(ws, avail, &budget);
                    if cap.abs_diff(last) >= HYSTERESIS && apply_cap(budget.min_ws, cap) {
                        last = cap;
                        st.cap.store(cap, Ordering::Relaxed);
                    }
                }
            })
            .ok()?;
        Some(Self { status, stop, handle: Some(handle) })
    }
}

impl Drop for Governor {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        unsafe {
            let _ = SetProcessWorkingSetSizeEx(GetCurrentProcess(), 0, 0, QUOTA_LIMITS_HARDWS_MAX_DISABLE);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: u64 = 1024 * MIB;

    fn b(ceiling: Option<u64>) -> Budget {
        Budget { reserve: G, floor: 300 * MIB, ceiling, min_ws: 0 }
    }

    #[test]
    fn grows_into_spare_ram() {
        // 500 MiB used, 3 GiB free, keep 1 GiB: may use 500 MiB + 2 GiB.
        assert_eq!(next_cap(500 * MIB, 3 * G, &b(None)), 500 * MIB + 2 * G);
    }

    #[test]
    fn shrinks_when_reserve_is_short() {
        // Only 600 MiB free: give back the 424 MiB the reserve is missing.
        assert_eq!(next_cap(2 * G, 600 * MIB, &b(None)), 2 * G - 424 * MIB);
    }

    #[test]
    fn never_below_floor() {
        assert_eq!(next_cap(400 * MIB, 0, &b(None)), 300 * MIB);
    }

    #[test]
    fn respects_user_ceiling_but_not_below_floor() {
        assert_eq!(next_cap(500 * MIB, 6 * G, &b(Some(2 * G))), 2 * G);
        assert_eq!(next_cap(500 * MIB, 6 * G, &b(Some(100 * MIB))), 300 * MIB);
    }
}
