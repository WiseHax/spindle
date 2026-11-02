//! In-RAM Expert Cache for streaming MoE weights from disk.
//!
//! Replaces mmap-demand paging for MoE expert layers with an explicit, bounded RAM cache.
//! Sized to a target memory budget (e.g. 256 MiB–1 GiB), it holds `N` 4096-byte-aligned
//! expert slots. Misses are read with buffered random-access I/O, either from the repacked
//! `.spindle.pack` (one read per expert) or straight from the GGUF (gate, up and down as
//! three reads), so misses can be served from the OS file cache. Unbuffered direct I/O
//! remains opt-in via `SPINDLE_DIRECT_IO=1`, for packs only.

use std::{
    alloc::{alloc, dealloc, Layout},
    collections::HashMap,
    fs::File,
    os::windows::fs::{FileExt, OpenOptionsExt},
    sync::{Arc, Condvar, Mutex},
    thread::{self, JoinHandle},
};

use windows::Win32::{
    Storage::FileSystem::{FILE_FLAG_NO_BUFFERING, FILE_FLAG_RANDOM_ACCESS, FILE_FLAG_SEQUENTIAL_SCAN},
    System::{
        SystemInformation::{CpuSetInformation, GetSystemCpuSetInformation, SYSTEM_CPU_SET_INFORMATION},
        Threading::{GetCurrentProcess, GetCurrentThread, SetThreadAffinityMask, SetThreadSelectedCpuSets},
    },
};

/// Pins the current reader thread to the E-core group (EfficiencyClass < max) if the CPU is
/// hybrid, so background `ReadFile` / kernel page-cache copies never preempt the 4 P-cores
/// while they run AVX2 `ffn_forward` on already-cached experts.
fn pin_current_thread_to_ecores() {
    unsafe {
        let mut needed = 0u32;
        let _ = GetSystemCpuSetInformation(None, 0, &mut needed, GetCurrentProcess(), 0);
        if needed == 0 {
            return;
        }
        let mut buf = vec![0u8; needed as usize];
        if !GetSystemCpuSetInformation(
            Some(buf.as_mut_ptr() as *mut SYSTEM_CPU_SET_INFORMATION),
            needed,
            &mut needed,
            GetCurrentProcess(),
            0,
        )
        .as_bool()
        {
            return;
        }
        let mut cpus = Vec::new();
        let mut off = 0usize;
        while off + std::mem::size_of::<SYSTEM_CPU_SET_INFORMATION>() <= needed as usize {
            let info = &*(buf.as_ptr().add(off) as *const SYSTEM_CPU_SET_INFORMATION);
            if info.Type == CpuSetInformation {
                let s = &info.Anonymous.CpuSet;
                cpus.push((s.Id, s.LogicalProcessorIndex, s.EfficiencyClass));
            }
            let step = info.Size as usize;
            if step == 0 {
                break;
            }
            off += step;
        }
        let max_eff = cpus.iter().map(|c| c.2).max().unwrap_or(0);
        let e_cpus: Vec<_> = cpus.into_iter().filter(|c| c.2 < max_eff).collect();
        if e_cpus.is_empty() {
            return;
        }
        let ids: Vec<u32> = e_cpus.iter().map(|c| c.0).collect();
        let mask: usize = e_cpus.iter().fold(0usize, |acc, c| acc | (1usize << (c.1 as usize)));
        let thread = GetCurrentThread();
        let _ = SetThreadSelectedCpuSets(thread, &ids);
        let _ = SetThreadAffinityMask(thread, mask);
    }
}

use crate::expert_pack::{PackIndex, PACK_ALIGN};

/// A 4096-byte-aligned buffer for direct Windows unbuffered I/O.
pub struct AlignedSlotBuf {
    ptr: *mut u8,
    layout: Layout,
    pub len: usize,
}

unsafe impl Send for AlignedSlotBuf {}
unsafe impl Sync for AlignedSlotBuf {}

impl AlignedSlotBuf {
    /// `None` if the allocation fails, so a tight machine gets a smaller cache, not an abort.
    pub fn try_new(len: usize) -> Option<Self> {
        assert!(len % PACK_ALIGN == 0, "slot length must be 4K aligned");
        let layout = Layout::from_size_align(len, PACK_ALIGN).ok()?;
        // SAFETY: `layout` has a non-zero size (slots hold at least one expert).
        let ptr = unsafe { alloc(layout) };
        (!ptr.is_null()).then_some(Self { ptr, layout, len })
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for AlignedSlotBuf {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr, self.layout) }
    }
}

struct CacheSlot {
    layer: usize,
    expert: usize,
    last_used: u64,
    freq: u32,
    buf: AlignedSlotBuf,
}

struct ReadTask {
    miss_idx: usize,
    expert: usize,
    slot_idx: usize,
    offset: u64,
    dst_ptr: usize,
    len: usize,
}

struct WorkerJob {
    miss_idx: usize,
    expert: usize,
    slot_idx: usize,
    offset: u64,
    dst_ptr: usize,
    len: usize,
}

struct ReaderWorkerState {
    jobs: Vec<WorkerJob>,
    remaining: usize,
    expert_rem: Vec<usize>,
    expert_failed: Vec<Option<std::io::Error>>,
    failed: Vec<(usize, std::io::Error)>,
    shutdown: bool,
}

struct ReaderPool {
    state: Arc<(Mutex<ReaderWorkerState>, Condvar)>,
    done_cvar: Arc<Condvar>,
    handles: Vec<JoinHandle<()>>,
}

impl ReaderPool {
    pub fn new(pack_path: &str, flags: u32, n_workers: usize) -> std::io::Result<Self> {
        let state = Arc::new((
            Mutex::new(ReaderWorkerState {
                jobs: Vec::new(),
                remaining: 0,
                expert_rem: Vec::new(),
                expert_failed: Vec::new(),
                failed: Vec::new(),
                shutdown: false,
            }),
            Condvar::new(),
        ));
        let done_cvar = Arc::new(Condvar::new());
        let mut handles = Vec::with_capacity(n_workers);

        for i in 0..n_workers {
            let file = std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(flags)
                .open(pack_path)?;
            let state = Arc::clone(&state);
            let done_cvar = Arc::clone(&done_cvar);

            let handle = thread::Builder::new()
                .name(format!("expert-reader-{i}"))
                .spawn(move || {
                    pin_current_thread_to_ecores();
                    let (lock, cvar) = &*state;
                    loop {
                        let job = {
                            let mut s = lock.lock().unwrap();
                            loop {
                                if s.shutdown {
                                    return;
                                }
                                if let Some(j) = s.jobs.pop() {
                                    break j;
                                }
                                s = cvar.wait(s).unwrap();
                            }
                        };

                        let slice = unsafe { std::slice::from_raw_parts_mut(job.dst_ptr as *mut u8, job.len) };
                        let res = read_exact_at(&file, slice, job.offset);

                        let mut s = lock.lock().unwrap();
                        if let Err(e) = res {
                            if job.miss_idx < s.expert_failed.len() && s.expert_failed[job.miss_idx].is_none() {
                                s.expert_failed[job.miss_idx] = Some(std::io::Error::new(e.kind(), e.to_string()));
                            }
                            s.failed.push((job.expert, e));
                        }
                        s.remaining -= 1;
                        let mut wake = s.remaining == 0;
                        if job.miss_idx < s.expert_rem.len() {
                            s.expert_rem[job.miss_idx] -= 1;
                            if s.expert_rem[job.miss_idx] == 0 {
                                wake = true;
                            }
                        }
                        if wake {
                            done_cvar.notify_one();
                        }
                    }
                })?;
            handles.push(handle);
        }

        Ok(Self { state, done_cvar, handles })
    }

    pub fn start_batch(&self, tasks: &[ReadTask], n_misses: usize) {
        if tasks.is_empty() {
            return;
        }
        let (lock, cvar) = &*self.state;
        let mut s = lock.lock().unwrap();
        s.expert_rem.clear();
        s.expert_rem.resize(n_misses, 0);
        s.expert_failed.clear();
        s.expert_failed.resize_with(n_misses, || None);
        for t in tasks {
            if t.miss_idx < n_misses {
                s.expert_rem[t.miss_idx] += 1;
            }
        }
        // Workers pop from the back of `s.jobs`, so reverse `tasks` so `miss_idx = 0` is
        // popped and completed first for per-expert pipelined compute.
        s.jobs = tasks
            .iter()
            .rev()
            .map(|t| WorkerJob {
                miss_idx: t.miss_idx,
                expert: t.expert,
                slot_idx: t.slot_idx,
                offset: t.offset,
                dst_ptr: t.dst_ptr,
                len: t.len,
            })
            .collect();
        s.remaining = tasks.len();
        s.failed.clear();
        cvar.notify_all();
    }

    pub fn wait_one(&self, miss_idx: usize) -> Option<std::io::Error> {
        let (lock, _) = &*self.state;
        let mut s = lock.lock().unwrap();
        while miss_idx < s.expert_rem.len() && s.expert_rem[miss_idx] > 0 {
            s = self.done_cvar.wait(s).unwrap();
        }
        s.expert_failed.get_mut(miss_idx).and_then(|e| e.take())
    }

    pub fn wait_batch(&self) -> Vec<(usize, std::io::Error)> {
        let (lock, _) = &*self.state;
        let mut s = lock.lock().unwrap();
        while s.remaining > 0 {
            s = self.done_cvar.wait(s).unwrap();
        }
        std::mem::take(&mut s.failed)
    }

    pub fn read_batch(&self, tasks: &[ReadTask]) -> Vec<(usize, std::io::Error)> {
        if tasks.is_empty() {
            return Vec::new();
        }
        let n_misses = tasks.iter().map(|t| t.miss_idx + 1).max().unwrap_or(0);
        self.start_batch(tasks, n_misses);
        self.wait_batch()
    }
}

impl Drop for ReaderPool {
    fn drop(&mut self) {
        {
            let (lock, cvar) = &*self.state;
            let mut s = lock.lock().unwrap();
            s.shutdown = true;
            cvar.notify_all();
        }
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

/// Where one expert's bytes live in the source file: up to three `(file_offset, len)`
/// segments, copied back to back into a slot. A pack has one segment per expert; a GGUF
/// has three (gate, up, down), so the slot ends up with the same layout either way.
pub type ExpertSegments = [(u64, usize); 3];

pub struct ExpertCache {
    slots: Vec<CacheSlot>,
    /// `(layer, expert)` -> index in `slots`
    map: HashMap<(usize, usize), usize>,
    tick: u64,
    /// Persistent LFU sketch across evictions (`[layer * n_expert + expert]`), aged periodically
    /// so cold one-off outliers don't evict proven hot experts (TinyLFU admission).
    expert_freq: Vec<u32>,
    /// `segments[layer * n_expert + expert]`
    segments: Vec<ExpertSegments>,
    pub n_layers: usize,
    pub n_expert: usize,
    reader_pool: ReaderPool,
    pub slot_bytes: usize,
    pub hits: u64,
    pub misses: u64,
    pub bytes_read: u64,
}

impl ExpertCache {
    /// Opens the repacked `.spindle.pack` file and allocates up to `budget_bytes`
    /// of expert slots in RAM.
    pub fn open(pack_path: &str, budget_bytes: usize) -> std::io::Result<Self> {
        let mut meta_file = File::open(pack_path)?;
        let pack_idx = PackIndex::read_from_file(&mut meta_file)?;
        drop(meta_file);

        let slot_bytes = pack_idx.hdr.expert_slot_bytes as usize;
        let segments = pack_idx.offsets.iter().map(|&off| [(off, slot_bytes), (0, 0), (0, 0)]).collect();
        // Pack slots are 4K aligned, so unbuffered I/O is allowed here.
        Self::open_segments(
            pack_path,
            segments,
            pack_idx.hdr.n_layers as usize,
            pack_idx.hdr.n_expert as usize,
            slot_bytes,
            budget_bytes,
            true,
        )
    }

    /// Reads experts straight from the model file, no pack needed: each expert's gate, up
    /// and down slices (`segments`, file offsets) are copied back to back into a slot.
    /// GGUF tensor data is only 32-byte aligned, so this always uses buffered reads.
    pub fn open_gguf(
        model_path: &str,
        segments: Vec<ExpertSegments>,
        n_layers: usize,
        n_expert: usize,
        budget_bytes: usize,
    ) -> std::io::Result<Self> {
        let payload = segments.iter().map(|s| s.iter().map(|x| x.1).sum::<usize>()).max().unwrap_or(0);
        let slot_bytes = (payload + PACK_ALIGN - 1) & !(PACK_ALIGN - 1);
        Self::open_segments(model_path, segments, n_layers, n_expert, slot_bytes, budget_bytes, false)
    }

    fn open_segments(
        path: &str,
        segments: Vec<ExpertSegments>,
        n_layers: usize,
        n_expert: usize,
        slot_bytes: usize,
        budget_bytes: usize,
        allow_direct: bool,
    ) -> std::io::Result<Self> {
        let total_experts = n_layers * n_expert;
        if slot_bytes == 0 || segments.len() != total_experts {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "expert table doesn't match the model shape"));
        }
        let wanted_slots = (budget_bytes / slot_bytes).min(total_experts);
        // Enough slots to hold all active experts a layer picks (plus headroom); capped at
        // 32 so 128-expert models (Qwen3-30B-A3B, 8 active) don't demand 128 slots minimum.
        let min_slots = n_expert.min(32);
        if wanted_slots < min_slots {
            return Err(std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                format!(
                    "budget of {} MiB is below one layer of experts ({} MiB)",
                    budget_bytes >> 20,
                    (min_slots * slot_bytes) >> 20
                ),
            ));
        }

        let mut slots = Vec::with_capacity(wanted_slots);
        for _ in 0..wanted_slots {
            let Some(buf) = AlignedSlotBuf::try_new(slot_bytes) else { break };
            slots.push(CacheSlot { layer: usize::MAX, expert: usize::MAX, last_used: 0, freq: 0, buf });
        }
        if slots.len() < min_slots {
            return Err(std::io::Error::new(std::io::ErrorKind::OutOfMemory, "not enough RAM for expert cache slots"));
        }

        // Buffered random-access I/O by default so misses can hit the OS file cache.
        // Unbuffered direct I/O remains opt-in via SPINDLE_DIRECT_IO=1.
        let direct = allow_direct && std::env::var("SPINDLE_DIRECT_IO").map_or(false, |v| v == "1");
        let flags = if direct {
            (FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN).0
        } else {
            FILE_FLAG_RANDOM_ACCESS.0
        };
        let reader_pool = ReaderPool::new(path, flags, 8)?;

        Ok(Self {
            slots,
            map: HashMap::with_capacity(wanted_slots),
            tick: 0,
            expert_freq: vec![0; total_experts],
            segments,
            n_layers,
            n_expert,
            reader_pool,
            slot_bytes,
            hits: 0,
            misses: 0,
            bytes_read: 0,
        })
    }

    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Private memory held by the slots.
    pub fn bytes(&self) -> usize {
        self.slots.len() * self.slot_bytes
    }

    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            (self.hits as f64) / (total as f64) * 100.0
        }
    }

    /// Returns the pointer to the expert's slab if it is resident in RAM.
    pub fn get_expert_ptr(&self, layer: usize, expert: usize) -> Option<*const u8> {
        self.map.get(&(layer, expert)).map(|&idx| self.slots[idx].buf.as_ptr())
    }

    /// Starts fetching any missing experts for `layer` in the background on the E-core
    /// reader pool, while marking already-resident (hit) experts so they are ready for
    /// immediate compute on the P-cores. Call `finish_layer` after computing the hits.
    ///
    /// Returns `(lookup_us, pending_misses)` where each entry in `pending_misses` is `(expert, slot_idx)`.
    pub fn begin_layer(&mut self, layer: usize, experts: &[usize]) -> (u64, Vec<(usize, usize)>) {
        let t_start = std::time::Instant::now();
        self.tick += 1;
        let now = self.tick;

        // Windowed aging (W-TinyLFU): halve historical frequencies every 32 tokens so topic
        // shifts can replace stale experts while cold 1-off outliers still get filtered.
        let window = (self.n_layers.max(1) as u64) * 32;
        if now % window == 0 {
            for f in &mut self.expert_freq {
                *f = (*f + 1) >> 1;
            }
            for s in &mut self.slots {
                if s.layer != usize::MAX {
                    s.freq = (s.freq + 1) >> 1;
                }
            }
        }

        let mut misses = Vec::with_capacity(experts.len());
        for &e in experts {
            let f_idx = layer * self.n_expert + e;
            self.expert_freq[f_idx] = self.expert_freq[f_idx].saturating_add(1);
            let freq = self.expert_freq[f_idx];
            if let Some(&slot_idx) = self.map.get(&(layer, e)) {
                self.hits += 1;
                let slot = &mut self.slots[slot_idx];
                slot.last_used = now;
                slot.freq = freq;
            } else {
                misses.push(e);
            }
        }

        if misses.is_empty() {
            let lookup_us = t_start.elapsed().as_micros() as u64;
            return (lookup_us, Vec::new());
        }

        // Choose victim slots for each miss without evicting any of the currently active
        // experts. W-TinyLFU admission: if the coldest victim was used recently (within 8
        // tokens) and has a strictly higher frequency than the candidate, keep the hot
        // victim in its slot and serve the cold outlier from the mapped file this token.
        let n_layers_u64 = self.n_layers.max(1) as u64;
        let mut to_fetch = Vec::with_capacity(misses.len());
        let mut reserved_slots = Vec::with_capacity(misses.len());

        for &e in &misses {
            let cand_freq = self.expert_freq[layer * self.n_expert + e];
            let victim_idx = self.choose_victim(layer, now, experts, &reserved_slots);
            let victim = &self.slots[victim_idx];
            let victim_token_age = now.saturating_sub(victim.last_used) / n_layers_u64 + 1;
            if victim.layer != usize::MAX && cand_freq < victim.freq && victim_token_age <= 8 {
                self.misses += 1;
                continue;
            }
            reserved_slots.push(victim_idx);
            to_fetch.push((e, victim_idx));

            if victim.layer != usize::MAX {
                self.map.remove(&(victim.layer, victim.expert));
            }
        }

        if to_fetch.is_empty() {
            let lookup_us = t_start.elapsed().as_micros() as u64;
            return (lookup_us, Vec::new());
        }

        let mut tasks = Vec::with_capacity(to_fetch.len() * 3);
        for (miss_idx, &(e, slot_idx)) in to_fetch.iter().enumerate() {
            let segs = self.segments[layer * self.n_expert + e];
            let base = self.slots[slot_idx].buf.as_mut_slice().as_mut_ptr() as usize;
            let mut at = 0usize;
            for &(off, len) in segs.iter().filter(|s| s.1 > 0) {
                tasks.push(ReadTask { miss_idx, expert: e, slot_idx, offset: off, dst_ptr: base + at, len });
                at += len;
            }
        }

        self.reader_pool.start_batch(&tasks, to_fetch.len());
        let lookup_us = t_start.elapsed().as_micros() as u64;
        (lookup_us, to_fetch)
    }

    /// Waits for `to_fetch[miss_idx]` to finish reading and installs it into the cache map
    /// immediately, so the P-cores can compute `to_fetch[miss_idx]` while `to_fetch[miss_idx+1..]`
    /// continue reading on the E-cores. Returns the microseconds spent waiting.
    pub fn wait_one_miss(&mut self, layer: usize, to_fetch: &[(usize, usize)], miss_idx: usize) -> u64 {
        let now = self.tick;
        let t_read = std::time::Instant::now();
        let failed = self.reader_pool.wait_one(miss_idx);
        let disk_us = t_read.elapsed().as_micros() as u64;

        let (e, slot_idx) = to_fetch[miss_idx];
        self.misses += 1;
        if let Some(err) = failed {
            crate::vlog!("[expert:cache] read failed for layer {layer} expert {e}: {err}; falling back to mmap");
            let slot = &mut self.slots[slot_idx];
            slot.layer = usize::MAX;
            slot.expert = usize::MAX;
            return disk_us;
        }

        self.bytes_read += self.segments[layer * self.n_expert + e].iter().map(|s| s.1 as u64).sum::<u64>();
        let freq = self.expert_freq[layer * self.n_expert + e].max(1);
        let slot = &mut self.slots[slot_idx];
        slot.layer = layer;
        slot.expert = e;
        slot.last_used = now;
        slot.freq = freq;

        self.map.insert((layer, e), slot_idx);
        disk_us
    }

    /// Waits for the in-flight reads started by `begin_layer` to complete and installs the
    /// newly fetched experts into the cache map. Returns `disk_wait_us` (the time spent
    /// waiting after hit-expert compute finished).
    pub fn finish_layer(&mut self, layer: usize, to_fetch: &[(usize, usize)]) -> u64 {
        if to_fetch.is_empty() {
            return 0;
        }
        let now = self.tick;
        let t_read = std::time::Instant::now();
        let failed = self.reader_pool.wait_batch();
        let disk_us = t_read.elapsed().as_micros() as u64;

        let failed_map: HashMap<usize, std::io::Error> = failed.into_iter().collect();

        for &(e, slot_idx) in to_fetch {
            self.misses += 1;
            if let Some(err) = failed_map.get(&e) {
                crate::vlog!("[expert:cache] read failed for layer {layer} expert {e}: {err}; falling back to mmap");
                let slot = &mut self.slots[slot_idx];
                slot.layer = usize::MAX;
                slot.expert = usize::MAX;
                continue;
            }

            self.bytes_read += self.segments[layer * self.n_expert + e].iter().map(|s| s.1 as u64).sum::<u64>();
            let freq = self.expert_freq[layer * self.n_expert + e].max(1);
            let slot = &mut self.slots[slot_idx];
            slot.layer = layer;
            slot.expert = e;
            slot.last_used = now;
            slot.freq = freq;

            self.map.insert((layer, e), slot_idx);
        }

        disk_us
    }

    /// Ensures that all `experts` for `layer` are resident in cache.
    /// Returns `(lookup_us, disk_us)`.
    pub fn ensure_experts(&mut self, layer: usize, experts: &[usize]) -> std::io::Result<(u64, u64)> {
        let (lookup_us, to_fetch) = self.begin_layer(layer, experts);
        let disk_us = self.finish_layer(layer, &to_fetch);
        Ok((lookup_us, disk_us))
    }

    /// Fetches the base pointer of packed expert `(layer, expert)`.
    pub fn get_or_fetch(&mut self, layer: usize, expert: usize) -> std::io::Result<*const u8> {
        self.ensure_experts(layer, &[expert])?;
        self.get_expert_ptr(layer, expert)
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::Other, "expert read failed"))
    }

    /// Chooses an empty slot first, or evicts using fair-share layer balancing and frequency aging.
    fn choose_victim(&self, layer: usize, now: u64, pinned_experts: &[usize], reserved_slots: &[usize]) -> usize {
        // 1. Any empty slot not already reserved?
        for (i, slot) in self.slots.iter().enumerate() {
            if slot.layer == usize::MAX && !reserved_slots.contains(&i) {
                return i;
            }
        }

        // 2. Fair-share per layer: in cyclic transformer scans, Layer 23 should never
        // evict Layer 0's hot repeat experts.
        // Measure age in tokens (passes over all layers), not raw layer ticks: otherwise an
        // expert that skipped 1 token looks 24-48x older than a cold 1-off loaded this layer,
        // and Layer 0 looks 24-48x older than Layer N-1 within the very same token.
        let n_layers = self.n_layers.max(1);
        let token_age = |last: u64| ((now.saturating_sub(last)) / (n_layers as u64) + 1) as f64;
        let fair_share = (self.slots.len() / n_layers).max(1);
        let this_layer_count = self.slots.iter().filter(|s| s.layer == layer).count();

        // If this layer already holds >= fair_share, evict the coldest slot belonging to this layer
        if this_layer_count >= fair_share {
            let mut best_idx = None;
            let mut lowest_score = f64::INFINITY;
            for (i, slot) in self.slots.iter().enumerate() {
                if slot.layer == layer
                    && !pinned_experts.contains(&slot.expert)
                    && !reserved_slots.contains(&i)
                {
                    let age = token_age(slot.last_used);
                    let freq = (slot.freq as f64).powi(2);
                    let score = freq / age;
                    if score < lowest_score {
                        lowest_score = score;
                        best_idx = Some(i);
                    }
                }
            }
            if let Some(idx) = best_idx {
                return idx;
            }
        }

        // 3. Otherwise evict from a colder layer, or global frequency-squared aging
        let mut best_idx = None;
        let mut best_score = f64::INFINITY;

        for (i, slot) in self.slots.iter().enumerate() {
            if reserved_slots.contains(&i) {
                continue;
            }
            if slot.layer == layer && pinned_experts.contains(&slot.expert) {
                continue;
            }
            let age = token_age(slot.last_used);
            let freq = (slot.freq as f64).powi(2);
            let score = freq / age;
            if score < best_score {
                best_score = score;
                best_idx = Some(i);
            }
        }

        best_idx.unwrap_or(0)
    }

    /// Prefetches missing experts for `layer` in a batch.
    pub fn prefetch_batch(&mut self, layer: usize, experts: &[usize]) -> std::io::Result<()> {
        self.ensure_experts(layer, experts)?;
        Ok(())
    }
}

/// `seek_read` may return fewer bytes than asked; keep reading until the slot is full, so a
/// short read can never leave stale bytes from a previous expert in the slot.
fn read_exact_at(file: &File, buf: &mut [u8], mut off: u64) -> std::io::Result<()> {
    let mut done = 0;
    while done < buf.len() {
        let n = file.seek_read(&mut buf[done..], off)?;
        if n == 0 {
            return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "pack file ended inside an expert"));
        }
        done += n;
        off += n as u64;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert_pack::repack_experts;

    #[test]
    fn expert_cache_hits_and_eviction() {
        let n_layers = 2;
        let n_expert = 4;
        let gate_len = 128;
        let up_len = 128;
        let down_len = 256;

        let total_size = 64 * 1024;
        let mut model_bytes = vec![0u8; total_size];
        for (i, b) in model_bytes.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }

        let expert_spans = move |l: usize, e: usize| -> [(usize, usize); 3] {
            let base = (l * 4 + e) * 1024;
            [
                (base, gate_len),
                (base + gate_len, up_len),
                (base + gate_len + up_len, down_len),
            ]
        };

        let temp_pack = std::env::temp_dir().join("test_cache.spindle.pack");
        let pack_path_str = temp_pack.to_str().unwrap();

        repack_experts(
            &model_bytes,
            model_bytes.len() as u64,
            pack_path_str,
            n_layers,
            n_expert,
            &expert_spans,
            |_l, _nl, _cur, _tot| {},
        )
        .unwrap();

        // Open cache with budget for only 4 slots (half the 8 total experts)
        let mut cache = ExpertCache::open(pack_path_str, 4 * 4096).unwrap();
        assert_eq!(cache.capacity(), 4);

        // Fetch (0, 0), (0, 1), (0, 2), (0, 3) -> 4 misses
        for e in 0..4 {
            let ptr = cache.get_or_fetch(0, e).unwrap();
            assert!(!ptr.is_null());
        }
        assert_eq!(cache.misses, 4);
        assert_eq!(cache.hits, 0);

        // Fetch (0, 0) and (0, 1) again -> 2 hits!
        let _ = cache.get_or_fetch(0, 0).unwrap();
        let _ = cache.get_or_fetch(0, 1).unwrap();
        assert_eq!(cache.hits, 2);
        assert_eq!(cache.misses, 4);

        // Fetch (1, 0) -> miss, causes eviction of coldest slot
        let _ = cache.get_or_fetch(1, 0).unwrap();
        assert_eq!(cache.misses, 5);
        assert!(cache.hit_rate() > 0.0);

        drop(cache);
        let _ = std::fs::remove_file(temp_pack);
    }

    #[test]
    fn test_parallel_reads_from_synthetic_pack() {
        let n_layers = 2;
        let n_expert = 8;
        let gate_len = 128;
        let up_len = 128;
        let down_len = 256;
        let expert_len = gate_len + up_len + down_len;

        let total_size = n_layers * n_expert * 1024;
        let mut model_bytes = vec![0u8; total_size];
        for l in 0..n_layers {
            for e in 0..n_expert {
                let base = (l * n_expert + e) * 1024;
                for i in 0..expert_len {
                    model_bytes[base + i] = ((l * 43 + e * 19 + i * 7) % 251) as u8;
                }
            }
        }

        let expert_spans = move |l: usize, e: usize| -> [(usize, usize); 3] {
            let base = (l * n_expert + e) * 1024;
            [
                (base, gate_len),
                (base + gate_len, up_len),
                (base + gate_len + up_len, down_len),
            ]
        };

        let temp_pack = std::env::temp_dir().join("test_parallel_cache.spindle.pack");
        let pack_path_str = temp_pack.to_str().unwrap();

        repack_experts(
            &model_bytes,
            model_bytes.len() as u64,
            pack_path_str,
            n_layers,
            n_expert,
            &expert_spans,
            |_l, _nl, _cur, _tot| {},
        )
        .unwrap();

        // Open cache with budget for 8 slots (each 4096-aligned)
        let mut cache = ExpertCache::open(pack_path_str, 8 * 4096).unwrap();
        assert_eq!(cache.capacity(), 8);

        // Read all 8 experts for layer 0 simultaneously in parallel across the reader pool
        let all_experts: Vec<usize> = (0..8).collect();
        let _ = cache.ensure_experts(0, &all_experts).unwrap();
        assert_eq!(cache.misses, 8);
        assert_eq!(cache.hits, 0);

        // Check EVERY BYTE of every expert in parallel slots
        for e in 0..8 {
            let ptr = cache.get_expert_ptr(0, e).expect("expert must be resident in cache");
            let cached_bytes = unsafe { std::slice::from_raw_parts(ptr, expert_len) };
            let base = e * 1024;
            let expected_bytes = &model_bytes[base..base + expert_len];
            assert_eq!(cached_bytes, expected_bytes, "byte mismatch in layer 0 expert {e}");
        }

        // Test layer 1 experts in parallel
        let _ = cache.ensure_experts(1, &all_experts).unwrap();
        assert_eq!(cache.misses, 16);
        for e in 0..8 {
            let ptr = cache.get_expert_ptr(1, e).expect("expert must be resident in cache");
            let cached_bytes = unsafe { std::slice::from_raw_parts(ptr, expert_len) };
            let base = (1 * n_expert + e) * 1024;
            let expected_bytes = &model_bytes[base..base + expert_len];
            assert_eq!(cached_bytes, expected_bytes, "byte mismatch in layer 1 expert {e}");
        }

        drop(cache);
        let _ = std::fs::remove_file(temp_pack);
    }

    /// GGUF-direct mode: gate, up and down live in three separate tensors (far apart in
    /// the file) and their sizes differ per layer. Each slot must hold gate|up|down back
    /// to back, byte for byte.
    #[test]
    fn gguf_direct_reads_scattered_segments() {
        let (n_layers, n_expert) = (2usize, 4usize);
        let file_bytes: Vec<u8> = (0..256 * 1024).map(|i| ((i * 31 + i / 997) % 253) as u8).collect();
        let path = std::env::temp_dir().join("test_gguf_direct.bin");
        std::fs::write(&path, &file_bytes).unwrap();

        // Tensor regions: gate at 0x1000.., up at 0x10000.., down at 0x20000.., per layer.
        let lens = |l: usize| (300 + l * 17, 300 + l * 17, 700 + l * 2000);
        let mut segs = Vec::new();
        for l in 0..n_layers {
            let (g, u, d) = lens(l);
            for e in 0..n_expert {
                let k = (l * n_expert + e) as u64;
                segs.push([(0x1000 + k * g as u64 + 3, g), (0x10000 + k * u as u64 + 5, u), (0x20000 + k * d as u64 + 7, d)]);
            }
        }
        let mut cache = ExpertCache::open_gguf(path.to_str().unwrap(), segs.clone(), n_layers, n_expert, 64 * 4096).unwrap();
        let all: Vec<usize> = (0..n_expert).collect();
        for l in 0..n_layers {
            cache.ensure_experts(l, &all).unwrap();
            for e in 0..n_expert {
                let s = segs[l * n_expert + e];
                let expected: Vec<u8> = s.iter().flat_map(|&(o, n)| file_bytes[o as usize..o as usize + n].iter().copied()).collect();
                let ptr = cache.get_expert_ptr(l, e).expect("resident");
                let got = unsafe { std::slice::from_raw_parts(ptr, expected.len()) };
                assert_eq!(got, &expected[..], "layer {l} expert {e}");
            }
        }
        assert_eq!(cache.misses, 8);
        drop(cache);
        let _ = std::fs::remove_file(path);
    }
}

