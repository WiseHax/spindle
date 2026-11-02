//! In-RAM Expert Cache for streaming MoE weights from disk.
//!
//! Replaces mmap-demand paging for MoE expert layers with an explicit, bounded RAM cache.
//! Sized to a target memory budget (e.g. 256 MiB–1 GiB), it holds `N` 4096-byte-aligned
//! expert slots. Unbuffered direct reads (`FILE_FLAG_NO_BUFFERING`) fetch missing experts
//! directly into the slots from the repacked `.spindle.pack` file at full NVMe bandwidth.

use std::{
    alloc::{alloc, dealloc, Layout},
    collections::HashMap,
    fs::File,
    os::windows::{
        fs::{FileExt, OpenOptionsExt},
    },
};

use windows::Win32::Storage::FileSystem::{
    FILE_FLAG_NO_BUFFERING, FILE_FLAG_SEQUENTIAL_SCAN,
};

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

pub struct ExpertCache {
    slots: Vec<CacheSlot>,
    /// `(layer, expert)` -> index in `slots`
    map: HashMap<(usize, usize), usize>,
    tick: u64,
    pub pack_idx: PackIndex,
    pack_file: File,
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
        let total_experts = (pack_idx.hdr.n_layers as usize) * (pack_idx.hdr.n_expert as usize);
        let wanted_slots = (budget_bytes / slot_bytes).min(total_experts);
        // One layer's worth of slots is the minimum: every expert a token picks in a layer
        // must be resident at the same time.
        let min_slots = pack_idx.hdr.n_expert as usize;
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

        // Attempt direct unbuffered unthrottled NVMe I/O first
        let direct_flags = (FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN).0;
        let pack_file = match std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(direct_flags)
            .open(pack_path)
        {
            Ok(f) => f,
            Err(_) => {
                // Fallback to buffered sequential I/O if the filesystem does not permit direct I/O
                let fallback_flags = FILE_FLAG_SEQUENTIAL_SCAN.0;
                std::fs::OpenOptions::new()
                    .read(true)
                    .custom_flags(fallback_flags)
                    .open(pack_path)?
            }
        };

        Ok(Self {
            slots,
            map: HashMap::with_capacity(wanted_slots),
            tick: 0,
            pack_idx,
            pack_file,
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

    /// Ensures that all `experts` for `layer` are resident in cache.
    ///
    /// For any misses:
    /// 1. Finds distinct victim slots that don't collide with any of `experts`.
    /// 2. Reads missing expert slabs directly from NVMe into the slots.
    /// 3. Updates cache mappings.
    pub fn ensure_experts(&mut self, layer: usize, experts: &[usize]) -> std::io::Result<()> {
        self.tick += 1;
        let now = self.tick;

        let mut misses = Vec::with_capacity(experts.len());
        for &e in experts {
            if let Some(&slot_idx) = self.map.get(&(layer, e)) {
                self.hits += 1;
                let slot = &mut self.slots[slot_idx];
                slot.last_used = now;
                slot.freq = slot.freq.saturating_add(1);
            } else {
                misses.push(e);
            }
        }

        if misses.is_empty() {
            return Ok(());
        }

        // Choose victim slots for each miss without evicting any of the currently active experts
        let mut to_fetch = Vec::with_capacity(misses.len());
        let mut reserved_slots = Vec::with_capacity(misses.len());

        for &e in &misses {
            let victim_idx = self.choose_victim(layer, now, experts, &reserved_slots);
            reserved_slots.push(victim_idx);
            to_fetch.push((e, victim_idx));

            let victim = &self.slots[victim_idx];
            if victim.layer != usize::MAX {
                self.map.remove(&(victim.layer, victim.expert));
            }
        }

        // Read missing experts directly into their designated slots
        for &(e, slot_idx) in &to_fetch {
            self.misses += 1;
            let off = self.pack_idx.offset(layer, e);
            read_exact_at(&self.pack_file, self.slots[slot_idx].buf.as_mut_slice(), off)?;
            self.bytes_read += self.slot_bytes as u64;

            let slot = &mut self.slots[slot_idx];
            slot.layer = layer;
            slot.expert = e;
            slot.last_used = now;
            slot.freq = 1;

            self.map.insert((layer, e), slot_idx);
        }

        Ok(())
    }

    /// Fetches the base pointer of packed expert `(layer, expert)`.
    pub fn get_or_fetch(&mut self, layer: usize, expert: usize) -> std::io::Result<*const u8> {
        self.ensure_experts(layer, &[expert])?;
        Ok(self.get_expert_ptr(layer, expert).expect("expert must be resident after ensure_experts"))
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
        let n_layers = (self.pack_idx.hdr.n_layers as usize).max(1);
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
                    let age = (now.saturating_sub(slot.last_used) + 1) as f64;
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
            let age = (now.saturating_sub(slot.last_used) + 1) as f64;
            let freq = (slot.freq as f64).powi(2);
            let score = freq / age;
            if score < best_score {
                best_score = score;
                best_idx = Some(i);
            }
        }

        best_idx.expect("expert cache: failed to find an evictable slot")
    }

    /// Prefetches missing experts for `layer` in a batch.
    pub fn prefetch_batch(&mut self, layer: usize, experts: &[usize]) -> std::io::Result<()> {
        self.ensure_experts(layer, experts)
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
}

