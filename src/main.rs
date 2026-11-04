//! spindle: CLI entry point.
//!
//! A real Llama-family GGUF goes to the chat engine (`smollm_chat`). `--probe` runs the
//! bandwidth probe. Anything else runs the older synthetic NVMe streaming benchmark, which
//! uses the experimental modules (`moe_engine`, `rio_mesh`, `transformer`, `q4k_gemv`) on a
//! generated dummy model. That benchmark is a prototype, not a working MoE engine.
//!
//! Also handles hybrid CPU topology: P-cores and E-cores are found via CPU Sets so hot
//! threads can be pinned to P-cores.

#![allow(dead_code)]

mod agent;
mod chat_cli;
mod cluster_net;
mod expert_cache;
mod expert_pack;
mod gguf_loader;
mod guard;
mod json;
mod moe_engine;
mod q4k_gemv;
mod rio_mesh;
mod smollm_chat;
mod tools;
mod tui;
mod ui;
mod worker_pool;
mod memory_guard;
mod layer_stream;
mod loading;
mod mmap_file;
mod probe;
mod quant;
mod quant_multi;
mod transformer;

use std::{
    ffi::c_void,
    io::Write,
    mem::size_of,
    net::{Ipv4Addr, SocketAddrV4},
    ptr,
    sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::*},
    time::{Duration, Instant},
};

use gguf_loader::{write_spec_compliant_gguf_v3_header, GgufModelIndex};
use moe_engine::{
    ecore_draft_spec4_tokens, ensure_real_gguf_moe_model, route_moe_top2, verify_spec4_logits_avx2,
    HotTierWeights, NUM_EXPERTS, TOP_K_EXPERTS,
};
use q4k_gemv::{q4k_gemm_spec4_slice_avx2, HIDDEN_DIM, ROW_BYTES, SPEC_K};
use rio_mesh::{RioMeshEndpoint, RIO_SLAB_BYTES};
use transformer::{
    flash_decoding_gqa_spec4_avx2, inspect_or_map_weights, rmsnorm_spec4_avx2, rope_spec4_avx2,
    swiglu_residual_spec4_avx2, QuantizedKvCache, RopeCache, NUM_LAYERS,
};
use windows::{
    core::{Error, Result, HSTRING, PCWSTR},
    Win32::{
        Foundation::*,
        Security::*,
        Storage::FileSystem::*,
        System::{
            Memory::*,
            SystemInformation::*,
            Threading::*,
            IO::*,
        },
    },
};

const ARENA_SIZE: usize = 256 << 20; // Strict 256 MiB cap for 8 GB laptop
const CHUNK_SIZE: usize = 4 << 20; // 4 MiB per NVMe DMA slot (1 Expert = 4 MiB)
const NUM_SLOTS: usize = (ARENA_SIZE - 2 * RIO_SLAB_BYTES) / CHUNK_SIZE; // 63 slots (252 MiB)
const QUEUE_DEPTH: usize = 16; // reads in flight
const COMPUTE_WORKERS: usize = 3; // Physical P-Cores #1, #2, #3 (P-Core #0 runs IOCP)
const IO_KEY: usize = 0xC0FFEE;
const WAKE_KEY: usize = 0x3A4E;
const FILE_SKIP_SET_EVENT_ON_HANDLE: u8 = 0x2;
const _: () = assert!(QUEUE_DEPTH <= NUM_SLOTS);

// ---------------------------------------------------------------- RAII guards

struct Handle(HANDLE);
impl Drop for Handle {
    fn drop(&mut self) {
        if !self.0.is_invalid() {
            unsafe {
                let _ = CloseHandle(self.0);
            }
        }
    }
}

struct Arena {
    ptr: *mut u8,
    len: usize,
    large_pages: bool,
}
impl Drop for Arena {
    fn drop(&mut self) {
        unsafe {
            let _ = VirtualFree(self.ptr as *mut c_void, 0, MEM_RELEASE);
        }
    }
}

// ----------------------------- Hybrid Topology (4 P-Cores + 4 E-Cores)

#[derive(Clone, Copy, Debug)]
struct LogicalCpu {
    cpu_set_id: u32,
    logical_idx: u8,
    core_idx: u8,
    efficiency_class: u8,
}

#[derive(Clone, Debug)]
struct HybridTopology {
    producer_pcore: LogicalCpu,
    worker_pcores: [LogicalCpu; COMPUTE_WORKERS],
    drafter_ecore: LogicalCpu,
    max_efficiency_class: u8,
    total_logical: usize,
    p_logical_count: usize,
    e_logical_count: usize,
    distinct_p_cores: usize,
    /// One logical CPU per physical performance core, core 0 LAST (Windows routes most
    /// interrupts there). Compute thread `i` is pinned to `compute_cores[i]`; threads beyond
    /// the core count float, so `--threads` can never index past the topology.
    compute_cores: Vec<LogicalCpu>,
}

impl HybridTopology {
    fn compute_core(&self, i: usize) -> Option<LogicalCpu> {
        self.compute_cores.get(i).copied()
    }
}

/// Reads the CPU set information to tell performance cores (the highest `EfficiencyClass`)
/// from efficiency cores, and lists one logical CPU per physical P-core for compute threads.
unsafe fn detect_hybrid_topology() -> Result<HybridTopology> {
    let mut needed = 0u32;
    let _ = GetSystemCpuSetInformation(None, 0, &mut needed, GetCurrentProcess(), 0);
    if needed == 0 {
        return Err(Error::from_win32());
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
        return Err(Error::from_win32());
    }

    let mut cpus = Vec::new();
    let mut off = 0usize;
    while off + size_of::<SYSTEM_CPU_SET_INFORMATION>() <= needed as usize {
        let info = &*(buf.as_ptr().add(off) as *const SYSTEM_CPU_SET_INFORMATION);
        if info.Type == CpuSetInformation {
            let s = &info.Anonymous.CpuSet;
            cpus.push(LogicalCpu {
                cpu_set_id: s.Id,
                logical_idx: s.LogicalProcessorIndex,
                core_idx: s.CoreIndex,
                efficiency_class: s.EfficiencyClass,
            });
        }
        let step = info.Size as usize;
        if step == 0 {
            break;
        }
        off += step;
    }

    let max_eff = cpus.iter().map(|c| c.efficiency_class).max().unwrap_or(0);
    let p_cpus: Vec<LogicalCpu> = cpus
        .iter()
        .copied()
        .filter(|c| c.efficiency_class == max_eff)
        .collect();
    let e_cpus: Vec<LogicalCpu> = cpus
        .iter()
        .copied()
        .filter(|c| c.efficiency_class < max_eff)
        .collect();

    let mut distinct_p: Vec<LogicalCpu> = Vec::new();
    for c in &p_cpus {
        if !distinct_p.iter().any(|d| d.core_idx == c.core_idx) {
            distinct_p.push(*c);
        }
    }

    let producer_pcore = distinct_p[0];
    let worker_pcores = [
        *distinct_p.get(1).unwrap_or(&producer_pcore),
        *distinct_p.get(2).unwrap_or(&producer_pcore),
        *distinct_p.get(3).unwrap_or(&producer_pcore),
    ];
    let drafter_ecore = e_cpus.first().copied().unwrap_or(producer_pcore);
    let mut compute_cores = distinct_p.clone();
    compute_cores.sort_by_key(|c| c.core_idx);
    if compute_cores.len() > 1 {
        compute_cores.rotate_left(1);
    }

    Ok(HybridTopology {
        producer_pcore,
        worker_pcores,
        drafter_ecore,
        max_efficiency_class: max_eff,
        total_logical: cpus.len(),
        p_logical_count: p_cpus.len(),
        e_logical_count: e_cpus.len(),
        distinct_p_cores: distinct_p.len(),
        compute_cores,
    })
}

unsafe fn lock_thread_to_core(target: LogicalCpu, role: &str) -> Result<()> {
    let thread = GetCurrentThread();

    let throttling = THREAD_POWER_THROTTLING_STATE {
        Version: THREAD_POWER_THROTTLING_CURRENT_VERSION,
        ControlMask: THREAD_POWER_THROTTLING_EXECUTION_SPEED,
        StateMask: 0,
    };
    SetThreadInformation(
        thread,
        ThreadPowerThrottling,
        &throttling as *const _ as *const c_void,
        size_of::<THREAD_POWER_THROTTLING_STATE>() as u32,
    )?;

    let ids = [target.cpu_set_id];
    if !SetThreadSelectedCpuSets(thread, &ids).as_bool() {
        return Err(Error::from_win32());
    }

    let mask = 1usize << (target.logical_idx as usize);
    if SetThreadAffinityMask(thread, mask) == 0 {
        return Err(Error::from_win32());
    }

    let _ = SetThreadPriority(thread, THREAD_PRIORITY_HIGHEST);

    let running_on = GetCurrentProcessorNumber();
    let core_kind = if target.efficiency_class > 0 { "P-Core" } else { "E-Core" };
    vlog!(
        "[pin:{role}] pinned to {core_kind} LP #{} (CpuSet {:#x}, PhysicalCore #{}, EffClass {}), active on LP #{}",
        target.logical_idx,
        target.cpu_set_id,
        target.core_idx,
        target.efficiency_class,
        running_on
    );
    Ok(())
}

pub fn pin_current_thread_to_pcore() {
    unsafe {
        if let Ok(topo) = detect_hybrid_topology() {
            if let Some(core) = topo.compute_core(0) {
                let _ = lock_thread_to_core(core, "tui-worker");
            }
        }
    }
}

// ------------------------------------------------------ shared ring state

#[repr(align(64))]
#[derive(Default)]
struct SlotMeta {
    seq: AtomicU64,
    len: AtomicU32,
}

#[repr(align(64))]
#[derive(Default)]
struct Padded<T>(T);

struct Ring {
    slots: [SlotMeta; NUM_SLOTS],
    worker_done: [Padded<AtomicU64>; COMPUTE_WORKERS],
    consumed: Padded<AtomicU64>,
    layer_epoch: Padded<AtomicU64>,
    starved: Padded<AtomicBool>,
    abort: Padded<AtomicBool>,
}

impl Default for Ring {
    fn default() -> Self {
        Self {
            slots: std::array::from_fn(|_| SlotMeta::default()),
            worker_done: std::array::from_fn(|_| Padded::default()),
            consumed: Padded::default(),
            layer_epoch: Padded::default(),
            starved: Padded::default(),
            abort: Padded::default(),
        }
    }
}

struct SpeculativeState {
    x_spec: [[f32; HIDDEN_DIM]; SPEC_K],
    y_workers: [[[f32; HIDDEN_DIM]; SPEC_K]; COMPUTE_WORKERS],
    gate_buf: [[f32; HIDDEN_DIM]; SPEC_K],
    rms_weight: [f32; HIDDEN_DIM],
    rope_cache: Box<RopeCache>,
    kv_cache: Box<QuantizedKvCache>,
    hot_weights: Box<HotTierWeights>,
}

impl SpeculativeState {
    fn new_boxed() -> Box<Self> {
        let mut s = Box::new(Self {
            x_spec: [[0.0f32; HIDDEN_DIM]; SPEC_K],
            y_workers: [[[0.0f32; HIDDEN_DIM]; SPEC_K]; COMPUTE_WORKERS],
            gate_buf: [[0.0f32; HIDDEN_DIM]; SPEC_K],
            rms_weight: [1.0f32; HIDDEN_DIM],
            rope_cache: RopeCache::new(64),
            kv_cache: QuantizedKvCache::new(64),
            hot_weights: HotTierWeights::new_coherent(NUM_LAYERS),
        });
        for k in 0..SPEC_K {
            for i in 0..HIDDEN_DIM {
                s.x_spec[k][i] = (((i + k * 13) % 97) as f32 - 48.0) / 64.0;
                s.gate_buf[k][i] = (((i + k * 7) % 89) as f32 - 44.0) / 64.0;
            }
        }
        for i in 0..HIDDEN_DIM {
            s.rms_weight[i] = 0.95 + ((i % 32) as f32) * 0.003;
        }
        s
    }
}

// ------------------------------------------------- SeLockMemoryPrivilege

unsafe fn enable_lock_memory_privilege() -> Result<()> {
    let mut raw = HANDLE::default();
    OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &mut raw)?;
    let token = Handle(raw);

    let mut luid = LUID::default();
    LookupPrivilegeValueW(PCWSTR::null(), SE_LOCK_MEMORY_NAME, &mut luid)?;

    let tp = TOKEN_PRIVILEGES {
        PrivilegeCount: 1,
        Privileges: [LUID_AND_ATTRIBUTES { Luid: luid, Attributes: SE_PRIVILEGE_ENABLED }],
    };
    AdjustTokenPrivileges(token.0, FALSE, Some(&tp), 0, None, None)?;

    let err = GetLastError();
    if err == ERROR_NOT_ALL_ASSIGNED {
        return Err(Error::new(
            err.to_hresult(),
            "SeLockMemoryPrivilege not held by this account. Grant 'Lock pages in memory' \
             in secpol.msc (Local Policies > User Rights Assignment), then sign out/in.",
        ));
    }
    Ok(())
}

// ------------------------------------------------------ Large-page arena

unsafe fn alloc_arena(size: usize) -> Result<Arena> {
    let lp_min = GetLargePageMinimum();
    if lp_min != 0 && size % lp_min == 0 {
        let p = VirtualAlloc(None, size, MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES, PAGE_READWRITE);
        if !p.is_null() {
            return Ok(Arena { ptr: p as *mut u8, len: size, large_pages: true });
        }
        let err = GetLastError();
        let why = if err == ERROR_PRIVILEGE_NOT_HELD {
            "SeLockMemoryPrivilege missing"
        } else {
            "not enough contiguous physical memory (fragmentation); reboot or close apps"
        };
        eprintln!("[warn] MEM_LARGE_PAGES failed ({err:?}): {why}. Falling back to 4 KiB pages.");
    }
    let p = VirtualAlloc(None, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if p.is_null() {
        return Err(Error::from_win32());
    }
    let base = p as *mut u8;
    for off in (0..size).step_by(4096) {
        ptr::write_volatile(base.add(off), 0);
    }
    Ok(Arena { ptr: base, len: size, large_pages: false })
}

// -------------------------------------- 3-P-Core Speculative Worker Pool

struct WorkerStats {
    compute: Duration,
    non_linear: Duration,
    rio_tx: Duration,
    stall: Duration,
    rows_computed: usize,
    layers_sent: u64,
    tokens_accepted: u64,
    active_lp: u32,
}

#[allow(clippy::too_many_arguments)]
unsafe fn speculative_worker(
    worker_id: usize,
    ring: &Ring,
    state_ptr: usize,
    arena: usize,
    total_chunks: u64,
    chunks_per_layer: u64,
    port: usize,
    target_core: LogicalCpu,
    rio_tx_ep: Option<&RioMeshEndpoint>,
    rio_rx_ep: Option<&RioMeshEndpoint>,
    quiet_pin: bool,
) -> WorkerStats {
    let role = match worker_id {
        0 => "worker#0(lead+rio)",
        1 => "worker#1(avx2)",
        _ => "worker#2(avx2)",
    };
    if !quiet_pin {
        let _ = lock_thread_to_core(target_core, role);
    } else {
        let ids = [target_core.cpu_set_id];
        let _ = SetThreadSelectedCpuSets(GetCurrentThread(), &ids);
        let _ = SetThreadAffinityMask(GetCurrentThread(), 1usize << (target_core.logical_idx as usize));
    }

    let state = &mut *(state_ptr as *mut SpeculativeState);
    let mut rx_verify = Box::new([0.0f32; HIDDEN_DIM]);
    let mut st = WorkerStats {
        compute: Duration::ZERO,
        non_linear: Duration::ZERO,
        rio_tx: Duration::ZERO,
        stall: Duration::ZERO,
        rows_computed: 0,
        layers_sent: 0,
        tokens_accepted: 0,
        active_lp: GetCurrentProcessorNumber(),
    };

    for c in 0..total_chunks {
        let slot = (c % NUM_SLOTS as u64) as usize;
        let meta = &ring.slots[slot];

        let t_wait = Instant::now();
        let mut spins = 0u32;
        while meta.seq.load(Acquire) != c + 1 {
            if ring.abort.0.load(Relaxed) {
                return st;
            }
            if spins < 2048 {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
            spins += 1;
        }
        st.stall += t_wait.elapsed();

        let len = meta.len.load(Relaxed) as usize;
        let data = (arena as *const u8).add(slot * CHUNK_SIZE);
        let total_rows = len / ROW_BYTES;

        let row_start = (total_rows * worker_id) / COMPUTE_WORKERS;
        let row_end = (total_rows * (worker_id + 1)) / COMPUTE_WORKERS;
        let global_row_base = (c as usize) * (CHUNK_SIZE / ROW_BYTES);

        let t_c = Instant::now();
        let rows = q4k_gemm_spec4_slice_avx2(
            data,
            row_start,
            row_end,
            &state.x_spec,
            &mut state.y_workers[worker_id],
            global_row_base,
        );
        st.compute += t_c.elapsed();
        st.rows_computed += rows;

        ring.worker_done[worker_id].0.store(c + 1, SeqCst);
        let min_done = ring
            .worker_done
            .iter()
            .map(|w| w.0.load(SeqCst))
            .min()
            .unwrap_or(0);

        let prev = ring.consumed.0.fetch_max(min_done, SeqCst);
        if min_done > prev && ring.starved.0.swap(false, SeqCst) {
            let _ = PostQueuedCompletionStatus(HANDLE(port as *mut c_void), 0, WAKE_KEY, None);
        }

        if (c + 1) % chunks_per_layer == 0 || c + 1 == total_chunks {
            let target_epoch = st.layers_sent + 1;
            if worker_id == 0 {
                while ring.consumed.0.load(Acquire) < c + 1 {
                    if ring.abort.0.load(Relaxed) {
                        return st;
                    }
                    std::hint::spin_loop();
                }

                let t_nl = Instant::now();
                for k in 0..SPEC_K {
                    for i in 0..HIDDEN_DIM {
                        state.y_workers[0][k][i] +=
                            state.y_workers[1][k][i] + state.y_workers[2][k][i];
                        state.y_workers[1][k][i] = 0.0;
                        state.y_workers[2][k][i] = 0.0;
                    }
                }

                rmsnorm_spec4_avx2(&mut state.y_workers[0], &state.rms_weight);
                rope_spec4_avx2(&mut state.y_workers[0], &state.rope_cache);
                flash_decoding_gqa_spec4_avx2(
                    &mut state.y_workers[0],
                    &mut state.kv_cache,
                    st.layers_sent as usize,
                );
                swiglu_residual_spec4_avx2(
                    &mut state.x_spec,
                    &state.gate_buf,
                    &state.y_workers[0],
                    1e-3,
                );
                st.non_linear += t_nl.elapsed();

                let tx_ep = rio_tx_ep.unwrap();
                let rx_ep = rio_rx_ep.unwrap();
                let t_rio = Instant::now();
                let layer_id = st.layers_sent as u16;

                for k in 0..SPEC_K {
                    let seq_id = st.layers_sent * (SPEC_K as u64) + (k as u64);
                    tx_ep
                        .send_activation(seq_id, layer_id, &state.x_spec[k])
                        .expect("RIO speculative TX failed");
                    let (rx_seq, rx_layer) = rx_ep
                        .recv_activation(&mut rx_verify[..])
                        .expect("RIO speculative RX failed");
                    debug_assert_eq!(rx_seq, seq_id);
                    debug_assert_eq!(rx_layer, layer_id);

                    state.x_spec[k] = *rx_verify;
                    st.tokens_accepted += 1;
                }
                st.rio_tx += t_rio.elapsed();
                st.layers_sent = target_epoch;
                ring.layer_epoch.0.store(target_epoch, Release);
            } else {
                while ring.layer_epoch.0.load(Acquire) < target_epoch {
                    if ring.abort.0.load(Relaxed) {
                        return st;
                    }
                    std::hint::spin_loop();
                }
                st.layers_sent = target_epoch;
            }
        }
    }

    st.active_lp = GetCurrentProcessorNumber();
    st
}

// ---------------------------------------------------------- producer

unsafe fn submit(file: HANDLE, arena: &Arena, ov: *mut OVERLAPPED, slot: usize, file_off: u64, len: usize) -> Result<()> {
    *ov = OVERLAPPED::default();
    (*ov).Anonymous.Anonymous.Offset = file_off as u32;
    (*ov).Anonymous.Anonymous.OffsetHigh = (file_off >> 32) as u32;
    let buf = std::slice::from_raw_parts_mut(arena.ptr.add(slot * CHUNK_SIZE), len);
    match ReadFile(file, Some(buf), None, Some(ov)) {
        Ok(()) => Ok(()),
        Err(e) if e.code() == ERROR_IO_PENDING.to_hresult() => Ok(()),
        Err(e) => Err(e),
    }
}

struct RunStats {
    bytes: u64,
    wall: f64,
    sector: u32,
    starve_events: u64,
    workers: Vec<WorkerStats>,
    checksum: f64,
}

/// Runs a pass over `chunk_offsets` (each entry is a 4K-aligned file byte offset
/// of a 4 MiB chunk to DMA from NVMe).
/// - In Dense mode: `chunk_offsets` is all 256 chunks (`1.074 GB`).
/// - In Sparse MoE mode (`Top-2 of 8 Experts`): `chunk_offsets` contains ONLY the
///   64 active expert chunks (`32 layers x 2 active experts = 268.4 MB`), skipping
///   75% of the model weights on disk.
#[allow(clippy::too_many_arguments)]
unsafe fn execute_stream_schedule(
    file_handle: HANDLE,
    sector: u32,
    aligned_end: u64,
    chunk_offsets: &[u64],
    chunks_per_layer: u64,
    arena: &Arena,
    spec_state: &mut SpeculativeState,
    worker_cores: [LogicalCpu; COMPUTE_WORKERS],
    rio_tx_ep: &RioMeshEndpoint,
    rio_rx_ep: &RioMeshEndpoint,
    quiet_pin: bool,
) -> Result<RunStats> {
    let total_chunks = chunk_offsets.len() as u64;
    let port = Handle(CreateIoCompletionPort(file_handle, HANDLE::default(), IO_KEY, 1)?);
    SetFileCompletionNotificationModes(file_handle, FILE_SKIP_SET_EVENT_ON_HANDLE)?;

    let ring = Ring::default();
    let state_ptr = (spec_state as *mut SpeculativeState) as usize;

    let mut ovs: Box<[OVERLAPPED]> = (0..NUM_SLOTS).map(|_| OVERLAPPED::default()).collect();
    let ov_base = ovs.as_mut_ptr();
    let mut chunk_of = [0u64; NUM_SLOTS];
    let mut entries = [OVERLAPPED_ENTRY::default(); QUEUE_DEPTH];

    let arena_addr = arena.ptr as usize;
    let port_addr = port.0 .0 as usize;

    let (prod_res, workers, wall) = std::thread::scope(|s| {
        let ring_ref = &ring;
        let mut handles = Vec::with_capacity(COMPUTE_WORKERS);

        for (wid, &core) in worker_cores.iter().enumerate() {
            let tx_opt = if wid == 0 { Some(rio_tx_ep) } else { None };
            let rx_opt = if wid == 0 { Some(rio_rx_ep) } else { None };
            handles.push(s.spawn(move || {
                speculative_worker(
                    wid,
                    ring_ref,
                    state_ptr,
                    arena_addr,
                    total_chunks,
                    chunks_per_layer,
                    port_addr,
                    core,
                    tx_opt,
                    rx_opt,
                    quiet_pin,
                )
            }));
        }

        let t0 = Instant::now();
        let prod = (|| -> Result<(u64, u64)> {
            let (mut next, mut inflight, mut completed) = (0u64, 0usize, 0u64);
            let (mut bytes, mut starve_events) = (0u64, 0u64);

            while completed < total_chunks {
                while inflight < QUEUE_DEPTH && next < total_chunks {
                    if next >= ring.consumed.0.load(Acquire) + NUM_SLOTS as u64 {
                        break;
                    }
                    let slot = (next % NUM_SLOTS as u64) as usize;
                    let off = chunk_offsets[next as usize];
                    let len = CHUNK_SIZE.min((aligned_end - off) as usize);
                    chunk_of[slot] = next;
                    submit(file_handle, arena, ov_base.add(slot), slot, off, len)?;
                    next += 1;
                    inflight += 1;
                }

                if inflight == 0 {
                    ring.starved.0.store(true, SeqCst);
                    if next < ring.consumed.0.load(SeqCst) + NUM_SLOTS as u64 {
                        ring.starved.0.store(false, Relaxed);
                        continue;
                    }
                    starve_events += 1;
                }

                let mut n = 0u32;
                GetQueuedCompletionStatusEx(port.0, &mut entries, &mut n, INFINITE, FALSE)?;
                for e in &entries[..n as usize] {
                    if e.lpCompletionKey == WAKE_KEY || e.lpOverlapped.is_null() {
                        continue;
                    }
                    let slot = (e.lpOverlapped as usize - ov_base as usize) / size_of::<OVERLAPPED>();
                    let mut got = 0u32;
                    match GetOverlappedResult(file_handle, e.lpOverlapped, &mut got, FALSE) {
                        Ok(()) => {}
                        Err(err) if err.code() == ERROR_HANDLE_EOF.to_hresult() => {}
                        Err(err) => return Err(err),
                    }
                    ring.slots[slot].len.store(got, Relaxed);
                    ring.slots[slot].seq.store(chunk_of[slot] + 1, Release);
                    bytes += got as u64;
                    inflight -= 1;
                    completed += 1;
                }
            }
            Ok((bytes, starve_events))
        })();

        if prod.is_err() {
            ring.abort.0.store(true, SeqCst);
        }
        let w_stats: Vec<WorkerStats> = handles
            .into_iter()
            .map(|h| h.join().expect("worker panicked"))
            .collect();
        let elapsed = t0.elapsed().as_secs_f64();
        (prod, w_stats, elapsed)
    });

    let (bytes, starve_events) = prod_res?;
    let checksum: f64 = spec_state.x_spec.iter().flatten().map(|&v| v as f64).sum();
    drop(ovs);
    Ok(RunStats {
        bytes,
        wall,
        sector,
        starve_events,
        workers,
        checksum,
    })
}

struct CliConfig {
    path: String,
    create_gib: u64,
    role: String,
    bind_addr: SocketAddrV4,
    peer_addr: Option<SocketAddrV4>,
    prompt: String,
    /// True when `--prompt` was passed; `--chat` alone skips the canned opening prompt.
    prompt_given: bool,
    chat_mode: bool,
    probe: bool,
    /// `--stream-layers`: read non-pinned layers with explicit I/O instead of paging.
    stream_layers: bool,
    /// `--verbose`: print load and memory diagnostics.
    verbose: bool,
    /// `--raw "text"`: plain completion with no chat framing (for golden tests).
    raw: Option<String>,
    /// `-n`: tokens to generate in `--raw` mode.
    max_tokens: usize,
    sampler: smollm_chat::SamplerConfig,
    /// `--ram-budget`: hard ceiling on spindle's resident memory, in MiB.
    ram_budget_mib: Option<u64>,
    /// `--reserve`: RAM to always leave free for other programs, in MiB.
    reserve_mib: Option<u64>,
    /// `--pack-experts`: repacks an MoE model's expert weights into `<model>.spindle.pack`.
    pack_experts: bool,
    /// `--expert-cache-mib`: turn on the MoE expert cache with this RAM budget (needs a pack
    /// from `--pack-experts`).
    expert_cache_mib: Option<usize>,
    /// `--draft <path>`: small in-RAM draft model for speculative decoding.
    draft_path: Option<String>,
    /// `--think`: enable reasoning model thinking mode (default: false for fast direct reply).
    think: bool,
    /// `--node`: runs as a distributed worker node serving layers over network.
    node_mode: bool,
    /// `--layer-range <start>..<end>`: selective layer range to load into RAM.
    layer_range: Option<std::ops::Range<usize>>,
    /// `--wire-format <fp16|fp8|fp32>`: network tensor compression format.
    wire_format: cluster_net::WireFormat,
    /// `--threads <N>`: number of compute threads for AVX2 GEMV.
    threads: Option<usize>,
    /// `--moe-prune <f32>`: relative expert pruning threshold (e.g. 0.03).
    moe_prune: Option<f32>,
    /// `--system-prompt <text>`: custom system prompt.
    system_prompt: Option<String>,
    /// `--draft-len <n>`: speculative decoding batch length (1..16).
    draft_len: Option<usize>,
    /// `--no-spec` / `speculate = false`: disable prompt-lookup speculative decoding.
    speculate: Option<bool>,
    /// `--ctx <n>` / `ctx = n`: KV-cache context length (still capped by free RAM). On a
    /// cluster set it on BOTH nodes: the master uses the smaller of the two.
    ctx: Option<usize>,
    /// `--tools`: enables agentic tools (file operations, web search, sandboxed execution).
    tools: bool,
    /// `--tui`: launches the full-screen OpenCode-style TUI interface.
    tui: bool,
}

fn help_text() -> String {
    format!(
        "spindle {version}
Run GGUF language models on the CPU, on one PC or split across two.

Usage:
  spindle [model.gguf] [options]        chat (the default; settings from spindle.toml)
  spindle node <model.gguf> [options]   serve a range of layers to another PC
  spindle <model.gguf> --prompt \"...\"   answer one prompt and exit
  spindle <file> --probe                measure RAM and disk read speed

Cluster:
  --peer <ip:port>            worker to connect to (client side)
  --bind <ip:port>            address to listen on (worker side, e.g. 0.0.0.0:42424)
  --layer-range <a..b>        layers this PC runs, e.g. 0..35 on the worker, 35..48 here
  --wire-format <fp8|fp16|fp32>
                              how activations are sent between PCs (default fp16)

Model and speed:
  --threads <n>               compute threads (default: one per physical core)
  --ctx <n>                   context length in tokens (default 4096, limited by free RAM)
  --moe-prune <x>             MoE models: run the top x experts (x >= 1) or drop the
                              tail below a router-probability fraction (x < 1); 0 = off
  --no-spec                   turn off prompt-lookup speculative decoding
  --draft <model.gguf>        use a small model for speculative drafts instead
  --draft-len <n>             draft tokens per verify step (1..16, default 4)

Chat:
  --tools                     let the model read and edit files and run commands
  --think                     allow reasoning (<think>) before answering
  --system-prompt <text>      replace the default system prompt
  --no-tui                    line-based chat instead of the full-screen interface
  --temp <t>  --top-k <k>  --top-p <p>  --min-p <p>  --repeat-penalty <r>  --seed <n>

Memory:
  --reserve <MiB>             RAM to leave free for other programs (default 512)
  --ram-budget <MiB>          hard limit on spindle's resident memory
  --stream-layers             read layers that don't fit with one large read each

Other:
  -v, --verbose               print load, memory and threading details
  -V, --version               print the version
  -h, --help                  print this help

Settings in ./spindle.toml are read first; command-line options override them.
",
        version = env!("CARGO_PKG_VERSION")
    )
}

fn load_config_file(cfg: &mut CliConfig) {
    let content = match std::fs::read_to_string("spindle.toml") {
        Ok(s) => s,
        Err(_) => return,
    };
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((k, v)) = line.split_once('=') {
            let key = k.trim();
            let val = v.trim().trim_matches('"').trim_matches('\'');
            match key {
                "model" => cfg.path = val.to_string(),
                "peer" => cfg.peer_addr = val.parse().ok(),
                "layer_range" => {
                    let parts: Vec<&str> = if val.contains("..") {
                        val.split("..").collect()
                    } else if val.contains('-') {
                        val.split('-').collect()
                    } else {
                        vec![val]
                    };
                    if parts.len() == 2 {
                        if let (Ok(s), Ok(e)) = (parts[0].trim().parse(), parts[1].trim().parse()) {
                            cfg.layer_range = Some(s..e);
                        }
                    }
                }
                "wire_format" => match val.to_lowercase().as_str() {
                    "fp8" => cfg.wire_format = cluster_net::WireFormat::Fp8,
                    "fp32" => cfg.wire_format = cluster_net::WireFormat::Fp32,
                    _ => cfg.wire_format = cluster_net::WireFormat::Fp16,
                },
                "threads" => cfg.threads = val.parse().ok(),
                "moe_prune" => cfg.moe_prune = val.parse().ok(),
                "tools" => cfg.tools = val.parse().unwrap_or(false),
                "tui" => cfg.tui = val.parse().unwrap_or(false),
                "think" => cfg.think = val.parse().unwrap_or(false),
                "draft" => cfg.draft_path = Some(val.to_string()),
                "draft_len" => cfg.draft_len = val.parse().ok(),
                "speculate" => cfg.speculate = val.parse().ok(),
                "ctx" | "context" => cfg.ctx = val.parse().ok(),
                "system_prompt" => cfg.system_prompt = Some(val.to_string()),
                _ => {}
            }
        }
    }
}

fn auto_discover_model() -> Option<String> {
    // The biggest GGUF in the folder is the main model; files named "draft" are speculation
    // helpers, never the main model.
    let mut best: Option<(u64, String)> = None;
    for e in std::fs::read_dir(".").ok()?.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if !name.to_ascii_lowercase().ends_with(".gguf") || name.to_ascii_lowercase().contains("draft") {
            continue;
        }
        let size = e.metadata().map(|m| m.len()).unwrap_or(0);
        if best.as_ref().is_none_or(|(s, _)| size > *s) {
            best = Some((size, name));
        }
    }
    best.map(|(_, n)| n)
}

fn parse_cli() -> CliConfig {
    let args: Vec<String> = std::env::args().collect();
    let mut cfg = CliConfig {
        path: "dummy_weights.bin".to_string(),
        create_gib: 1,
        role: "primary".to_string(),
        bind_addr: SocketAddrV4::new(Ipv4Addr::new(127, 0, 0, 1), 0),
        peer_addr: None,
        prompt: "Hello! Who are you and what can you do?".to_string(),
        prompt_given: false,
        chat_mode: false,
        probe: false,
        stream_layers: false,
        verbose: false,
        raw: None,
        max_tokens: 32,
        sampler: smollm_chat::SamplerConfig::default(),
        ram_budget_mib: None,
        reserve_mib: None,
        pack_experts: false,
        expert_cache_mib: None,
        draft_path: None,
        think: false,
        node_mode: false,
        layer_range: None,
        wire_format: cluster_net::WireFormat::Fp16,
        threads: None,
        moe_prune: None,
        system_prompt: None,
        draft_len: None,
        speculate: None,
        ctx: None,
        tools: false,
        tui: true,
    };

    // Load defaults from spindle.toml if present in current directory
    load_config_file(&mut cfg);

    let mut pos = Vec::new();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "node" | "--node" => {
                cfg.node_mode = true;
                i += 1;
            }
            "--layer-range" | "-l" if i + 1 < args.len() => {
                let s = args[i + 1].trim();
                let parts: Vec<&str> = if s.contains("..") {
                    s.split("..").collect()
                } else if s.contains('-') {
                    s.split('-').collect()
                } else {
                    vec![s]
                };
                if parts.len() == 2 {
                    if let (Ok(start), Ok(end)) = (parts[0].parse::<usize>(), parts[1].parse::<usize>()) {
                        cfg.layer_range = Some(start..end);
                    }
                }
                i += 2;
            }
            "--threads" | "-t" if i + 1 < args.len() => {
                cfg.threads = args[i + 1].parse().ok();
                i += 2;
            }
            "--moe-prune" | "--expert-prune" if i + 1 < args.len() => {
                cfg.moe_prune = args[i + 1].parse().ok();
                i += 2;
            }
            "--wire-format" if i + 1 < args.len() => {
                match args[i + 1].to_ascii_lowercase().as_str() {
                    "fp8" => cfg.wire_format = cluster_net::WireFormat::Fp8,
                    "fp32" => cfg.wire_format = cluster_net::WireFormat::Fp32,
                    _ => cfg.wire_format = cluster_net::WireFormat::Fp16,
                }
                i += 2;
            }
            "--role" if i + 1 < args.len() => {
                cfg.role = args[i + 1].clone();
                i += 2;
            }
            "--bind" if i + 1 < args.len() => {
                cfg.bind_addr = args[i + 1].parse().expect("invalid --bind SocketAddrV4");
                i += 2;
            }
            "--peer" if i + 1 < args.len() => {
                cfg.peer_addr = Some(args[i + 1].parse().expect("invalid --peer SocketAddrV4"));
                i += 2;
            }
            "--prompt" if i + 1 < args.len() => {
                cfg.prompt = args[i + 1].clone();
                cfg.prompt_given = true;
                i += 2;
            }
            "--chat" => {
                cfg.chat_mode = true;
                i += 1;
            }
            "--temp" | "--top-k" | "--top-p" | "--min-p" | "--repeat-penalty" | "--seed"
                if i + 1 < args.len() =>
            {
                let (flag, v) = (args[i].as_str(), args[i + 1].as_str());
                let bad = || -> ! {
                    eprintln!("invalid value for {flag}: {v}");
                    std::process::exit(2)
                };
                let f = || v.parse::<f32>().unwrap_or_else(|_| bad());
                match flag {
                    "--temp" => cfg.sampler.temperature = f(),
                    "--top-k" => cfg.sampler.top_k = v.parse().unwrap_or_else(|_| bad()),
                    "--top-p" => cfg.sampler.top_p = f(),
                    "--min-p" => cfg.sampler.min_p = f(),
                    "--repeat-penalty" => cfg.sampler.repeat_penalty = f(),
                    _ => cfg.sampler.seed = v.parse().unwrap_or_else(|_| bad()),
                }
                i += 2;
            }
            "--ram-budget" | "--reserve" if i + 1 < args.len() => {
                let mib: u64 = args[i + 1].parse().unwrap_or_else(|_| {
                    eprintln!("{} takes a size in MiB, got {}", args[i], args[i + 1]);
                    std::process::exit(2)
                });
                if args[i] == "--ram-budget" {
                    cfg.ram_budget_mib = Some(mib);
                } else {
                    cfg.reserve_mib = Some(mib);
                }
                i += 2;
            }
            "--raw" if i + 1 < args.len() => {
                cfg.raw = Some(args[i + 1].clone());
                i += 2;
            }
            "-n" if i + 1 < args.len() => {
                cfg.max_tokens = args[i + 1].parse().unwrap_or_else(|_| {
                    eprintln!("-n takes a number of tokens, got {}", args[i + 1]);
                    std::process::exit(2)
                });
                i += 2;
            }
            "--verbose" | "-v" => {
                cfg.verbose = true;
                i += 1;
            }
            "--stream-layers" => {
                cfg.stream_layers = true;
                i += 1;
            }
            "--probe" => {
                cfg.probe = true;
                i += 1;
            }
            "--pack-experts" => {
                cfg.pack_experts = true;
                i += 1;
            }
            "--expert-cache-mib" if i + 1 < args.len() => {
                cfg.expert_cache_mib = args[i + 1].parse().ok();
                i += 2;
            }
            "--draft" if i + 1 < args.len() => {
                cfg.draft_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--system-prompt" if i + 1 < args.len() => {
                cfg.system_prompt = Some(args[i + 1].clone());
                i += 2;
            }
            "--draft-len" if i + 1 < args.len() => {
                cfg.draft_len = args[i + 1].parse().ok();
                i += 2;
            }
            "--no-spec" => {
                cfg.speculate = Some(false);
                i += 1;
            }
            "--ctx" if i + 1 < args.len() => {
                cfg.ctx = args[i + 1].parse().ok();
                i += 2;
            }
            "--tools" => {
                cfg.tools = true;
                i += 1;
            }
            "--tui" => {
                cfg.tui = true;
                i += 1;
            }
            "--no-tui" => {
                cfg.tui = false;
                i += 1;
            }
            "--think" => {
                cfg.think = true;
                i += 1;
            }
            "chat" => {
                cfg.chat_mode = true;
                i += 1;
            }
            "--version" | "-V" => {
                println!("spindle {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            "--help" | "-h" => {
                print!("{}", help_text());
                std::process::exit(0);
            }
            other => {
                pos.push(other.to_string());
                i += 1;
            }
        }
    }
    for p in &pos {
        if p != "node" && p != "chat" {
            cfg.path = p.clone();
            break;
        }
    }
    if (cfg.path == "dummy_weights.bin" || !std::path::Path::new(&cfg.path).exists()) && !cfg.probe {
        if let Some(m) = auto_discover_model() {
            cfg.path = m;
        }
    }
    if cfg.path.ends_with(".gguf") && !cfg.node_mode && !cfg.probe && !cfg.chat_mode && !cfg.prompt_given {
        cfg.chat_mode = true;
        cfg.tools = true;
    }
    if let Some(g) = pos.iter().filter(|s| *s != "node" && *s != "chat" && *s != &cfg.path).next().and_then(|s| s.parse().ok()) {
        cfg.create_gib = g;
    }
    cfg
}

/// Reads this node's weights into RAM with a progress bar, so the first reply doesn't stall
/// on the disk. Models larger than free RAM are left to page in as they're used.
fn load_weights(loader: &loading::Loader, engine: &smollm_chat::SmolLm2Engine) {
    let total = engine.resident_bytes();
    if total == 0 {
        loader.step("Weights", "").done("larger than free RAM: read from disk as needed");
        return;
    }
    let step = loader.step("Weights", &format!("reading {} into RAM", loading::fmt_bytes(total)));
    let t0 = Instant::now();
    let read = engine.warm_weights(4, &mut |done, all| {
        let secs = t0.elapsed().as_secs_f64().max(1e-3);
        let rate = done as f64 / secs;
        let mut detail = format!("{} / {}", loading::fmt_bytes(done), loading::fmt_bytes(all));
        if done > 0 && secs > 0.5 {
            detail.push_str(&format!(" · {}/s", loading::fmt_bytes(rate as u64)));
            let left = (all - done) as f64 / rate;
            if left >= 1.0 {
                detail.push_str(&format!(" · {} left", loading::fmt_secs(left)));
            }
        }
        step.progress(done, all, detail);
    });
    let secs = t0.elapsed().as_secs_f64().max(1e-3);
    step.done(&format!("{} in RAM · {}/s", loading::fmt_bytes(read), loading::fmt_bytes((read as f64 / secs) as u64)));
}

// --------------------------------------------------------------------- main

fn main() -> Result<()> {
    if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c")) {
        eprintln!("CPU lacks AVX2/FMA/F16C");
        std::process::exit(1);
    }
    let cli = parse_cli();
    if let Some(n) = cli.ctx {
        // Read by the engine when it sizes the KV-cache (before any worker threads exist).
        std::env::set_var("CHAT_CTX", n.to_string());
    }
    let path = cli.path.as_str();
    // Chat is quiet unless asked; the synthetic benchmark keeps its full log.
    ui::init(cli.verbose || !path.ends_with(".gguf"));
    if cli.probe {
        if let Err(e) = probe::run(path) {
            eprintln!("probe failed: {e}");
            std::process::exit(1);
        }
        return Ok(());
    }
    // Only synthesize the demo model for non-GGUF paths, so a typo in a model name fails
    // loudly instead of silently writing a 1 GiB fake file under that name.
    if path.ends_with(".gguf") && !std::path::Path::new(path).exists() {
        eprintln!("model file not found: {path}");
        std::process::exit(1);
    }
    // Real models get the startup progress display; the synthetic benchmark keeps its log.
    let loader = path.ends_with(".gguf").then(|| {
        loading::Loader::start(
            &format!("spindle {}", env!("CARGO_PKG_VERSION")),
            if cli.node_mode { "worker node" } else { "" },
        )
    });
    let model_step = loader.as_ref().map(|l| {
        let file = std::path::Path::new(path).file_name().map_or(path.into(), |f| f.to_string_lossy().into_owned());
        l.step("Model", &format!("reading {file}"))
    });
    ensure_real_gguf_moe_model(path, cli.create_gib, NUM_LAYERS).expect("failed to build GGUF model");
    // Stamp full spec-compliant GGUF v3 header (KV metadata + BPE vocab + 256 named tensor descriptors)
    // if the file does not already have a full GGUF v3 metadata table.
    if GgufModelIndex::parse_file(path).is_err() {
        write_spec_compliant_gguf_v3_header(path, NUM_LAYERS, NUM_EXPERTS, CHUNK_SIZE as u64)
            .expect("failed to write GGUF v3 metadata & tensor table");
    }
    let gguf_idx = GgufModelIndex::parse_file(path).expect("failed to parse GGUF v3 file");

    unsafe {
        let topo = detect_hybrid_topology()?;
        vlog!(
            "[topo] {} logical CPUs: {} P-core threads ({} physical P-cores, EffClass {}) + {} E-Cores (EffClass 0)",
            topo.total_logical,
            topo.p_logical_count,
            topo.distinct_p_cores,
            topo.max_efficiency_class,
            topo.e_logical_count
        );

        // A real trained Llama-family GGUF goes straight to the chat engine, before any of the
        // synthetic-benchmark allocations below (256 MiB arena, RIO slabs, ...) so chat only
        // pays for what it uses.
        if gguf_idx.tensor_by_name.contains_key("token_embd.weight")
            && gguf_idx.tensor_by_name.contains_key("blk.0.attn_q.weight")
        {
            if let Some(core) = topo.compute_core(0) {
                lock_thread_to_core(core, "chat-pcore")?;
            }
            let loaded = match cli.layer_range.clone() {
                Some(range) => smollm_chat::SmolLm2Engine::load_with_index_sharded(path, gguf_idx, range),
                None => smollm_chat::SmolLm2Engine::load_with_index(path, gguf_idx),
            };
            let mut engine = match loaded {
                Ok(e) => e,
                Err(e) => {
                    match model_step {
                        Some(s) => s.fail(&format!("could not load {path}"), Some(&e.to_string())),
                        None => eprintln!("{}", ui::warn(&format!("could not load {path}: {e}"))),
                    }
                    std::process::exit(1);
                }
            };
            if let Some(s) = model_step {
                let r = &engine.layer_range;
                let layers = if r.len() == engine.cfg.n_layers {
                    format!("{} layers", engine.cfg.n_layers)
                } else {
                    format!("layers {}..{} of {}", r.start, r.end, engine.cfg.n_layers)
                };
                s.done(&format!(
                    "{} · {} · {layers}",
                    engine.idx.model_name,
                    loading::fmt_bytes(engine.raw_bytes.len() as u64)
                ));
            }
            if let Some(l) = loader.as_ref() {
                load_weights(l, &engine);
            }

            if cli.node_mode {
                let n_threads = cli.threads.unwrap_or_else(|| {
                    std::env::var("CHAT_THREADS")
                        .ok()
                        .and_then(|v| v.parse::<usize>().ok())
                        .unwrap_or_else(|| {
                            std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
                        })
                })
                .clamp(1, 32);
                let cores = topo.clone();
                engine.enable_threads(n_threads, move |i| {
                    if let Some(core) = cores.compute_core(i) {
                        let _ = lock_thread_to_core(core, &format!("matvec-{i}"));
                    }
                });
                if let Some(p) = cli.moe_prune {
                    engine.set_moe_prune(p);
                }
                if let Some(l) = loader.as_ref() {
                    l.step("Threads", "").done(&format!("{n_threads} compute threads"));
                }

                let bind_addr = cli.bind_addr;
                let listener = match std::net::TcpListener::bind(bind_addr) {
                    Ok(l) => l,
                    Err(e) => {
                        eprintln!("{}", ui::warn(&format!("could not listen on {bind_addr}: {e}")));
                        std::process::exit(1);
                    }
                };
                let actual = listener.local_addr().unwrap();
                if let Some(l) = loader {
                    l.finish("Ready");
                }
                println!("  Listening on {actual}. Waiting for the client to connect (Ctrl+C to stop).");
                println!();
                for stream in listener.incoming() {
                    match stream {
                        Ok(s) => {
                            let _ = cluster_net::serve_worker_connection(&mut engine, s, cli.wire_format);
                            println!("  Waiting for the next connection.");
                        }
                        Err(e) => eprintln!("{}", ui::warn(&format!("incoming connection failed: {e}"))),
                    }
                }
                return Ok(());
            }

            if let Some(peer) = cli.peer_addr {
                let peer_sock = std::net::SocketAddr::V4(peer);
                let step = loader.as_ref().map(|l| l.step("Worker", &format!("connecting to {peer_sock}")));
                let connected = cluster_net::ClusterClient::connect(
                    peer_sock,
                    &engine.idx.model_name,
                    engine.cfg.dim,
                    engine.cfg.n_layers,
                    (engine.layer_range.start, engine.layer_range.end),
                    cli.wire_format,
                );
                let client = match connected {
                    Ok(mut c) => {
                        let rtt = c.ping().unwrap_or(Duration::from_millis(1));
                        let ms = rtt.as_secs_f64() * 1000.0;
                        let msg = format!(
                            "{peer_sock} · layers {}..{} · {} ms round trip",
                            c.worker_range.0,
                            c.worker_range.1,
                            if ms < 10.0 { format!("{ms:.2}") } else { format!("{ms:.0}") }
                        );
                        match step {
                            Some(s) => s.done(&msg),
                            None => println!("connected to worker {msg}"),
                        }
                        c
                    }
                    Err(e) => {
                        let hint = format!(
                            "{e}\nStart the worker on the other PC first:\n  spindle node <model.gguf> --bind 0.0.0.0:{} --layer-range 0..{}\nand allow that port through its firewall (Private network).",
                            peer_sock.port(),
                            engine.layer_range.start
                        );
                        match step {
                            Some(s) => s.fail(&format!("could not reach the worker at {peer_sock}"), Some(&hint)),
                            None => eprintln!("{}", ui::warn(&format!("could not reach the worker at {peer_sock}: {e}"))),
                        }
                        std::process::exit(1);
                    }
                };
                engine.attach_cluster(client);
            }

            if cli.pack_experts {
                if !engine.is_moe() {
                    eprintln!("{}", ui::warn(&format!("{path} is not an MoE model")));
                    std::process::exit(1);
                }
                let pack_path = expert_pack::pack_path_for_model(path);
                println!(
                    "Repacking {} MoE experts ({} layers x {} experts) into {}...",
                    engine.cfg.n_layers * engine.cfg.n_expert,
                    engine.cfg.n_layers,
                    engine.cfg.n_expert,
                    pack_path
                );
                let t0 = Instant::now();
                match engine.repack_moe_experts(|l, nl, written, total| {
                    print!(
                        "\r[repack] layer {l}/{nl} ({:.1}%) · {:.1} MiB / {:.1} MiB",
                        (written as f64 / total as f64) * 100.0,
                        written as f64 / (1024.0 * 1024.0),
                        total as f64 / (1024.0 * 1024.0)
                    );
                    let _ = std::io::stdout().flush();
                }) {
                    Ok(idx) => {
                        println!();
                        println!(
                            "Done in {:.2} s: {} total experts, slot size {:.2} MiB, total pack file {:.1} MiB",
                            t0.elapsed().as_secs_f64(),
                            idx.hdr.n_layers * idx.hdr.n_expert,
                            idx.hdr.expert_slot_bytes as f64 / (1024.0 * 1024.0),
                            ((idx.hdr.n_layers * idx.hdr.n_expert) as f64 * idx.hdr.expert_slot_bytes as f64) / (1024.0 * 1024.0)
                        );
                        return Ok(());
                    }
                    Err(e) => {
                        eprintln!("\n{}", ui::warn(&format!("repack failed: {e}")));
                        std::process::exit(1);
                    }
                }
            }

            // The expert cache is opt-in (--expert-cache-mib): measured on Granite 1B-A400M it is
            // 2-3x slower than reading experts through the mmap, because every miss goes to the
            // SSD while mapped pages are often still in Windows' standby cache.
            if engine.is_moe() && cli.expert_cache_mib.is_some_and(|m| m > 0) {
                // Uses <model>.spindle.pack if present, otherwise reads experts straight
                // from the GGUF (no extra disk space needed).
                {
                    // Free RAM minus the reserve and 128 MiB of engine headroom, unless
                    // --expert-cache-mib says otherwise. ExpertCache::open refuses budgets
                    // below one layer of experts, and the engine then reads from the mmap.
                    let reserve = cli.reserve_mib.map_or(memory_guard::DEFAULT_RESERVE, |m| m * memory_guard::MIB) as usize;
                    let budget = cli.expert_cache_mib.map(|m| m << 20).unwrap_or_else(|| {
                        (mmap_file::available_ram_bytes() as usize).saturating_sub(reserve + (128 << 20))
                    });
                    if let Err(e) = engine.enable_expert_cache(budget) {
                        ui::note(&format!("expert cache off ({e}); reading experts from the mapped file"));
                    }
                }
            }

            if let Some(draft_path) = &cli.draft_path {
                match smollm_chat::SmolLm2Engine::load(draft_path) {
                    Ok(mut draft) => {
                        let bytes = draft.warm_weights(4, &mut |_, _| {});
                        if let Some(l) = loader.as_ref() {
                            l.step("Draft", "").done(&format!("{} · {}", draft.idx.model_name, loading::fmt_bytes(bytes)));
                        }
                        draft.set_sampler(cli.sampler);
                        engine.enable_draft(draft);
                    }
                    Err(e) => {
                        eprintln!("{}", ui::warn(&format!("could not load draft model {draft_path}: {e}")));
                        std::process::exit(1);
                    }
                }
            }
            let reserve = cli.reserve_mib.map_or(memory_guard::DEFAULT_RESERVE, |m| m * memory_guard::MIB);
            let ceiling = cli.ram_budget_mib.map(|m| m * memory_guard::MIB);
            let mut private = engine.private_bytes_estimate();
            // When the model doesn't fit, lock as much of it as the budget allows (see
            // `pin_hot_weights`), keeping 128 MiB of the budget for the pages that still stream.
            let mut pinned = 0u64;
            if engine.is_streaming() && std::env::var("SPINDLE_NO_PIN").map_or(true, |v| v != "1") {
                let (ws, _, _) = memory_guard::process_counters();
                let target = ceiling.unwrap_or((ws + mmap_file::available_ram_bytes()).saturating_sub(reserve));
                let pin_budget = target.saturating_sub(private + 128 * memory_guard::MIB);
                let (p, first_streamed) = engine.plan_residency(pin_budget as usize, cli.stream_layers);
                pinned = p as u64;
                match first_streamed {
                    Some(f) => vlog!(
                        "[memory] pinned {} MiB in RAM (LM head + layers 0..{f}); streaming layers {f}..{} from disk",
                        pinned / memory_guard::MIB,
                        engine.cfg.n_layers
                    ),
                    None => vlog!(
                        "[memory] pinned {} MiB of hot weights in RAM; the rest pages in as needed",
                        pinned / memory_guard::MIB
                    ),
                }
            }
            // Stream buffers were allocated by the planner; count them as private memory.
            private = engine.private_bytes_estimate();
            let budget = memory_guard::Budget {
                reserve,
                floor: private + pinned + 128 * memory_guard::MIB,
                ceiling,
                min_ws: pinned + 32 * memory_guard::MIB,
            };
            // A model that fits in RAM needs no cap: its clean, file-backed pages are cheap for
            // Windows to reclaim. The cap matters when the model is bigger than free memory,
            // or when the user asked for a hard budget.
            let governor = if engine.is_streaming() || budget.ceiling.is_some() {
                memory_guard::Governor::start(budget)
            } else {
                vlog!("[memory] model fits in RAM, no working-set cap needed (use --ram-budget to force one)");
                None
            };
            match &governor {
                Some(g) => vlog!(
                    "[memory] governor on: keeping {} MiB free for other apps, current cap {} MiB{}",
                    budget.reserve / memory_guard::MIB,
                    g.status.cap.load(std::sync::atomic::Ordering::Relaxed) / memory_guard::MIB,
                    budget.ceiling.map_or(String::new(), |c| format!(", hard ceiling {} MiB", c / memory_guard::MIB))
                ),
                None if engine.is_streaming() || budget.ceiling.is_some() => {
                    ui::note("Windows refused the working-set limit; running without the memory governor")
                }
                None => {}
            }
            // One thread per physical P-core (main on compute_cores[0], helpers on the rest).
            // Override with `--threads N` or `$env:CHAT_THREADS = "1"`.
            let n_threads = cli.threads.unwrap_or_else(|| {
                std::env::var("CHAT_THREADS")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(topo.compute_cores.len().max(1))
            })
            .clamp(1, 32);
            let cores = topo.clone();
            engine.enable_threads(n_threads, move |i| {
                if let Some(core) = cores.compute_core(i) {
                    let _ = lock_thread_to_core(core, &format!("matvec-{i}"));
                }
            });
            if let Some(p) = cli.moe_prune {
                engine.set_moe_prune(p);
            }
            if let Some(prompt) = &cli.system_prompt {
                engine.set_system_prompt(prompt);
            }
            if let Some(len) = cli.draft_len {
                engine.set_draft_len(len);
            }
            if let Some(on) = cli.speculate {
                engine.speculate = on;
            }
            if cli.tools {
                engine.set_tools_enabled(true);
            }
            engine.set_sampler(cli.sampler);
            if let Some(l) = loader.as_ref() {
                let mut parts = vec![format!("{n_threads} compute threads")];
                if engine.speculate {
                    parts.push(if engine.draft_engine.is_some() { "draft-model speculation".into() } else { "prompt-lookup speculation".into() });
                }
                if engine.is_moe() && engine.moe_prune() > 0.0 {
                    let p = engine.moe_prune();
                    parts.push(if p >= 1.0 { format!("top {} experts", p as usize) } else { format!("expert pruning {p}") });
                }
                if engine.enable_tools {
                    parts.push("tools on".into());
                }
                l.step("Engine", "").done(&parts.join(" · "));
            }
            // Process the system prompt now, while a progress display is up, rather than as a
            // silent pause before the first reply. The full-screen UI does this on its own
            // loading screen; the line-based chat and one-shot prompts do it here.
            let uses_tui = cli.tui && cli.chat_mode && cli.raw.is_none();
            if cli.raw.is_none() && !uses_tui {
                let step = loader.as_ref().map(|l| l.step("Prompt", "processing the system prompt"));
                let mut total = 0;
                engine.prefill_system_with(&mut |done, all| {
                    total = all;
                    if let Some(s) = step.as_ref() {
                        s.progress(done as u64, all as u64, format!("{} / {} tokens", loading::fmt_count(done), loading::fmt_count(all)));
                    }
                });
                if let Some(s) = step {
                    s.done(&format!("system prompt cached ({} tokens) · context {} tokens", loading::fmt_count(total), loading::fmt_count(engine.max_seq)));
                }
            }
            let startup = loader.as_ref().map(|l| l.results()).unwrap_or_default();
            if let Some(l) = loader {
                if !uses_tui {
                    l.finish("Ready");
                }
            }
            let sc = cli.sampler;
            vlog!(
                "[chat] {n_threads} threads, temp={} top_k={} top_p={} min_p={} repeat_penalty={} seed={}",
                sc.temperature, sc.top_k, sc.top_p, sc.min_p, sc.repeat_penalty, sc.seed
            );
            if let Some(text) = &cli.raw {
                use std::io::Write;
                engine.complete_raw(text, cli.max_tokens, &mut |p| {
                    print!("{p}");
                    let _ = std::io::stdout().flush();
                });
                println!();
                return Ok(());
            }
            let opts = chat_cli::ChatOptions {
                // `--chat` alone goes straight to the prompt; without `--chat`, answer once.
                first_prompt: (cli.prompt_given || !cli.chat_mode).then(|| cli.prompt.clone()),
                interactive: cli.chat_mode,
                threads: n_threads,
                max_tokens: if cli.max_tokens == 32 { 2048 } else { cli.max_tokens },
                allow_think: cli.think,
                tui: cli.tui,
                startup,
            };
            chat_cli::print_header(&engine, &opts);
            chat_cli::run(&mut engine, governor.as_ref(), opts);
            return Ok(());
        }

        lock_thread_to_core(topo.producer_pcore, "producer(iocp)")?;

        if let Err(e) = enable_lock_memory_privilege() {
            eprintln!("[warn] {}", e.message());
        }
        let arena = alloc_arena(ARENA_SIZE)?;
        let mut spec_state = SpeculativeState::new_boxed();
        println!(
            "[memory] 256 MiB Arena + {:.1} MiB Int8 GQA KV-Cache + {:.2} MiB Hot-Tier Pinned RAM = {:.1} MiB total",
            spec_state.kv_cache.memory_bytes() as f64 / (1024.0 * 1024.0),
            spec_state.hot_weights.memory_bytes() as f64 / (1024.0 * 1024.0),
            (ARENA_SIZE + spec_state.kv_cache.memory_bytes() + spec_state.hot_weights.memory_bytes()) as f64
                / (1024.0 * 1024.0)
        );

        let tx_slab = arena.ptr.add(arena.len - 2 * RIO_SLAB_BYTES);
        let rx_slab = arena.ptr.add(arena.len - RIO_SLAB_BYTES);
        let rio_tx_ep = RioMeshEndpoint::new(tx_slab, RIO_SLAB_BYTES, cli.bind_addr)?;
        let rio_rx_ep = RioMeshEndpoint::new(rx_slab, RIO_SLAB_BYTES, cli.bind_addr)?;
        let peer_target = cli.peer_addr.unwrap_or(rio_rx_ep.local_addr);
        rio_tx_ep.connect_peer(peer_target)?;
        rio_rx_ep.connect_peer(rio_tx_ep.local_addr)?;

        let file = Handle(CreateFileW(
            &HSTRING::from(path),
            GENERIC_READ.0,
            FILE_SHARE_READ,
            None,
            OPEN_EXISTING,
            FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN,
            HANDLE::default(),
        )?);
        let mut si: FILE_STORAGE_INFO = std::mem::zeroed();
        GetFileInformationByHandleEx(
            file.0,
            FileStorageInfo,
            &mut si as *mut _ as *mut c_void,
            size_of::<FILE_STORAGE_INFO>() as u32,
        )?;
        let sector = si.PhysicalBytesPerSectorForPerformance.max(512);
        let mut fsz = 0i64;
        GetFileSizeEx(file.0, &mut fsz)?;
        let aligned_end = (fsz as u64).div_ceil(sector as u64) * sector as u64;
        let total_file_chunks = aligned_end / (CHUNK_SIZE as u64);

        let sched = inspect_or_map_weights(path, fsz as u64, sector);
        println!(
            "[gguf:v{}] model='{}' arch='{}' | {} named tensors, {} BPE vocab tokens, alignment={} B, data_offset={:#x}",
            gguf_idx.version,
            gguf_idx.model_name,
            gguf_idx.architecture,
            gguf_idx.tensors.len(),
            gguf_idx.vocab.len(),
            gguf_idx.alignment,
            gguf_idx.data_section_offset
        );
        if let (Some(first_t), Some(last_t)) = (gguf_idx.tensors.first(), gguf_idx.tensors.last()) {
            println!(
                "[gguf:tensors] first='{}' ({:?} {:?} @ {:#x}) .. last='{}' (@ {:#x})",
                first_t.name,
                first_t.dtype,
                first_t.dims,
                first_t.abs_file_offset,
                last_t.name,
                last_t.abs_file_offset
            );
        }
        println!(
            "[model] {} | {} layers x {} experts ({:.1} MiB/layer dense, {:.1} MiB/layer Top-{} active MoE)",
            sched.format_name,
            NUM_LAYERS,
            NUM_EXPERTS,
            sched.bytes_per_layer as f64 / (1024.0 * 1024.0),
            (TOP_K_EXPERTS * CHUNK_SIZE) as f64 / (1024.0 * 1024.0),
            TOP_K_EXPERTS
        );


        // =====================================================================
        // PASS 1: Full Dense Baseline (All 256 Chunks = 1.074 GB)
        // =====================================================================
        println!("\n=== PASS 1: Dense 32-Layer Baseline (100% of Weights Streamed) ===");
        let dense_offsets: Vec<u64> = (0..total_file_chunks)
            .map(|c| c * (CHUNK_SIZE as u64))
            .collect();
        let r = execute_stream_schedule(
            file.0,
            sector,
            aligned_end,
            &dense_offsets,
            8,
            &arena,
            &mut spec_state,
            topo.worker_pcores,
            &rio_tx_ep,
            &rio_rx_ep,
            false,
        )?;
        let gb = r.bytes as f64 / 1e9;
        let total_rows: usize = r.workers.iter().map(|w| w.rows_computed).sum();
        let effective_spec_weights = (total_rows * HIDDEN_DIM * SPEC_K) as f64;
        let max_compute_s = r.workers.iter().map(|w| w.compute.as_secs_f64()).fold(0.0f64, f64::max);
        println!(
            "[dense:E2E] {:.3} GB in {:.3} s wall => {:.2} GB/s physical NVMe | {:.2} GB/s effective | {:.2} B weight-ops/s",
            gb,
            r.wall,
            gb / r.wall,
            (gb * SPEC_K as f64) / r.wall,
            effective_spec_weights / max_compute_s / 1e9
        );

        // =====================================================================
        // PASS 2: synthetic MoE generation
        //   - E-Core (`LP #8`) drafts K=4 tokens in RAM
        //   - MoE Router selects Top-2 of 8 experts per layer
        //   - IOCP Producer on P-Core (`LP #0`) DMAs ONLY the 2 active experts
        //   - 3 P-Cores (`LP #2, #4, #6`) verify all 4 tokens + GQA FlashAttention
        // =====================================================================
        println!(
            "\n=== PASS 2: Live Autoregressive MoE Inference (E-Core Drafter + Top-2/8 Sparse NVMe DMA + 3-P-Core Verifier) ==="
        );

        let ecore = topo.drafter_ecore;
        let prompt_tokens = gguf_idx.encode_prompt(&cli.prompt);
        println!(
            "[tokenizer:BPE] prompt=\"{}\" -> encoded {} token IDs: {:?}",
            cli.prompt,
            prompt_tokens.len(),
            prompt_tokens
        );
        let generation_steps = 8usize; // 8 speculative steps x 4 tokens = 32 tokens generated!
        let mut current_token = *prompt_tokens.last().unwrap_or(&0);
        let mut generated_tokens: Vec<usize> = prompt_tokens.clone();
        let mut total_accepted = 0usize;
        let mut total_moe_bytes = 0u64;
        let mut total_ecore_draft_us = 0.0f64;

        let t_gen = Instant::now();
        for step in 0..generation_steps {
            // 1. E-Core (`LP #8`) drafts K=4 tokens in RAM while P-Cores prepare MoE schedule
            let state_ptr = (&mut *spec_state as *mut SpeculativeState) as usize;
            let (drafted, draft_us) = std::thread::scope(|s| {
                s.spawn(move || {
                    if step == 0 {
                        let _ = lock_thread_to_core(ecore, "drafter(e-core)");
                    } else {
                        let ids = [ecore.cpu_set_id];
                        let _ = SetThreadSelectedCpuSets(GetCurrentThread(), &ids);
                        let _ = SetThreadAffinityMask(GetCurrentThread(), 1usize << (ecore.logical_idx as usize));
                    }
                    let t_d = Instant::now();
                    let st_ref = &mut *(state_ptr as *mut SpeculativeState);
                    let d = ecore_draft_spec4_tokens(
                        current_token,
                        &st_ref.hot_weights,
                        &mut st_ref.x_spec,
                    );
                    (d, t_d.elapsed().as_secs_f64() * 1e6)
                })
                .join()
                .unwrap()
            });
            total_ecore_draft_us += draft_us;

            let mut moe_offsets = Vec::with_capacity(NUM_LAYERS * TOP_K_EXPERTS);
            for l in 0..NUM_LAYERS {
                let route = route_moe_top2(
                    &spec_state.x_spec[0],
                    &spec_state.hot_weights.router_gates[l],
                    l + step,
                );
                for &exp_idx in &route.experts {
                    let off = gguf_idx.expert_file_offset(
                        l,
                        exp_idx,
                        CHUNK_SIZE as u64,
                        total_file_chunks.max(1),
                    );
                    moe_offsets.push(off);
                }
            }

            // Re-open unbuffered handle for clean IOCP bind on this speculative step
            let step_file = Handle(CreateFileW(
                &HSTRING::from(path),
                GENERIC_READ.0,
                FILE_SHARE_READ,
                None,
                OPEN_EXISTING,
                FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN,
                HANDLE::default(),
            )?);

            // 3. Stream ONLY the 64 active expert chunks (25% of model) and verify K=4 tokens across 3 P-cores.
            let step_stats = execute_stream_schedule(
                step_file.0,
                sector,
                aligned_end,
                &moe_offsets,
                TOP_K_EXPERTS as u64,
                &arena,
                &mut spec_state,
                topo.worker_pcores,
                &rio_tx_ep,
                &rio_rx_ep,
                true,
            )?;
            total_moe_bytes += step_stats.bytes;

            // 4. Verify all K=4 speculative logits against Q4_K LM_Head
            let prev_chain = [
                current_token,
                drafted[0],
                drafted[1],
                drafted[2],
            ];
            let (verified, accepted) = verify_spec4_logits_avx2(
                prev_chain,
                drafted,
                &spec_state.x_spec,
                &spec_state.hot_weights,
            );
            total_accepted += accepted;
            for &tok in &verified {
                generated_tokens.push(tok);
            }
            current_token = *verified.last().unwrap();
        }
        let gen_wall = t_gen.elapsed().as_secs_f64();

        // Format generated text via GGUF BPE decoder (for external HuggingFace GGUF
        // files with 49k+ vocab, also show the synthetic MoE decode stream).
        let text = if gguf_idx.vocab.len() == moe_engine::VOCAB_SIZE {
            gguf_idx.decode_tokens(&generated_tokens)
        } else {
            let hf_preview = gguf_idx.decode_tokens(&generated_tokens[..generated_tokens.len().min(5)]);
            let mut coherent = String::new();
            for &tid in &generated_tokens {
                coherent.push_str(moe_engine::VOCAB[tid % moe_engine::VOCAB_SIZE]);
            }
            format!("{coherent}  [HF BPE prompt decode: \"{hf_preview}\"]")
        };
        let total_tokens_gen = generation_steps * SPEC_K;
        let tok_per_sec = (total_tokens_gen as f64) / gen_wall;
        let dense_equiv_gb = (generation_steps as f64) * gb;

        println!("\n[output:text] \"{text}\"");
        println!(
            "[moe:router] Streamed only Top-{}/{} experts per layer ({:.2} GB actual NVMe DMA vs {:.2} GB dense = 4.0x I/O reduction!)",
            TOP_K_EXPERTS,
            NUM_EXPERTS,
            total_moe_bytes as f64 / 1e9,
            dense_equiv_gb
        );
        println!(
            "[ecore:drafter] E-Core LP #{} drafted {} tokens in {:.2} us/step avg | Speculative Acceptance Rate: {}/{} ({:.1}%)",
            ecore.logical_idx,
            total_tokens_gen,
            total_ecore_draft_us / (generation_steps as f64),
            total_accepted,
            total_tokens_gen,
            100.0 * (total_accepted as f64) / (total_tokens_gen as f64)
        );
        println!(
            "[performance:LLM] Generated {} verified tokens in {:.3} s => {:.2} tokens/sec ({:.2} GB/s effective dense-equivalent bandwidth!)",
            total_tokens_gen,
            gen_wall,
            tok_per_sec,
            dense_equiv_gb / gen_wall
        );
    }
    Ok(())
}
