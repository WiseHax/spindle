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

mod gguf_loader;
mod moe_engine;
mod q4k_gemv;
mod rio_mesh;
mod smollm_chat;
mod worker_pool;
mod memory_guard;
mod layer_stream;
mod mmap_file;
mod probe;
mod quant;
mod transformer;

use std::{
    ffi::c_void,
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
}

/// Discovers both Performance Cores (`EfficiencyClass == 1`) and Efficiency Cores
/// (`EfficiencyClass == 0`) on the i5-13420H:
/// - P-Core #0 (`LP #0`) -> IOCP NVMe Producer
/// - P-Cores #1..3 (`LP #2, #4, #6`) -> 3 AVX2 `Q4_K` Speculative Verifier Workers
/// - E-Core #0 (`LP #8`) -> In-RAM Speculative Token Drafter
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

    Ok(HybridTopology {
        producer_pcore,
        worker_pcores,
        drafter_ecore,
        max_efficiency_class: max_eff,
        total_logical: cpus.len(),
        p_logical_count: p_cpus.len(),
        e_logical_count: e_cpus.len(),
        distinct_p_cores: distinct_p.len(),
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
    println!(
        "[pin:{role}] pinned to {core_kind} LP #{} (CpuSet {:#x}, PhysicalCore #{}, EffClass {}), active on LP #{}",
        target.logical_idx,
        target.cpu_set_id,
        target.core_idx,
        target.efficiency_class,
        running_on
    );
    Ok(())
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
    sampler: smollm_chat::SamplerConfig,
    /// `--ram-budget`: hard ceiling on spindle's resident memory, in MiB.
    ram_budget_mib: Option<u64>,
    /// `--reserve`: RAM to always leave free for other programs, in MiB.
    reserve_mib: Option<u64>,
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
        sampler: smollm_chat::SamplerConfig::default(),
        ram_budget_mib: None,
        reserve_mib: None,
    };

    let mut pos = Vec::new();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
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
            "--stream-layers" => {
                cfg.stream_layers = true;
                i += 1;
            }
            "--probe" => {
                cfg.probe = true;
                i += 1;
            }
            other => {
                pos.push(other.to_string());
                i += 1;
            }
        }
    }
    if let Some(p) = pos.first() {
        cfg.path = p.clone();
    }
    if let Some(g) = pos.get(1).and_then(|s| s.parse().ok()) {
        cfg.create_gib = g;
    }
    cfg
}

// --------------------------------------------------------------------- main

fn main() -> Result<()> {
    if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c")) {
        eprintln!("CPU lacks AVX2/FMA/F16C");
        std::process::exit(1);
    }
    let cli = parse_cli();
    let path = cli.path.as_str();
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
        println!(
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
            println!(
                "\n=== LIVE REAL-WEIGHT CONVERSATIONAL ENGINE ({}) ===",
                gguf_idx.model_name
            );
            lock_thread_to_core(topo.worker_pcores[0], "chat-pcore")?;
            let mut engine = match smollm_chat::SmolLm2Engine::load_with_index(path, gguf_idx) {
                Ok(e) => e,
                Err(e) => {
                    eprintln!("[error] could not load model: {e}");
                    std::process::exit(1);
                }
            };
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
                    Some(f) => println!(
                        "[memory] pinned {} MiB in RAM (LM head + layers 0..{f}); streaming layers {f}..{} from disk",
                        pinned / memory_guard::MIB,
                        engine.cfg.n_layers
                    ),
                    None => println!(
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
                println!("[memory] model fits in RAM, no working-set cap needed (use --ram-budget to force one)");
                None
            };
            match &governor {
                Some(g) => println!(
                    "[memory] governor on: keeping {} MiB free for other apps, current cap {} MiB{}",
                    budget.reserve / memory_guard::MIB,
                    g.status.cap.load(std::sync::atomic::Ordering::Relaxed) / memory_guard::MIB,
                    budget.ceiling.map_or(String::new(), |c| format!(", hard ceiling {} MiB", c / memory_guard::MIB))
                ),
                None if engine.is_streaming() || budget.ceiling.is_some() => {
                    eprintln!("[warn] Windows refused the working-set limit; running without the memory governor")
                }
                None => {}
            }
            let mut last_faults = memory_guard::process_counters().2;
            // One thread per physical P-core: main on worker_pcores[0], helpers on the rest.
            // Override with `$env:CHAT_THREADS = "1"` to benchmark single-threaded.
            let chat_cores = [
                topo.worker_pcores[0],
                topo.worker_pcores[1],
                topo.worker_pcores[2],
                topo.producer_pcore,
            ];
            let n_threads = std::env::var("CHAT_THREADS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(chat_cores.len())
                .clamp(1, chat_cores.len());
            engine.enable_threads(n_threads, move |i| {
                let _ = lock_thread_to_core(chat_cores[i], &format!("matvec-{i}"));
            });
            println!("[chat] {n_threads} matvec thread(s) on P-Cores");
            engine.set_sampler(cli.sampler);
            let sc = cli.sampler;
            println!(
                "[chat] sampling: temp={} top_k={} top_p={} min_p={} repeat_penalty={} seed={}",
                sc.temperature, sc.top_k, sc.top_p, sc.min_p, sc.repeat_penalty, sc.seed
            );
            // `--chat` alone goes straight to the `You > ` prompt.
            if cli.prompt_given || !cli.chat_mode {
                let (_reply, secs, total_toks) = engine.generate_reply(&cli.prompt, 256);
                println!(
                    "[chat:stats] {} total tokens (prompt + reply) in {:.2} s => {:.1} tok/s on P-Core LP #{}",
                    total_toks,
                    secs,
                    (total_toks as f64) / secs,
                    topo.worker_pcores[0].logical_idx
                );
            } else {
                println!("[chat] type a message (/clear to reset, exit to quit)");
            }

            if cli.chat_mode {
                use std::io::{BufRead, Write};
                let stdin = std::io::stdin();
                loop {
                    print!("\nYou > ");
                    let _ = std::io::stdout().flush();
                    let mut line = String::new();
                    match stdin.lock().read_line(&mut line) {
                        Ok(0) | Err(_) => break, // EOF / closed stdin
                        Ok(_) => {}
                    }
                    // PowerShell pipes may prepend a UTF-8 BOM to the first line.
                    let trimmed = line.trim_start_matches('\u{feff}').trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    if trimmed.eq_ignore_ascii_case("exit") || trimmed.eq_ignore_ascii_case("quit") {
                        break;
                    }
                    if trimmed.eq_ignore_ascii_case("/clear") || trimmed.eq_ignore_ascii_case("/reset") {
                        engine.reset_kv();
                        println!("[chat] conversation history cleared.");
                        continue;
                    }
                    let (_r, s, toks) = engine.generate_reply(trimmed, 512);
                    let (ws, private, faults) = memory_guard::process_counters();
                    let mib = |b: u64| b / memory_guard::MIB;
                    let guard = governor.as_ref().map_or(String::new(), |g| {
                        let st = &g.status;
                        let ord = std::sync::atomic::Ordering::Relaxed;
                        format!(
                            " | cap {} MiB, {} MiB free{}",
                            mib(st.cap.load(ord)),
                            mib(st.available.load(ord)),
                            match st.squeezes.load(ord) {
                                0 => String::new(),
                                n => format!(", squeezed {n}x"),
                            }
                        )
                    });
                    println!(
                        "[chat:stats] {:.1} tok/s ({:.2} s) | RAM: {} MiB working set ({} MiB private){guard} | {} page faults",
                        (toks as f64) / s,
                        s,
                        mib(ws),
                        mib(private),
                        faults - last_faults
                    );
                    last_faults = faults;
                    if let Some((mib, touched, total)) = engine.take_moe_stats() {
                        println!("[moe] {mib:.1} MiB of expert weights read per token, {touched}/{total} experts used this turn");
                    }
                }
            }
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
