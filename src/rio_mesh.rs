//! Experimental: Winsock Registered I/O (RIO) transport for activation vectors.
//!
//! Sets up the RIO function table, registers a 2 MiB slice of the arena as a RIO buffer,
//! and sends/receives 16 KiB hidden-state vectors split into 12 fragments of 1408 B
//! (fits a 1500-byte MTU). Completions are polled in user space. Only used by the synthetic
//! benchmark; nothing in chat depends on it.

#![allow(dead_code)]

use std::{
    ffi::c_void,
    mem::{size_of, zeroed},
    net::{Ipv4Addr, SocketAddrV4},
    ptr,
    time::Instant,
};

use windows::{
    core::{Error, Result, GUID, PCSTR},
    Win32::Networking::WinSock::*,
};

/// Sentinel values from `mswsockdef.h` (`RIO_INVALID_BUFFERID = (RIO_BUFFERID)(ULONG_PTR)0xFFFFFFFF`,
/// `RIO_INVALID_CQ = 0`, `RIO_INVALID_RQ = 0`).
const RIO_INVALID_BUFFERID_VAL: RIO_BUFFERID = RIO_BUFFERID(0xFFFF_FFFF_u32 as isize);
const RIO_INVALID_CQ_VAL: RIO_CQ = RIO_CQ(0);
const RIO_INVALID_RQ_VAL: RIO_RQ = RIO_RQ(0);

/// Reserved tail of the 256 MiB arena used as the pinned RIO Slab (2 MiB).
/// Keeping it inside the existing 256 MiB arena respects the strict 8 GB RAM cap.
pub const RIO_SLAB_BYTES: usize = 2 << 20; // 2 MiB

/// Slot layout inside the 2 MiB RIO Slab:
/// - Slots 0..128   : RX payload slots (2048 B each = 256 KiB)
/// - Slots 128..256 : TX payload slots (2048 B each = 256 KiB)
const SLOT_STRIDE: usize = 2048;
const RX_RING_SLOTS: usize = 128;
const TX_RING_SLOTS: usize = 128;

/// Maximum payload per Ethernet UDP frame (1472 B fits inside standard 1500 MTU
/// without IP fragmentation: 1500 - 20 IPv4 - 8 UDP = 1472).
/// Header is 32 bytes -> 1408 bytes of tensor data (352 x f32) per frame.
pub const WIRE_HEADER_BYTES: usize = 32;
pub const FRAG_TENSOR_BYTES: usize = 1408; // 352 * f32, 64-byte aligned

const MAGIC_TENSOR: u32 = 0x5249_4F54; // "RIOT"
const MSG_KIND_DATA: u8 = 1;

/// 32-byte cache-aligned wire header prepended to every UDP datagram.
#[repr(C, align(8))]
#[derive(Clone, Copy, Debug)]
pub struct MeshHeader {
    pub magic: u32,
    pub msg_kind: u8,     // 1 = Activation Fragment, 2 = Layer/Token ACK
    pub frag_idx: u8,     // 0..frag_count
    pub frag_count: u8,   // 12 for 16 KiB (4096 x f32)
    pub flags: u8,
    pub token_seq: u64,   // monotonically increasing token/step counter
    pub layer_id: u16,    // transformer layer index (0..N)
    pub payload_len: u16, // bytes of tensor payload in this fragment
    pub checksum: u32,    // fast FNV1a-32 over header fields
    pub reserved: u64,
}

impl MeshHeader {
    #[inline(always)]
    pub fn compute_tag(&self) -> u32 {
        let mut h: u32 = 0x811c_9dc5;
        for b in self.token_seq.to_le_bytes() {
            h = (h ^ b as u32).wrapping_mul(0x0100_0193);
        }
        h ^= (self.layer_id as u32) << 16 | (self.frag_idx as u32) << 8 | (self.frag_count as u32);
        h.wrapping_mul(0x0100_0193)
    }
}

/// Wrapper around a Winsock RIO endpoint bound to a pinned slice of our Arena.
pub struct RioMeshEndpoint {
    sock: SOCKET,
    rio: RIO_EXTENSION_FUNCTION_TABLE,
    cq_rx: RIO_CQ,
    cq_tx: RIO_CQ,
    rq: RIO_RQ,
    slab_ptr: *mut u8,
    slab_buf_id: RIO_BUFFERID,
    pub local_addr: SocketAddrV4,
}

unsafe impl Send for RioMeshEndpoint {}
unsafe impl Sync for RioMeshEndpoint {}

impl RioMeshEndpoint {
    /// Initializes Winsock 2.2, resolves the RIO function table, registers `slab_ptr`
    /// (`RIO_SLAB_BYTES`) with the NIC driver via `RIORegisterBuffer`, and pre-posts
    /// `RX_RING_SLOTS` zero-copy receive descriptors.
    pub unsafe fn new(slab_ptr: *mut u8, slab_len: usize, bind_addr: SocketAddrV4) -> Result<Self> {
        assert!(slab_len >= RIO_SLAB_BYTES, "RIO slab must be at least 2 MiB");

        let mut wsa_data: WSADATA = zeroed();
        let err = WSAStartup(0x0202, &mut wsa_data);
        if err != 0 {
            return Err(Error::from_win32());
        }

        // 1. Create a RIO-enabled UDP socket (`WSA_FLAG_REGISTERED_IO`).
        let sock = WSASocketW(
            AF_INET.0 as i32,
            SOCK_DGRAM.0,
            IPPROTO_UDP.0,
            None,
            0,
            WSA_FLAG_REGISTERED_IO | WSA_FLAG_OVERLAPPED,
        )?;

        let sin = sockaddr_in_from(bind_addr);
        if bind(sock, &sin as *const _ as *const SOCKADDR, size_of::<SOCKADDR_IN>() as i32) != 0 {
            let e = Error::from_win32();
            let _ = closesocket(sock);
            return Err(e);
        }

        // Query assigned port if bound to port 0.
        let mut actual_sin: SOCKADDR_IN = zeroed();
        let mut namelen = size_of::<SOCKADDR_IN>() as i32;
        getsockname(sock, &mut actual_sin as *mut _ as *mut SOCKADDR, &mut namelen);
        let local_addr = SocketAddrV4::new(
            Ipv4Addr::from(u32::from_be(actual_sin.sin_addr.S_un.S_addr)),
            u16::from_be(actual_sin.sin_port),
        );

        // 2. Query the RIO_EXTENSION_FUNCTION_TABLE via WSAIoctl.
        let mut rio: RIO_EXTENSION_FUNCTION_TABLE = zeroed();
        rio.cbSize = size_of::<RIO_EXTENSION_FUNCTION_TABLE>() as u32;
        let guid: GUID = WSAID_MULTIPLE_RIO;
        let mut bytes_ret = 0u32;
        let rc = WSAIoctl(
            sock,
            SIO_GET_MULTIPLE_EXTENSION_FUNCTION_POINTER,
            Some(&guid as *const _ as *const c_void),
            size_of::<GUID>() as u32,
            Some(&mut rio as *mut _ as *mut c_void),
            size_of::<RIO_EXTENSION_FUNCTION_TABLE>() as u32,
            &mut bytes_ret,
            None,
            None,
        );
        if rc != 0 {
            let e = Error::from_win32();
            let _ = closesocket(sock);
            return Err(e);
        }

        // 3. Pin the 2 MiB arena slice with RIORegisterBuffer (single MDL for RX, TX, and addresses).
        let register_buf = rio.RIORegisterBuffer.expect("RIORegisterBuffer missing");
        let slab_buf_id = register_buf(PCSTR(slab_ptr), slab_len as u32);
        if slab_buf_id == RIO_INVALID_BUFFERID_VAL || slab_buf_id.0 == -1 {
            let e = Error::from_win32();
            let _ = closesocket(sock);
            return Err(e);
        }

        // 4. Create polled Completion Queues (null notification => pure userspace polling).
        let create_cq = rio.RIOCreateCompletionQueue.expect("RIOCreateCompletionQueue missing");
        let cq_rx = create_cq((RX_RING_SLOTS * 2) as u32, ptr::null());
        if cq_rx == RIO_INVALID_CQ_VAL {
            return Err(Error::from_win32());
        }
        let cq_tx = create_cq((TX_RING_SLOTS * 2) as u32, ptr::null());
        if cq_tx == RIO_INVALID_CQ_VAL {
            return Err(Error::from_win32());
        }

        // 5. Create the Request Queue bound to this socket and both CQs.
        let create_rq = rio.RIOCreateRequestQueue.expect("RIOCreateRequestQueue missing");
        let rq = create_rq(
            sock,
            RX_RING_SLOTS as u32,
            1,
            TX_RING_SLOTS as u32,
            1,
            cq_rx,
            cq_tx,
            ptr::null(),
        );
        if rq == RIO_INVALID_RQ_VAL {
            return Err(Error::from_win32());
        }

        let ep = Self {
            sock,
            rio,
            cq_rx,
            cq_tx,
            rq,
            slab_ptr,
            slab_buf_id,
            local_addr,
        };

        // 6. Pre-post all RX slots so incoming fragments land directly in pinned RAM.
        for slot in 0..RX_RING_SLOTS {
            ep.post_rx_slot(slot, true)?;
        }
        ep.commit_rx()?;

        Ok(ep)
    }

    /// Connects the underlying UDP socket to `peer` so `RIOSend` / `RIOReceive` can
    /// operate on the fast connected-UDP path without per-packet address buffers.
    pub unsafe fn connect_peer(&self, peer: SocketAddrV4) -> Result<()> {
        let sin = sockaddr_in_from(peer);
        if connect(self.sock, &sin as *const _ as *const SOCKADDR, size_of::<SOCKADDR_IN>() as i32) != 0 {
            return Err(Error::from_win32());
        }
        Ok(())
    }

    #[inline(always)]
    unsafe fn post_rx_slot(&self, slot: usize, defer: bool) -> Result<()> {
        let data_buf = RIO_BUF {
            BufferId: self.slab_buf_id,
            Offset: (slot * SLOT_STRIDE) as u32,
            Length: SLOT_STRIDE as u32,
        };
        let recv = self.rio.RIOReceive.unwrap_unchecked();
        let flags = if defer { RIO_MSG_DEFER } else { 0 };
        let ok = recv(self.rq, &data_buf, 1, flags, (slot + 1) as *const c_void);
        if !ok.as_bool() {
            return Err(Error::from_win32());
        }
        Ok(())
    }

    #[inline(always)]
    unsafe fn commit_rx(&self) -> Result<()> {
        let recv = self.rio.RIOReceive.unwrap_unchecked();
        let ok = recv(self.rq, ptr::null(), 0, RIO_MSG_COMMIT_ONLY, ptr::null());
        if !ok.as_bool() {
            return Err(Error::from_win32());
        }
        Ok(())
    }

    /// Transmits a complete f32 activation vector (`tensor`, e.g. 4096 floats = 16 KiB)
    /// to the connected peer using deferred `RIOSend` descriptors and a single
    /// `RIO_MSG_COMMIT_ONLY` kernel transition for the entire 12-fragment burst.
    pub unsafe fn send_activation(
        &self,
        token_seq: u64,
        layer_id: u16,
        tensor: &[f32],
    ) -> Result<usize> {
        let raw_bytes = tensor.len() * size_of::<f32>();
        let frag_count = raw_bytes.div_ceil(FRAG_TENSOR_BYTES);
        assert!(frag_count <= TX_RING_SLOTS);

        let send = self.rio.RIOSend.unwrap_unchecked();
        let src_bytes = tensor.as_ptr() as *const u8;

        for frag in 0..frag_count {
            let tx_slot = RX_RING_SLOTS + frag;
            let slot_ptr = self.slab_ptr.add(tx_slot * SLOT_STRIDE);
            let byte_off = frag * FRAG_TENSOR_BYTES;
            let chunk_len = FRAG_TENSOR_BYTES.min(raw_bytes - byte_off);

            let mut hdr = MeshHeader {
                magic: MAGIC_TENSOR,
                msg_kind: MSG_KIND_DATA,
                frag_idx: frag as u8,
                frag_count: frag_count as u8,
                flags: 0,
                token_seq,
                layer_id,
                payload_len: chunk_len as u16,
                checksum: 0,
                reserved: 0,
            };
            hdr.checksum = hdr.compute_tag();

            ptr::copy_nonoverlapping(&hdr as *const _ as *const u8, slot_ptr, WIRE_HEADER_BYTES);
            ptr::copy_nonoverlapping(
                src_bytes.add(byte_off),
                slot_ptr.add(WIRE_HEADER_BYTES),
                chunk_len,
            );

            let data_buf = RIO_BUF {
                BufferId: self.slab_buf_id,
                Offset: (tx_slot * SLOT_STRIDE) as u32,
                Length: (WIRE_HEADER_BYTES + chunk_len) as u32,
            };

            let ok = send(
                self.rq,
                &data_buf,
                1,
                RIO_MSG_DEFER,
                (frag + 1) as *const c_void,
            );
            if !ok.as_bool() {
                return Err(Error::from_win32());
            }
        }

        // Single syscall commit doorbell for all 12 fragments.
        let ok = send(self.rq, ptr::null(), 0, RIO_MSG_COMMIT_ONLY, ptr::null());
        if !ok.as_bool() {
            return Err(Error::from_win32());
        }

        // Reap TX completions in userspace.
        let dequeue = self.rio.RIODequeueCompletion.unwrap_unchecked();
        let mut results = [zeroed::<RIORESULT>(); 32];
        let mut done = 0usize;
        while done < frag_count {
            let n = dequeue(self.cq_tx, results.as_mut_ptr(), results.len() as u32);
            if n == RIO_CORRUPT_CQ {
                return Err(Error::from_win32());
            }
            if n == 0 {
                std::hint::spin_loop();
                continue;
            }
            done += n as usize;
        }
        Ok(frag_count)
    }

    /// Busy-polls the RIO RX Completion Queue in userspace (zero syscalls until
    /// re-arming the RX ring) to reassemble a full `tensor` activation vector.
    pub unsafe fn recv_activation(&self, out_tensor: &mut [f32]) -> Result<(u64, u16)> {
        let raw_bytes = out_tensor.len() * size_of::<f32>();
        let expected_frags = raw_bytes.div_ceil(FRAG_TENSOR_BYTES);
        let dst_bytes = out_tensor.as_mut_ptr() as *mut u8;

        let dequeue = self.rio.RIODequeueCompletion.unwrap_unchecked();
        let mut results = [zeroed::<RIORESULT>(); 32];
        let mut received_mask: u32 = 0;
        let full_mask: u32 = (1u32 << expected_frags) - 1;
        let mut token_seq = 0u64;
        let mut layer_id = 0u16;

        while received_mask != full_mask {
            let n = dequeue(self.cq_rx, results.as_mut_ptr(), results.len() as u32);
            if n == RIO_CORRUPT_CQ {
                return Err(Error::from_win32());
            }
            if n == 0 {
                std::hint::spin_loop();
                continue;
            }

            for r in &results[..n as usize] {
                let slot = (r.RequestContext as usize) - 1;
                if r.Status == 0 && (r.BytesTransferred as usize) >= WIRE_HEADER_BYTES {
                    let slot_ptr = self.slab_ptr.add(slot * SLOT_STRIDE);
                    let hdr = ptr::read_unaligned(slot_ptr as *const MeshHeader);
                    if hdr.magic == MAGIC_TENSOR
                        && hdr.msg_kind == MSG_KIND_DATA
                        && hdr.checksum == hdr.compute_tag()
                    {
                        let idx = hdr.frag_idx as usize;
                        let len = (hdr.payload_len as usize)
                            .min(raw_bytes.saturating_sub(idx * FRAG_TENSOR_BYTES));
                        ptr::copy_nonoverlapping(
                            slot_ptr.add(WIRE_HEADER_BYTES),
                            dst_bytes.add(idx * FRAG_TENSOR_BYTES),
                            len,
                        );
                        token_seq = hdr.token_seq;
                        layer_id = hdr.layer_id;
                        received_mask |= 1u32 << idx;
                    }
                }
                // Re-queue RX descriptor (deferred).
                self.post_rx_slot(slot, true)?;
            }
            self.commit_rx()?;
        }

        Ok((token_seq, layer_id))
    }
}

impl Drop for RioMeshEndpoint {
    fn drop(&mut self) {
        unsafe {
            if let Some(close_cq) = self.rio.RIOCloseCompletionQueue {
                close_cq(self.cq_rx);
                close_cq(self.cq_tx);
            }
            if let Some(dereg) = self.rio.RIODeregisterBuffer {
                dereg(self.slab_buf_id);
            }
            let _ = closesocket(self.sock);
            let _ = WSACleanup();
        }
    }
}

fn sockaddr_in_from(addr: SocketAddrV4) -> SOCKADDR_IN {
    SOCKADDR_IN {
        sin_family: AF_INET,
        sin_port: addr.port().to_be(),
        sin_addr: IN_ADDR {
            S_un: IN_ADDR_0 {
                S_addr: u32::from_ne_bytes(addr.ip().octets()),
            },
        },
        sin_zero: [0; 8],
    }
}

/// Self-contained zero-copy RIO benchmark: carves Two 2 MiB slabs out of our
/// existing 256 MiB arena and runs `iterations` full 16 KiB tensor transfers
/// (4096 x f32 = 12 UDP fragments per token step) to measure true one-way latency
/// and effective mesh throughput.
pub unsafe fn benchmark_rio_mesh(arena_ptr: *mut u8, arena_len: usize, iterations: usize) -> Result<()> {
    assert!(arena_len >= 2 * RIO_SLAB_BYTES);
    // Carve two 2 MiB slabs from the tail of the 256 MiB arena (zero extra VirtualAlloc).
    let tx_slab = arena_ptr.add(arena_len - 2 * RIO_SLAB_BYTES);
    let rx_slab = arena_ptr.add(arena_len - RIO_SLAB_BYTES);

    let lo = Ipv4Addr::new(127, 0, 0, 1);
    let node_a = RioMeshEndpoint::new(tx_slab, RIO_SLAB_BYTES, SocketAddrV4::new(lo, 0))?;
    let node_b = RioMeshEndpoint::new(rx_slab, RIO_SLAB_BYTES, SocketAddrV4::new(lo, 0))?;
    node_a.connect_peer(node_b.local_addr)?;
    node_b.connect_peer(node_a.local_addr)?;

    println!(
        "[rio] extension function table loaded; node A {} <-> node B {}",
        node_a.local_addr, node_b.local_addr
    );
    println!(
        "[rio] pinned 2 x 2 MiB RIO slabs inside existing 256 MiB arena (0 B extra RAM)"
    );

    let mut tx_tensor = vec![0.0f32; 4096];
    for (i, v) in tx_tensor.iter_mut().enumerate() {
        *v = (i as f32) * 0.00125 - 2.5;
    }
    let mut rx_tensor = vec![0.0f32; 4096];

    // Warm-up pass (populates ARP/loopback fast-path).
    for step in 0..8 {
        node_a.send_activation(step as u64, 1, &tx_tensor)?;
        node_b.recv_activation(&mut rx_tensor)?;
    }

    let mut latencies_us = Vec::with_capacity(iterations);
    let t_total = Instant::now();
    for step in 0..iterations {
        tx_tensor[0] = step as f32;
        let t0 = Instant::now();
        node_a.send_activation(step as u64, 15, &tx_tensor)?;
        let (seq, layer) = node_b.recv_activation(&mut rx_tensor)?;
        latencies_us.push(t0.elapsed().as_secs_f64() * 1e6);
        debug_assert_eq!(seq, step as u64);
        debug_assert_eq!(layer, 15);
    }
    let total_secs = t_total.elapsed().as_secs_f64();

    // Verify zero-loss bit-exact tensor reconstruction.
    assert_eq!(
        tx_tensor.as_slice(),
        rx_tensor.as_slice(),
        "RIO tensor activation mismatch!"
    );

    latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min_us = latencies_us[0];
    let p50_us = latencies_us[iterations / 2];
    let p99_us = latencies_us[(iterations * 99) / 100];
    let total_bytes = iterations * tx_tensor.len() * size_of::<f32>();
    let gbps = (total_bytes as f64 * 8.0) / total_secs / 1e9;

    println!(
        "[rio:mesh] {}x 16 KiB tensor activations (12 frags/tensor) verified bit-exact:",
        iterations
    );
    println!(
        "[rio:latency] min = {:.1} us | p50 = {:.1} us | p99 = {:.1} us  (sub-ms target: < 1000 us)",
        min_us, p50_us, p99_us
    );
    println!(
        "[rio:throughput] {:.2} Gbps ({:.1} MB/s) across zero-copy RIO rings",
        gbps,
        (total_bytes as f64) / total_secs / 1e6
    );

    Ok(())
}
