//! Cluster transport: the framed TCP protocol between the client and a worker node.
//!
//! The model's layers are split between two machines (for example layers 0..35 on a 16 GB
//! desktop and 35..48 on an 8 GB laptop). For every token, or batch of tokens during
//! prefill and speculative verify, the client sends token ids to the worker, the worker
//! runs its layers and sends back one hidden-state vector per token, and the client runs
//! the remaining layers and the LM head.
//!
//! - Frames have a 16-byte header (magic, type, wire format, payload length).
//! - Hidden states travel as FP16 (half the bytes of FP32) or FP8 (a quarter), converted
//!   with AVX2/F16C.
//! - `TCP_NODELAY` is set and buffers are reused, so a round trip on a LAN costs well under
//!   a millisecond of overhead beyond the network itself.
//! - New frame types are only ever added, never changed, so an older worker fails clearly
//!   instead of silently misreading a frame.

use std::{
    arch::x86_64::*,
    io::{Read, Write},
    net::{SocketAddr, TcpStream},
    time::{Duration, Instant},
};

use crate::quant::{f16_to_f32_slice, f32_to_f16_slice};

pub const MAGIC_SPIN: u32 = 0x5350_494E; // "SPIN"

pub const FRAME_HANDSHAKE: u8 = 0x01;
pub const FRAME_HANDSHAKE_ACK: u8 = 0x02;
pub const FRAME_PREFILL_REQ: u8 = 0x03;
pub const FRAME_PREFILL_RESP: u8 = 0x04;
pub const FRAME_STEP_REQ: u8 = 0x05;
pub const FRAME_STEP_RESP: u8 = 0x06;
pub const FRAME_RESET: u8 = 0x07;
pub const FRAME_PING: u8 = 0x08;
pub const FRAME_PONG: u8 = 0x09;
pub const FRAME_SET_PRUNE: u8 = 0x0A;
/// Master -> worker: apply the master's sliding-window KV shift (`n_keep`, `n_discard`,
/// master `cur_pos`, all u32 LE) so both halves of the model keep identical positions.
/// The worker replies with an empty `FRAME_KV_SHIFT`.
pub const FRAME_KV_SHIFT: u8 = 0x0B;
/// Master -> worker (empty): asks for the worker's KV capacity. Reply: `FRAME_KV_INFO` with
/// `max_seq: u32 LE`. Additive: workers that predate it simply don't answer (master times out).
pub const FRAME_KV_INFO: u8 = 0x0C;
/// Client -> worker (empty): asks how long the worker has spent computing. Reply:
/// `FRAME_TIMING` with `compute_us: u64 LE, requests: u64 LE`, both counted since the
/// connection opened (STEP and PREFILL requests, from receiving the request to sending the
/// answer). The client subtracts this from its round-trip time to get the network's share.
/// Workers older than 1.1 don't answer; the client then stops asking.
pub const FRAME_TIMING: u8 = 0x0D;
/// Client -> worker (empty): asks for the worker's expert pick counts since it loaded.
/// Reply: `FRAME_EXPERT_STATS` with `n_layers: u32`, then per layer `layer: u32,
/// expert_bytes: u64, n_expert: u32, picks: n_expert x u32` (all LE). Older workers don't
/// answer; the client then stops asking.
pub const FRAME_EXPERT_STATS: u8 = 0x0E;
/// Client -> worker: load or save the worker's part of the system prompt's KV entries.
/// Payload: `op: u8` (`PROMPT_CACHE_LOAD` / `PROMPT_CACHE_SAVE`), `key: u64`, `n: u32`, then
/// `n` token ids (u32 LE). Reply: `FRAME_PROMPT_CACHE` with one byte, 1 = done, 0 = no
/// matching cache (load) or not saved. Older workers don't answer; the client then stops
/// using the cache on that connection.
pub const FRAME_PROMPT_CACHE: u8 = 0x0F;
/// Client -> worker: turn resident-only expert routing on (1) or off (0) for the worker's
/// layers, using the experts the worker locked in RAM. Reply: `FRAME_SET_EXPERT_MODE` with
/// one byte, 1 if routing is now restricted on the worker. Older workers don't answer.
pub const FRAME_SET_EXPERT_MODE: u8 = 0x10;
pub const PROMPT_CACHE_LOAD: u8 = 0;
pub const PROMPT_CACHE_SAVE: u8 = 1;
pub const FRAME_SHUTDOWN: u8 = 0xFF;

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WireFormat {
    Fp32 = 0,
    Fp16 = 1,
    Fp8 = 2,
}

impl WireFormat {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => WireFormat::Fp16,
            2 => WireFormat::Fp8,
            _ => WireFormat::Fp32,
        }
    }
}

/// 16-byte wire header prepended to every frame.
#[repr(C, align(8))]
#[derive(Clone, Copy, Debug)]
pub struct WireHeader {
    pub magic: u32,
    pub frame_type: u8,
    pub wire_format: u8,
    pub flags: u16,
    pub payload_len: u32,
    pub checksum: u32,
}

impl WireHeader {
    #[inline(always)]
    pub fn compute_checksum(&self) -> u32 {
        let mut h: u32 = 0x811c_9dc5;
        h = (h ^ self.magic).wrapping_mul(0x0100_0193);
        h = (h ^ (self.frame_type as u32)).wrapping_mul(0x0100_0193);
        h = (h ^ ((self.wire_format as u32) << 8)).wrapping_mul(0x0100_0193);
        h = (h ^ ((self.flags as u32) << 16)).wrapping_mul(0x0100_0193);
        h = (h ^ self.payload_len).wrapping_mul(0x0100_0193);
        h
    }

    pub fn to_bytes(&self) -> [u8; 16] {
        let mut buf = [0u8; 16];
        buf[0..4].copy_from_slice(&self.magic.to_le_bytes());
        buf[4] = self.frame_type;
        buf[5] = self.wire_format;
        buf[6..8].copy_from_slice(&self.flags.to_le_bytes());
        buf[8..12].copy_from_slice(&self.payload_len.to_le_bytes());
        buf[12..16].copy_from_slice(&self.checksum.to_le_bytes());
        buf
    }

    pub fn from_bytes(buf: &[u8; 16]) -> Option<Self> {
        let magic = u32::from_le_bytes(buf[0..4].try_into().unwrap());
        if magic != MAGIC_SPIN {
            return None;
        }
        let frame_type = buf[4];
        let wire_format = buf[5];
        let flags = u16::from_le_bytes(buf[6..8].try_into().unwrap());
        let payload_len = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        let checksum = u32::from_le_bytes(buf[12..16].try_into().unwrap());
        let hdr = Self {
            magic,
            frame_type,
            wire_format,
            flags,
            payload_len,
            checksum,
        };
        if hdr.compute_checksum() != checksum {
            return None;
        }
        Some(hdr)
    }
}

// ---------------------------------------------------------------- SIMD Vector Wire Codecs

/// Compresses a slice of f32 activations into wire bytes.
/// - In FP16 mode: uses AVX2 / F16C intrinsics, exact half-precision (2 bytes/float).
/// - In FP8 mode: dynamic block quantization with 4-byte scale + 1-byte int8 per float.
/// - In FP32 mode: raw little-endian f32 (4 bytes/float).
pub fn pack_activation(src: &[f32], format: WireFormat, dst: &mut Vec<u8>) {
    dst.clear();
    match format {
        WireFormat::Fp32 => {
            dst.reserve(src.len() * 4);
            for &f in src {
                dst.extend_from_slice(&f.to_le_bytes());
            }
        }
        WireFormat::Fp16 => {
            let padded_len = (src.len() + 7) & !7;
            dst.resize(padded_len * 2, 0);
            let u16_slice = unsafe {
                std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u16, padded_len)
            };
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") {
                let full_chunks = (src.len() / 8) * 8;
                if full_chunks > 0 {
                    unsafe { f32_to_f16_slice(&src[..full_chunks], &mut u16_slice[..full_chunks]) };
                }
                for i in full_chunks..src.len() {
                    let mut tmp_in = [0.0f32; 8];
                    let mut tmp_out = [0u16; 8];
                    tmp_in[0] = src[i];
                    unsafe { f32_to_f16_slice(&tmp_in, &mut tmp_out) };
                    u16_slice[i] = tmp_out[0];
                }
            } else {
                for (i, &f) in src.iter().enumerate() {
                    let h = half_f32_to_f16_fallback(f);
                    u16_slice[i] = h;
                }
            }
            dst.truncate(src.len() * 2);
        }
        WireFormat::Fp8 => {
            // Adaptive symmetric FP8: 4 bytes f32 scale followed by signed 8-bit quantized values.
            let mut max_abs = 0.0f32;
            for &f in src {
                let a = f.abs();
                if a > max_abs {
                    max_abs = a;
                }
            }
            let scale = if max_abs > 1e-8 { max_abs / 127.0 } else { 1.0 };
            let inv_scale = 1.0 / scale;

            dst.reserve(4 + src.len());
            dst.extend_from_slice(&scale.to_le_bytes());
            for &f in src {
                let q = (f * inv_scale).round().clamp(-127.0, 127.0) as i8;
                dst.push(q as u8);
            }
        }
    }
}

/// Decompresses wire bytes into a destination f32 vector.
pub fn unpack_activation(src: &[u8], format: WireFormat, dst: &mut [f32]) -> Result<(), String> {
    match format {
        WireFormat::Fp32 => {
            if src.len() < dst.len() * 4 {
                return Err(format!("payload too short for FP32: {} < {}", src.len(), dst.len() * 4));
            }
            for (i, d) in dst.iter_mut().enumerate() {
                let b = [src[i * 4], src[i * 4 + 1], src[i * 4 + 2], src[i * 4 + 3]];
                *d = f32::from_le_bytes(b);
            }
        }
        WireFormat::Fp16 => {
            if src.len() < dst.len() * 2 {
                return Err(format!("payload too short for FP16: {} < {}", src.len(), dst.len() * 2));
            }
            let padded_len = (dst.len() + 7) & !7;
            let mut u16_buf = vec![0u16; padded_len];
            let raw_u16 = unsafe {
                std::slice::from_raw_parts(src.as_ptr() as *const u16, src.len() / 2)
            };
            u16_buf[..raw_u16.len()].copy_from_slice(raw_u16);

            let mut f32_buf = vec![0.0f32; padded_len];
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") {
                unsafe { f16_to_f32_slice(&u16_buf, &mut f32_buf) };
                dst.copy_from_slice(&f32_buf[..dst.len()]);
            } else {
                for (i, d) in dst.iter_mut().enumerate() {
                    *d = half_f16_to_f32_fallback(raw_u16[i]);
                }
            }
        }
        WireFormat::Fp8 => {
            if src.len() < 4 + dst.len() {
                return Err(format!("payload too short for FP8: {} < {}", src.len(), 4 + dst.len()));
            }
            let scale = f32::from_le_bytes(src[0..4].try_into().unwrap());
            let bytes = &src[4..4 + dst.len()];
            for (i, &b) in bytes.iter().enumerate() {
                let q = b as i8;
                dst[i] = (q as f32) * scale;
            }
        }
    }
    Ok(())
}

fn half_f32_to_f16_fallback(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let frac = bits & 0x007F_FFFF;
    if exp <= 0 {
        0
    } else if exp >= 31 {
        ((sign << 15) | (0x1F << 10)) as u16
    } else {
        ((sign << 15) | ((exp as u32) << 10) | (frac >> 13)) as u16
    }
}

fn half_f16_to_f32_fallback(val: u16) -> f32 {
    let sign = (val >> 15) & 1;
    let exp = (val >> 10) & 0x1F;
    let frac = val & 0x03FF;
    if exp == 0 {
        if frac == 0 {
            if sign == 1 { -0.0 } else { 0.0 }
        } else {
            let f = (frac as f32) / 1024.0 * (2.0f32).powi(-14);
            if sign == 1 { -f } else { f }
        }
    } else if exp == 31 {
        if frac == 0 {
            if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY }
        } else {
            f32::NAN
        }
    } else {
        let f = (1.0 + (frac as f32) / 1024.0) * (2.0f32).powi(exp as i32 - 15);
        if sign == 1 { -f } else { f }
    }
}

// ---------------------------------------------------------------- AVX2 XOR Erasure Parity

/// Computes an AVX2 XOR erasure parity block across N activation slices.
/// If any 1 slice is lost in WiFi transit, it is reconstructed in O(1) time without retransmission:
/// `R = P ^ S_0 ^ ... ^ S_{k-1} (skipping lost slice)`.
#[target_feature(enable = "avx2")]
pub unsafe fn xor_parity_encode_avx2(slices: &[&[u8]], out_parity: &mut [u8]) {
    assert!(!slices.is_empty());
    let len = out_parity.len();
    out_parity.fill(0);

    for slice in slices {
        let n_vec = len / 32;
        let p_ptr = out_parity.as_mut_ptr() as *mut __m256i;
        let s_ptr = slice.as_ptr() as *const __m256i;

        for i in 0..n_vec {
            let p_val = _mm256_loadu_si256(p_ptr.add(i));
            let s_val = _mm256_loadu_si256(s_ptr.add(i));
            _mm256_storeu_si256(p_ptr.add(i), _mm256_xor_si256(p_val, s_val));
        }

        for i in (n_vec * 32)..len {
            out_parity[i] ^= slice[i];
        }
    }
}

// ---------------------------------------------------------------- Stream Connection

pub struct ClusterStream {
    stream: TcpStream,
    read_buf: Vec<u8>,
}

impl ClusterStream {
    pub fn new(stream: TcpStream) -> std::io::Result<Self> {
        stream.set_nodelay(true)?;
        // Keep idle connections alive indefinitely during interactive chat sessions
        stream.set_read_timeout(None)?;
        stream.set_write_timeout(None)?;
        Ok(Self {
            stream,
            read_buf: Vec::with_capacity(64 * 1024),
        })
    }

    pub fn peer_addr(&self) -> std::io::Result<SocketAddr> {
        self.stream.peer_addr()
    }

    pub fn send_frame(
        &mut self,
        frame_type: u8,
        wire_format: WireFormat,
        payload: &[u8],
    ) -> std::io::Result<()> {
        let mut hdr = WireHeader {
            magic: MAGIC_SPIN,
            frame_type,
            wire_format: wire_format as u8,
            flags: 0,
            payload_len: payload.len() as u32,
            checksum: 0,
        };
        hdr.checksum = hdr.compute_checksum();
        let hdr_bytes = hdr.to_bytes();

        // Write header + payload together to minimize syscalls
        self.stream.write_all(&hdr_bytes)?;
        if !payload.is_empty() {
            self.stream.write_all(payload)?;
        }
        self.stream.flush()?;
        Ok(())
    }

    pub fn recv_frame_into(&mut self, dst: &mut Vec<u8>) -> std::io::Result<WireHeader> {
        let mut hdr_buf = [0u8; 16];
        self.stream.read_exact(&mut hdr_buf)?;

        let hdr = WireHeader::from_bytes(&hdr_buf).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "corrupt wire header or checksum mismatch",
            )
        })?;

        let len = hdr.payload_len as usize;
        dst.resize(len, 0);
        if len > 0 {
            self.stream.read_exact(dst)?;
        }
        Ok(hdr)
    }

    pub fn recv_frame(&mut self) -> std::io::Result<(WireHeader, Vec<u8>)> {
        let mut payload = Vec::new();
        let hdr = self.recv_frame_into(&mut payload)?;
        Ok((hdr, payload))
    }

    pub fn ping(&mut self) -> std::io::Result<Duration> {
        let t0 = Instant::now();
        self.send_frame(FRAME_PING, WireFormat::Fp32, &[])?;
        let (hdr, _) = self.recv_frame()?;
        if hdr.frame_type != FRAME_PONG {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "expected PONG frame",
            ));
        }
        Ok(t0.elapsed())
    }
}

// ---------------------------------------------------------------- Cluster Handshake Payloads

#[derive(Clone, Debug)]
pub struct HandshakePayload {
    pub model_name: String,
    pub dim: usize,
    pub total_layers: usize,
    pub layer_start: usize,
    pub layer_end: usize,
}

impl HandshakePayload {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        let name_bytes = self.model_name.as_bytes();
        buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(name_bytes);
        buf.extend_from_slice(&(self.dim as u32).to_le_bytes());
        buf.extend_from_slice(&(self.total_layers as u32).to_le_bytes());
        buf.extend_from_slice(&(self.layer_start as u32).to_le_bytes());
        buf.extend_from_slice(&(self.layer_end as u32).to_le_bytes());
        buf
    }

    pub fn from_bytes(buf: &[u8]) -> Option<Self> {
        if buf.len() < 20 {
            return None;
        }
        let name_len = u32::from_le_bytes(buf[0..4].try_into().ok()?) as usize;
        if buf.len() < 4 + name_len + 16 {
            return None;
        }
        let model_name = String::from_utf8_lossy(&buf[4..4 + name_len]).into_owned();
        let offset = 4 + name_len;
        let dim = u32::from_le_bytes(buf[offset..offset + 4].try_into().ok()?) as usize;
        let total_layers = u32::from_le_bytes(buf[offset + 4..offset + 8].try_into().ok()?) as usize;
        let layer_start = u32::from_le_bytes(buf[offset + 8..offset + 12].try_into().ok()?) as usize;
        let layer_end = u32::from_le_bytes(buf[offset + 12..offset + 16].try_into().ok()?) as usize;

        Some(Self {
            model_name,
            dim,
            total_layers,
            layer_start,
            layer_end,
        })
    }
}

#[derive(Clone, Debug)]
pub struct StepRequestPayload {
    pub token_id: usize,
    pub pos: usize,
}

impl StepRequestPayload {
    pub fn to_bytes(&self) -> [u8; 8] {
        let mut buf = [0u8; 8];
        buf[0..4].copy_from_slice(&(self.token_id as u32).to_le_bytes());
        buf[4..8].copy_from_slice(&(self.pos as u32).to_le_bytes());
        buf
    }

    pub fn from_bytes(buf: &[u8]) -> Option<Self> {
        if buf.len() < 8 {
            return None;
        }
        let token_id = u32::from_le_bytes(buf[0..4].try_into().ok()?) as usize;
        let pos = u32::from_le_bytes(buf[4..8].try_into().ok()?) as usize;
        Some(Self { token_id, pos })
    }
}

// ---------------------------------------------------------------- Cluster Client & Worker Engine

use crate::smollm_chat::SmolLm2Engine;

pub struct ClusterClient {
    pub stream: ClusterStream,
    pub wire_format: WireFormat,
    pub worker_range: (usize, usize),
    pub dim: usize,
    pub scratch_payload: Vec<u8>,
    /// Worker KV-cache capacity in positions (`None` = worker too old to report it).
    pub peer_max_seq: Option<usize>,
    /// Whether the worker answers `FRAME_TIMING` (`None` = not asked yet).
    pub timing_supported: Option<bool>,
    /// Whether the worker answers `FRAME_EXPERT_STATS` (`None` = not asked yet).
    pub expert_stats_supported: Option<bool>,
    /// Whether the worker answers `FRAME_PROMPT_CACHE` (`None` = not asked yet).
    pub prompt_cache_supported: Option<bool>,
}

impl ClusterClient {
    pub fn connect(
        peer: SocketAddr,
        model_name: &str,
        dim: usize,
        total_layers: usize,
        local_range: (usize, usize),
        wire_format: WireFormat,
    ) -> std::io::Result<Self> {
        let tcp = TcpStream::connect(peer)?;
        let mut stream = ClusterStream::new(tcp)?;

        // Send handshake
        let hs = HandshakePayload {
            model_name: model_name.to_string(),
            dim,
            total_layers,
            layer_start: local_range.0,
            layer_end: local_range.1,
        };
        let hs_bytes = hs.to_bytes();
        stream.send_frame(FRAME_HANDSHAKE, wire_format, &hs_bytes)?;

        // Receive handshake ack from worker
        let (hdr, payload) = stream.recv_frame()?;
        if hdr.frame_type != FRAME_HANDSHAKE_ACK {
            return Err(std::io::Error::new(
                std::io::ErrorKind::ConnectionRefused,
                "worker rejected handshake or sent unexpected frame",
            ));
        }

        let worker_hs = HandshakePayload::from_bytes(&payload).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "corrupt worker handshake payload",
            )
        })?;

        if worker_hs.dim != dim {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "model dimension mismatch: worker has dim {}, local has dim {}",
                    worker_hs.dim, dim
                ),
            ));
        }

        let mut client = Self {
            stream,
            wire_format,
            worker_range: (worker_hs.layer_start, worker_hs.layer_end),
            dim,
            scratch_payload: Vec::with_capacity(32 * 1024),
            peer_max_seq: None,
            timing_supported: None,
            expert_stats_supported: None,
            prompt_cache_supported: None,
        };
        client.peer_max_seq = client.query_kv_info();
        Ok(client)
    }

    /// Asks the worker how many KV positions it holds. An old worker ignores the frame, so
    /// this waits at most 3 s and then gives up (and the connection stays usable).
    fn query_kv_info(&mut self) -> Option<usize> {
        let tcp = &self.stream.stream;
        tcp.set_read_timeout(Some(Duration::from_secs(3))).ok()?;
        let res = self
            .stream
            .send_frame(FRAME_KV_INFO, WireFormat::Fp32, &[])
            .and_then(|_| self.stream.recv_frame());
        let _ = self.stream.stream.set_read_timeout(None);
        match res {
            Ok((hdr, p)) if hdr.frame_type == FRAME_KV_INFO && p.len() >= 4 => {
                Some(u32::from_le_bytes(p[0..4].try_into().unwrap()) as usize)
            }
            _ => None,
        }
    }

    /// The worker's cumulative `(compute_us, requests)`, or `None` if it doesn't report them.
    /// The first call waits at most 2 s for an answer; after a worker fails to answer once,
    /// later calls return `None` right away.
    pub fn worker_timing(&mut self) -> Option<(u64, u64)> {
        if self.timing_supported == Some(false) {
            return None;
        }
        let first = self.timing_supported.is_none();
        if first {
            self.stream.stream.set_read_timeout(Some(Duration::from_secs(2))).ok()?;
        }
        let res = self
            .stream
            .send_frame(FRAME_TIMING, WireFormat::Fp32, &[])
            .and_then(|_| self.stream.recv_frame());
        if first {
            let _ = self.stream.stream.set_read_timeout(None);
        }
        match res {
            Ok((hdr, p)) if hdr.frame_type == FRAME_TIMING && p.len() >= 16 => {
                self.timing_supported = Some(true);
                let rd = |i: usize| u64::from_le_bytes(p[i..i + 8].try_into().unwrap());
                Some((rd(0), rd(8)))
            }
            _ => {
                self.timing_supported = Some(false);
                None
            }
        }
    }

    /// Turns resident-only routing on or off on the worker. `Some(true)` = restricted there,
    /// `Some(false)` = not (nothing pinned), `None` = the worker doesn't support it.
    pub fn set_resident_only(&mut self, on: bool) -> Option<bool> {
        self.stream.stream.set_read_timeout(Some(Duration::from_secs(3))).ok()?;
        let res = self
            .stream
            .send_frame(FRAME_SET_EXPERT_MODE, WireFormat::Fp32, &[on as u8])
            .and_then(|_| self.stream.recv_frame());
        let _ = self.stream.stream.set_read_timeout(None);
        match res {
            Ok((hdr, r)) if hdr.frame_type == FRAME_SET_EXPERT_MODE => Some(r.first() == Some(&1)),
            _ => None,
        }
    }

    /// Asks the worker to load or save its part of the prompt cache. `Some(true)` = done,
    /// `Some(false)` = no matching cache / not saved, `None` = the worker doesn't support it.
    pub fn prompt_cache(&mut self, op: u8, key: u64, tokens: &[usize]) -> Option<bool> {
        if self.prompt_cache_supported == Some(false) {
            return None;
        }
        let first = self.prompt_cache_supported.is_none();
        let mut p = Vec::with_capacity(13 + tokens.len() * 4);
        p.push(op);
        p.extend_from_slice(&key.to_le_bytes());
        p.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
        for &t in tokens {
            p.extend_from_slice(&(t as u32).to_le_bytes());
        }
        // Saving writes ~100 MB on the worker; give it time. The first ask (support check)
        // gets a short timeout so an old worker can't stall startup.
        let wait = if first { 3 } else if op == PROMPT_CACHE_SAVE { 60 } else { 30 };
        self.stream.stream.set_read_timeout(Some(Duration::from_secs(wait))).ok()?;
        let res = self.stream.send_frame(FRAME_PROMPT_CACHE, WireFormat::Fp32, &p).and_then(|_| self.stream.recv_frame());
        let _ = self.stream.stream.set_read_timeout(None);
        match res {
            Ok((hdr, r)) if hdr.frame_type == FRAME_PROMPT_CACHE => {
                self.prompt_cache_supported = Some(true);
                Some(r.first() == Some(&1))
            }
            _ => {
                self.prompt_cache_supported = Some(false);
                None
            }
        }
    }

    /// The worker's expert pick counts (see `FRAME_EXPERT_STATS`), or `None` if it doesn't
    /// report them.
    pub fn worker_expert_picks(&mut self) -> Option<Vec<crate::smollm_chat::ExpertPicks>> {
        if self.expert_stats_supported == Some(false) {
            return None;
        }
        let first = self.expert_stats_supported.is_none();
        if first {
            self.stream.stream.set_read_timeout(Some(Duration::from_secs(2))).ok()?;
        }
        let res = self
            .stream
            .send_frame(FRAME_EXPERT_STATS, WireFormat::Fp32, &[])
            .and_then(|_| self.stream.recv_frame());
        if first {
            let _ = self.stream.stream.set_read_timeout(None);
        }
        let parsed = match res {
            Ok((hdr, p)) if hdr.frame_type == FRAME_EXPERT_STATS => decode_expert_stats(&p),
            _ => None,
        };
        self.expert_stats_supported = Some(parsed.is_some());
        parsed
    }

    /// Mirrors the master's KV shift on the worker (see `FRAME_KV_SHIFT`).
    pub fn kv_shift(&mut self, n_keep: usize, n_discard: usize, cur_pos: usize) -> std::io::Result<()> {
        let mut p = [0u8; 12];
        p[0..4].copy_from_slice(&(n_keep as u32).to_le_bytes());
        p[4..8].copy_from_slice(&(n_discard as u32).to_le_bytes());
        p[8..12].copy_from_slice(&(cur_pos as u32).to_le_bytes());
        self.stream.send_frame(FRAME_KV_SHIFT, WireFormat::Fp32, &p)?;
        self.recv_frame_type(FRAME_KV_SHIFT).map(|_| ())
    }

    pub fn ping(&mut self) -> std::io::Result<Duration> {
        self.stream.ping()
    }

    pub fn prefill(&mut self, tokens: &[usize], start_pos: usize) -> std::io::Result<Vec<Vec<f32>>> {
        let mut req_buf = Vec::with_capacity(8 + tokens.len() * 4);
        req_buf.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
        req_buf.extend_from_slice(&(start_pos as u32).to_le_bytes());
        for &t in tokens {
            req_buf.extend_from_slice(&(t as u32).to_le_bytes());
        }

        self.stream.send_frame(FRAME_PREFILL_REQ, self.wire_format, &req_buf)?;
        let (hdr, payload) = self.stream.recv_frame()?;
        if hdr.frame_type != FRAME_PREFILL_RESP {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("expected PREFILL_RESP, got frame type {}", hdr.frame_type),
            ));
        }

        if payload.len() < 8 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "prefill response header too short",
            ));
        }

        let n_tokens = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
        let dim = u32::from_le_bytes(payload[4..8].try_into().unwrap()) as usize;
        if n_tokens != tokens.len() || dim != self.dim {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "prefill token count or dim mismatch",
            ));
        }

        let wire_fmt = WireFormat::from_u8(hdr.wire_format);
        let bytes_per_tok = match wire_fmt {
            WireFormat::Fp32 => dim * 4,
            WireFormat::Fp16 => dim * 2,
            WireFormat::Fp8 => 4 + dim,
        };

        let mut out = Vec::with_capacity(n_tokens);
        let mut offset = 8;
        for _ in 0..n_tokens {
            if offset + bytes_per_tok > payload.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "truncated prefill activation slice",
                ));
            }
            let slice = &payload[offset..offset + bytes_per_tok];
            let mut vec = vec![0.0f32; dim];
            unpack_activation(slice, wire_fmt, &mut vec).map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, e)
            })?;
            out.push(vec);
            offset += bytes_per_tok;
        }

        Ok(out)
    }

    pub fn step(&mut self, token_id: usize, pos: usize) -> std::io::Result<Vec<f32>> {
        let req = StepRequestPayload { token_id, pos };
        self.stream.send_frame(FRAME_STEP_REQ, self.wire_format, &req.to_bytes())?;

        let hdr = self.stream.recv_frame_into(&mut self.scratch_payload)?;
        if hdr.frame_type != FRAME_STEP_RESP {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("expected STEP_RESP, got frame type {}", hdr.frame_type),
            ));
        }

        let mut out = vec![0.0f32; self.dim];
        let wire_fmt = WireFormat::from_u8(hdr.wire_format);
        unpack_activation(&self.scratch_payload, wire_fmt, &mut out).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e)
        })?;

        Ok(out)
    }

    pub fn reset_kv(&mut self) -> std::io::Result<()> {
        self.stream.send_frame(FRAME_RESET, WireFormat::Fp32, &[])?;
        let (hdr, _) = self.recv_frame_type(FRAME_RESET)?;
        if hdr.frame_type != FRAME_RESET {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "expected RESET confirmation",
            ));
        }
        Ok(())
    }

    pub fn set_moe_prune(&mut self, threshold: f32) -> std::io::Result<()> {
        let payload = threshold.to_le_bytes();
        self.stream.send_frame(FRAME_SET_PRUNE, WireFormat::Fp32, &payload)?;
        let (hdr, _) = self.recv_frame_type(FRAME_SET_PRUNE)?;
        if hdr.frame_type != FRAME_SET_PRUNE {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "expected SET_PRUNE confirmation",
            ));
        }
        Ok(())
    }

    fn recv_frame_type(&mut self, expected: u8) -> std::io::Result<(WireHeader, Vec<u8>)> {
        let (hdr, payload) = self.stream.recv_frame()?;
        if hdr.frame_type != expected {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("expected frame type {expected}, got {}", hdr.frame_type),
            ));
        }
        Ok((hdr, payload))
    }
}

pub fn encode_expert_stats(layers: &[crate::smollm_chat::ExpertPicks]) -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&(layers.len() as u32).to_le_bytes());
    for l in layers {
        p.extend_from_slice(&(l.layer as u32).to_le_bytes());
        p.extend_from_slice(&l.expert_bytes.to_le_bytes());
        p.extend_from_slice(&(l.picks.len() as u32).to_le_bytes());
        for &c in &l.picks {
            p.extend_from_slice(&c.to_le_bytes());
        }
    }
    p
}

pub fn decode_expert_stats(p: &[u8]) -> Option<Vec<crate::smollm_chat::ExpertPicks>> {
    let u32_at = |i: usize| p.get(i..i + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()));
    let u64_at = |i: usize| p.get(i..i + 8).map(|b| u64::from_le_bytes(b.try_into().unwrap()));
    let n = u32_at(0)? as usize;
    let mut i = 4;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let layer = u32_at(i)? as usize;
        let expert_bytes = u64_at(i + 4)?;
        let ne = u32_at(i + 12)? as usize;
        i += 16;
        let mut picks = Vec::with_capacity(ne);
        for _ in 0..ne {
            picks.push(u32_at(i)?);
            i += 4;
        }
        out.push(crate::smollm_chat::ExpertPicks { layer, picks, expert_bytes });
    }
    Some(out)
}

fn check_kv_bounds(engine: &SmolLm2Engine, start: usize, n: usize) -> std::io::Result<()> {
    if start + n > engine.max_seq || n == 0 {
        eprintln!(
            "  client asked for KV positions {start}..{} but this worker has {} slots. The two PCs are probably running different spindle versions; update both.",
            start + n,
            engine.max_seq
        );
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "KV position out of range"));
    }
    Ok(())
}

/// Runs the worker node event loop on an accepted connection.
pub fn serve_worker_connection(
    engine: &mut SmolLm2Engine,
    stream: TcpStream,
    wire_format: WireFormat,
) -> std::io::Result<()> {
    let mut cs = ClusterStream::new(stream)?;
    let peer = cs.peer_addr()?;
    println!("  Client connected from {peer}");

    // Wait for handshake
    let (hdr, payload) = cs.recv_frame()?;
    if hdr.frame_type != FRAME_HANDSHAKE {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "expected HANDSHAKE as first frame",
        ));
    }

    let master_hs = HandshakePayload::from_bytes(&payload).ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "corrupt handshake")
    })?;

    if master_hs.dim != engine.cfg.dim {
        eprintln!(
            "  Rejected: the client's model has hidden size {}, this worker's has {}. Both PCs must load the same GGUF file.",
            master_hs.dim, engine.cfg.dim
        );
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "dim mismatch",
        ));
    }

    println!(
        "  Session started: client runs layers {}..{}, this worker runs {}..{}",
        master_hs.layer_start, master_hs.layer_end, engine.layer_range.start, engine.layer_range.end
    );

    // Send Handshake ACK
    let worker_hs = HandshakePayload {
        model_name: engine.idx.model_name.clone(),
        dim: engine.cfg.dim,
        total_layers: engine.cfg.n_layers,
        layer_start: engine.layer_range.start,
        layer_end: engine.layer_range.end,
    };
    cs.send_frame(FRAME_HANDSHAKE_ACK, wire_format, &worker_hs.to_bytes())?;

    // Inference loop
    let mut pack_scratch = Vec::new();
    // Time spent answering STEP and PREFILL requests, reported by FRAME_TIMING.
    let (mut compute_us, mut requests) = (0u64, 0u64);
    let mut payload = Vec::with_capacity(32 * 1024);
    loop {
        let hdr = match cs.recv_frame_into(&mut payload) {
            Ok(h) => h,
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                println!("  Client disconnected.");
                let _ = engine.save_expert_stats();
                break;
            }
            Err(e) if e.kind() == std::io::ErrorKind::TimedOut || e.kind() == std::io::ErrorKind::WouldBlock => {
                continue;
            }
            Err(e) => {
                eprintln!("  Connection lost: {e}");
                let _ = engine.save_expert_stats();
                return Err(e);
            }
        };

        match hdr.frame_type {
            FRAME_PING => {
                cs.send_frame(FRAME_PONG, WireFormat::Fp32, &[])?;
            }
            FRAME_RESET => {
                engine.reset_kv();
                cs.send_frame(FRAME_RESET, WireFormat::Fp32, &[])?;
            }
            FRAME_SET_PRUNE => {
                if payload.len() >= 4 {
                    let p = f32::from_le_bytes(payload[0..4].try_into().unwrap());
                    engine.set_moe_prune(p);
                }
                cs.send_frame(FRAME_SET_PRUNE, WireFormat::Fp32, &[])?;
            }
            FRAME_KV_INFO => {
                cs.send_frame(FRAME_KV_INFO, WireFormat::Fp32, &(engine.max_seq as u32).to_le_bytes())?;
            }
            FRAME_SET_EXPERT_MODE => {
                let on = payload.first() == Some(&1);
                let active = engine.set_resident_only(on);
                if on {
                    println!(
                        "  Resident-only routing {}.",
                        if active { "on: experts are picked among those in RAM" } else { "requested, but nothing is pinned here, so routing is unchanged" }
                    );
                }
                cs.send_frame(FRAME_SET_EXPERT_MODE, WireFormat::Fp32, &[active as u8])?;
            }
            FRAME_PROMPT_CACHE => {
                let ok = (|| -> Option<bool> {
                    let op = *payload.first()?;
                    let key = u64::from_le_bytes(payload.get(1..9)?.try_into().ok()?);
                    let n = u32::from_le_bytes(payload.get(9..13)?.try_into().ok()?) as usize;
                    let tokens: Vec<usize> = payload
                        .get(13..13 + n * 4)?
                        .chunks_exact(4)
                        .map(|c| u32::from_le_bytes(c.try_into().unwrap()) as usize)
                        .collect();
                    Some(match op {
                        PROMPT_CACHE_LOAD => {
                            let hit = engine.load_prompt_kv(key, &tokens).unwrap_or(false);
                            if hit {
                                println!("  System prompt loaded from cache ({n} tokens).");
                            }
                            hit
                        }
                        PROMPT_CACHE_SAVE => engine.save_prompt_kv(key, &tokens).is_ok(),
                        _ => false,
                    })
                })()
                .unwrap_or(false);
                cs.send_frame(FRAME_PROMPT_CACHE, WireFormat::Fp32, &[ok as u8])?;
            }
            FRAME_EXPERT_STATS => {
                cs.send_frame(FRAME_EXPERT_STATS, WireFormat::Fp32, &encode_expert_stats(&engine.expert_picks()))?;
            }
            FRAME_TIMING => {
                let mut p = [0u8; 16];
                p[0..8].copy_from_slice(&compute_us.to_le_bytes());
                p[8..16].copy_from_slice(&requests.to_le_bytes());
                cs.send_frame(FRAME_TIMING, WireFormat::Fp32, &p)?;
            }
            FRAME_KV_SHIFT => {
                if payload.len() >= 12 {
                    let rd = |i: usize| u32::from_le_bytes(payload[i..i + 4].try_into().unwrap()) as usize;
                    let (n_keep, n_discard, cur_pos) = (rd(0), rd(4), rd(8));
                    engine.n_keep = n_keep;
                    engine.cur_pos = cur_pos.min(engine.max_seq);
                    engine.shift_kv(n_discard);
                }
                cs.send_frame(FRAME_KV_SHIFT, WireFormat::Fp32, &[])?;
            }
            FRAME_PREFILL_REQ => {
                if payload.len() < 8 {
                    continue;
                }
                let t_req = Instant::now();
                let n_tokens = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
                let start_pos = u32::from_le_bytes(payload[4..8].try_into().unwrap()) as usize;
                if payload.len() < 8 + n_tokens * 4 {
                    return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "truncated prefill request"));
                }
                let mut tokens = Vec::with_capacity(n_tokens);
                for i in 0..n_tokens {
                    let tid = u32::from_le_bytes(payload[8 + i * 4..12 + i * 4].try_into().unwrap()) as usize;
                    tokens.push(tid);
                }

                // The master owns the sliding window (it sends FRAME_KV_SHIFT); a worker that
                // shifted on its own would desync positions. Refuse instead of writing OOB.
                check_kv_bounds(engine, start_pos, tokens.len())?;
                let hiddens = unsafe { engine.forward_batch_slice(Some(&tokens), None, start_pos) };
                engine.cur_pos = start_pos + tokens.len();

                let mut resp_payload = Vec::new();
                resp_payload.extend_from_slice(&(n_tokens as u32).to_le_bytes());
                resp_payload.extend_from_slice(&(engine.cfg.dim as u32).to_le_bytes());

                for h in &hiddens {
                    pack_activation(h, wire_format, &mut pack_scratch);
                    resp_payload.extend_from_slice(&pack_scratch);
                }

                compute_us += t_req.elapsed().as_micros() as u64;
                requests += 1;
                cs.send_frame(FRAME_PREFILL_RESP, wire_format, &resp_payload)?;
            }
            FRAME_STEP_REQ => {
                let req = match StepRequestPayload::from_bytes(&payload) {
                    Some(r) => r,
                    None => continue,
                };

                let t_req = Instant::now();
                check_kv_bounds(engine, req.pos, 1)?;
                let act = unsafe { engine.forward_slice(Some(req.token_id), None, req.pos) };
                engine.cur_pos = req.pos + 1;

                pack_activation(&act, wire_format, &mut pack_scratch);
                compute_us += t_req.elapsed().as_micros() as u64;
                requests += 1;
                cs.send_frame(FRAME_STEP_RESP, wire_format, &pack_scratch)?;
                if requests % 512 == 0 {
                    engine.maybe_save_expert_stats();
                }
            }
            FRAME_SHUTDOWN => {
                println!("  Client ended the session.");
                let _ = engine.save_expert_stats();
                break;
            }
            other => {
                eprintln!("  Ignoring unknown message type {other} (is the client running a newer spindle?)");
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    #[test]
    fn test_wire_header_round_trip() {
        let mut hdr = WireHeader {
            magic: MAGIC_SPIN,
            frame_type: FRAME_STEP_REQ,
            wire_format: WireFormat::Fp16 as u8,
            flags: 0x42,
            payload_len: 4096,
            checksum: 0,
        };
        hdr.checksum = hdr.compute_checksum();
        let bytes = hdr.to_bytes();
        let parsed = WireHeader::from_bytes(&bytes).expect("valid header");
        assert_eq!(parsed.magic, MAGIC_SPIN);
        assert_eq!(parsed.frame_type, FRAME_STEP_REQ);
        assert_eq!(parsed.wire_format, WireFormat::Fp16 as u8);
        assert_eq!(parsed.payload_len, 4096);
        assert_eq!(parsed.checksum, hdr.checksum);
    }

    #[test]
    fn test_fp16_pack_unpack_avx2() {
        let original: Vec<f32> = (0..2048)
            .map(|i| (i as f32) * 0.005 - 5.0)
            .collect();
        let mut wire_bytes = Vec::new();
        pack_activation(&original, WireFormat::Fp16, &mut wire_bytes);
        assert_eq!(wire_bytes.len(), 2048 * 2);

        let mut recovered = vec![0.0f32; 2048];
        unpack_activation(&wire_bytes, WireFormat::Fp16, &mut recovered).expect("unpack");

        for (i, (&orig, &rec)) in original.iter().zip(&recovered).enumerate() {
            let diff = (orig - rec).abs();
            assert!(
                diff < 5e-3,
                "FP16 mismatch at {i}: orig={orig}, recovered={rec}, diff={diff}"
            );
        }
    }

    #[test]
    fn test_fp8_pack_unpack() {
        let original: Vec<f32> = (0..2048)
            .map(|i| (i as f32) * 0.005 - 5.0)
            .collect();
        let mut wire_bytes = Vec::new();
        pack_activation(&original, WireFormat::Fp8, &mut wire_bytes);
        assert_eq!(wire_bytes.len(), 4 + 2048); // 4-byte scale + 2048 bytes

        let mut recovered = vec![0.0f32; 2048];
        unpack_activation(&wire_bytes, WireFormat::Fp8, &mut recovered).expect("unpack");

        for (i, (&orig, &rec)) in original.iter().zip(&recovered).enumerate() {
            let diff = (orig - rec).abs();
            assert!(
                diff < 0.05,
                "FP8 mismatch at {i}: orig={orig}, recovered={rec}, diff={diff}"
            );
        }
    }

    #[test]
    fn test_cluster_loopback_distributed_inference() {
        let path = "smollm2-135m-q4_k_m.gguf";
        if !std::path::Path::new(path).exists() {
            return;
        }

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
        let local_addr = listener.local_addr().expect("local addr");

        // Spawn worker node on background thread with layers 0..15
        let handle = std::thread::spawn(move || {
            let mut worker_engine = SmolLm2Engine::load_sharded(path, 0..15).expect("load worker");
            let (stream, _) = listener.accept().expect("accept");
            serve_worker_connection(&mut worker_engine, stream, WireFormat::Fp16).expect("serve");
        });

        // Run Master on main thread with layers 15..30
        let mut master_engine = SmolLm2Engine::load_sharded(path, 15..30).expect("load master");
        let mut client = match ClusterClient::connect(
            local_addr,
            &master_engine.idx.model_name,
            master_engine.cfg.dim,
            master_engine.cfg.n_layers,
            (15, 30),
            WireFormat::Fp16,
        ) {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                println!("[test:cluster] Skipping live socket test (sandbox network isolation active: {e})");
                return;
            }
            Err(e) => panic!("client connect error: {e}"),
        };

        let rtt = client.ping().expect("ping");
        println!("[test:cluster] Loopback TCP ping RTT: {:.2} us", rtt.as_secs_f64() * 1e6);

        let prompt = "Rust programming";
        let tokens = master_engine.encode_chatml(prompt);
        assert!(!tokens.is_empty());

        // Prefill across the cluster socket!
        let worker_hiddens = client.prefill(&tokens, 0).expect("prefill");
        assert_eq!(worker_hiddens.len(), tokens.len());

        master_engine.ensure_kv_room(tokens.len());
        let master_hiddens = unsafe {
            master_engine.forward_batch_slice(None, Some(&worker_hiddens), 0)
        };
        master_engine.cur_pos = tokens.len();
        assert_eq!(master_hiddens.len(), tokens.len());

        let last_hidden = master_hiddens.last().unwrap();
        let mut recent = tokens.clone();
        let first_gen = unsafe {
            master_engine.sample_next_token(last_hidden, &recent, false, true)
        };
        recent.push(first_gen);

        // Run 4 more steps across the cluster socket!
        let mut cur_token = first_gen;
        for _ in 0..4 {
            let pos = master_engine.cur_pos;
            let worker_act = client.step(cur_token, pos).expect("step");
            master_engine.ensure_kv_room(1);
            let master_act = unsafe {
                master_engine.forward_slice(None, Some(&worker_act), pos)
            };
            master_engine.cur_pos += 1;
            let next_token = unsafe {
                master_engine.sample_next_token(&master_act, &recent, false, true)
            };
            recent.push(next_token);
            cur_token = next_token;
        }

        // Shut down worker
        let _ = client.stream.send_frame(FRAME_SHUTDOWN, WireFormat::Fp32, &[]);
        let _ = handle.join();

        let generated_text: String = recent[tokens.len()..]
            .iter()
            .map(|&tid| String::from_utf8_lossy(&master_engine.token_bytes(tid)).into_owned())
            .collect();
        println!("[test:cluster] Distributed Generation Success across socket! Output: {generated_text:?}");
        assert!(!generated_text.is_empty());
    }

    /// Runs one greedy reply on a loopback 2-node cluster (worker 0..mid, master mid..n, FP8
    /// wire like production). Returns `(reply, tok/s, spec_stats, round_trips)`.
    fn cluster_reply(path: &'static str, speculate: bool, prompt: &str, n: usize) -> Option<(String, f64, (usize, usize, usize), u64)> {
        let listener = TcpListener::bind("127.0.0.1:0").ok()?;
        let addr = listener.local_addr().ok()?;
        let n_layers = SmolLm2Engine::load_sharded(path, 0..1).ok()?.cfg.n_layers;
        let mid = n_layers / 2;
        let handle = std::thread::spawn(move || {
            let mut w = SmolLm2Engine::load_sharded(path, 0..mid).expect("load worker");
            let (stream, _) = listener.accept().expect("accept");
            let _ = serve_worker_connection(&mut w, stream, WireFormat::Fp8);
        });
        let mut m = SmolLm2Engine::load_sharded(path, mid..n_layers).expect("load master");
        let client = match ClusterClient::connect(addr, &m.idx.model_name, m.cfg.dim, n_layers, (mid, n_layers), WireFormat::Fp8) {
            Ok(c) => c,
            Err(e) => {
                println!("[test:cluster] Skipping live socket test: {e}");
                return None;
            }
        };
        m.attach_cluster(client);
        m.speculate = speculate;
        m.set_sampler(crate::smollm_chat::SamplerConfig { temperature: 0.0, repeat_penalty: 1.0, seed: 1, ..Default::default() });
        let (reply, secs, toks) = unsafe { m.generate_reply(prompt, n, false, &mut |_| {}) };
        let stats = (m.spec_stats, m.timing.steps);
        if let Some(mut c) = m.cluster_client.take() {
            let _ = c.stream.send_frame(FRAME_SHUTDOWN, WireFormat::Fp32, &[]);
        }
        let _ = handle.join();
        Some((reply, toks as f64 / secs, stats.0, stats.1))
    }

    /// Prompt-lookup speculation over the real TCP worker path: same reply as plain decoding,
    /// with fewer network round trips (each verify batch is one PREFILL frame).
    #[test]
    fn test_cluster_loopback_prompt_lookup_is_exact() {
        let path = "qwen3-0.6b-q8_0.gguf";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let prompt = "Copy this Rust function exactly, inside a rust code block:\n\n\
            fn fibonacci(n: u64) -> u64 {\n    let mut a = 0u64;\n    let mut b = 1u64;\n    for _ in 0..n {\n        let t = a + b;\n        a = b;\n        b = t;\n    }\n    a\n}\n";
        let Some((off, tps_off, _, rt_off)) = cluster_reply(path, false, prompt, 96) else { return };
        let Some((on, tps_on, st, rt_on)) = cluster_reply(path, true, prompt, 96) else { return };
        println!(
            "[cluster:lookup] off {tps_off:.1} tok/s, {rt_off} round trips | on {tps_on:.1} tok/s, {rt_on} round trips | accepted {}/{} in {} verifies",
            st.0, st.1, st.2
        );
        assert_eq!(off, on, "speculative cluster reply diverged");
        assert!(st.0 > 0 && rt_on < rt_off, "expected accepted drafts and fewer round trips");
    }

    /// A long conversation forces sliding-window KV shifts. The worker must mirror the
    /// master's shift exactly: a 2-node run (FP32 wire, so activations are bit-exact) must
    /// produce the same replies as one node with the same context size. The worker has the
    /// smaller KV cache here, so this also checks the context clamp in `attach_cluster`.
    #[test]
    fn test_cluster_kv_shift_stays_in_sync() {
        let path = "qwen3-0.6b-q8_0.gguf";
        if !std::path::Path::new(path).exists() {
            return;
        }
        const CTX: usize = 128;
        let turns = ["Count from one to twenty in words.", "Now list five colors.", "Name three planets and one fact each."];
        let greedy = crate::smollm_chat::SamplerConfig { temperature: 0.0, repeat_penalty: 1.0, seed: 1, ..Default::default() };

        let mut single = SmolLm2Engine::load(path).expect("load");
        single.max_seq = CTX;
        single.set_sampler(greedy);
        let want: Vec<String> = turns.iter().map(|t| unsafe { single.generate_reply(t, 48, false, &mut |_| {}).0 }).collect();
        drop(single);

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().unwrap();
        let n_layers = 28;
        let mid = n_layers / 2;
        let handle = std::thread::spawn(move || {
            let mut w = SmolLm2Engine::load_sharded(path, 0..mid).expect("load worker");
            w.max_seq = CTX;
            let (stream, _) = listener.accept().expect("accept");
            let _ = serve_worker_connection(&mut w, stream, WireFormat::Fp32);
        });
        let mut m = SmolLm2Engine::load_sharded(path, mid..n_layers).expect("load master");
        let client = match ClusterClient::connect(addr, &m.idx.model_name, m.cfg.dim, n_layers, (mid, n_layers), WireFormat::Fp32) {
            Ok(c) => c,
            Err(e) => {
                println!("[test:cluster] Skipping live socket test: {e}");
                return;
            }
        };
        assert_eq!(client.peer_max_seq, Some(CTX));
        m.attach_cluster(client);
        assert_eq!(m.max_seq, CTX, "master context must shrink to the worker's capacity");
        m.set_sampler(greedy);
        let got: Vec<String> = turns.iter().map(|t| unsafe { m.generate_reply(t, 48, false, &mut |_| {}).0 }).collect();
        let shifts = m.kv_shift_count();
        if let Some(mut c) = m.cluster_client.take() {
            let _ = c.stream.send_frame(FRAME_SHUTDOWN, WireFormat::Fp32, &[]);
        }
        let _ = handle.join();
        println!("[cluster:kv-shift] {shifts} shifts | replies: {got:?}");
        assert!(shifts > 0, "test must actually exercise a KV shift");
        assert_eq!(got, want, "cluster replies diverged from single-node after KV shifts");
    }

    #[test]
    fn test_cluster_loopback_speculative_decoding() {
        let path = "smollm2-135m-q4_k_m.gguf";
        if !std::path::Path::new(path).exists() {
            return;
        }

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
        let local_addr = listener.local_addr().expect("local addr");

        // Spawn worker node on background thread with layers 0..15
        let handle = std::thread::spawn(move || {
            let mut worker_engine = SmolLm2Engine::load_sharded(path, 0..15).expect("load worker");
            let (stream, _) = listener.accept().expect("accept");
            serve_worker_connection(&mut worker_engine, stream, WireFormat::Fp16).expect("serve");
        });

        // Run Master on main thread with layers 15..30
        let mut master_engine = SmolLm2Engine::load_sharded(path, 15..30).expect("load master");
        let client = match ClusterClient::connect(
            local_addr,
            &master_engine.idx.model_name,
            master_engine.cfg.dim,
            master_engine.cfg.n_layers,
            (15, 30),
            WireFormat::Fp16,
        ) {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                println!("[test:cluster] Skipping live socket test (sandbox network isolation active: {e})");
                return;
            }
            Err(e) => panic!("client connect error: {e}"),
        };

        master_engine.attach_cluster(client);
        let draft_engine = SmolLm2Engine::load(path).expect("load draft");
        master_engine.enable_draft(draft_engine);

        unsafe {
            let (reply, secs, toks) = master_engine.generate_reply("Hi", 8, false, &mut |_| {});
            println!("[cluster:spec:test] {toks} tokens in {secs:.3}s | stats: {:?} | reply: {reply:?}", master_engine.spec_stats);
            assert!(master_engine.spec_stats.0 > 0, "Expected accepted draft tokens across cluster");
        }

        if let Some(mut client) = master_engine.cluster_client.take() {
            let _ = client.stream.send_frame(FRAME_SHUTDOWN, WireFormat::Fp32, &[]);
        }
        let _ = handle.join();
    }
}

