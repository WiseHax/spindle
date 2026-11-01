//! Read-only memory-mapped model files (`CreateFileMappingW` + `MapViewOfFile`).
//!
//! Why mmap instead of reading the GGUF into a `Vec<u8>`:
//! - No copy: weights are used straight from the OS page cache, so startup is instant and RAM
//!   isn't spent twice (page cache + our heap).
//! - The pages are file-backed and clean, so under memory pressure Windows can drop them and
//!   re-read them from the SSD later instead of pushing the system into a swap storm.
//! - Only touched pages ever become resident, the foundation for streaming larger-than-RAM models.

use std::{ffi::c_void, ops::Deref};

use windows::{
    core::{HSTRING, PCWSTR},
    Win32::{
        Foundation::{CloseHandle, GENERIC_READ, HANDLE},
        Storage::FileSystem::{
            CreateFileW, GetFileSizeEx, FILE_ATTRIBUTE_NORMAL, FILE_SHARE_READ, OPEN_EXISTING,
        },
        System::{
            Memory::{
                CreateFileMappingW, MapViewOfFile, PrefetchVirtualMemory, UnmapViewOfFile, VirtualLock,
                FILE_MAP_READ, MEMORY_MAPPED_VIEW_ADDRESS, PAGE_READONLY, WIN32_MEMORY_RANGE_ENTRY,
            },
            ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS},
            SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX},
            Threading::GetCurrentProcess,
        },
    },
};

pub struct MappedFile {
    ptr: *const u8,
    len: usize,
    file: HANDLE,
    mapping: HANDLE,
}

// SAFETY: the view is read-only and lives until `Drop`; sharing `&[u8]` across threads is fine.
unsafe impl Send for MappedFile {}
unsafe impl Sync for MappedFile {}

impl MappedFile {
    pub fn open(path: &str) -> std::io::Result<Self> {
        unsafe {
            let file = CreateFileW(
                &HSTRING::from(path),
                GENERIC_READ.0,
                FILE_SHARE_READ,
                None,
                OPEN_EXISTING,
                FILE_ATTRIBUTE_NORMAL,
                HANDLE::default(),
            )?;
            let mut size = 0i64;
            if let Err(e) = GetFileSizeEx(file, &mut size) {
                let _ = CloseHandle(file);
                return Err(e.into());
            }
            let len = size as usize;
            let mapping = match CreateFileMappingW(file, None, PAGE_READONLY, 0, 0, PCWSTR::null()) {
                Ok(m) => m,
                Err(e) => {
                    let _ = CloseHandle(file);
                    return Err(e.into());
                }
            };
            let view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
            if view.Value.is_null() {
                let err = std::io::Error::last_os_error();
                let _ = CloseHandle(mapping);
                let _ = CloseHandle(file);
                return Err(err);
            }
            Ok(Self { ptr: view.Value as *const u8, len, file, mapping })
        }
    }

    /// Asks Windows to read the whole file into the page cache in large background I/Os,
    /// but only if it comfortably fits (≤ half of currently available RAM). Otherwise pages
    /// are faulted in on demand. Returns whether a prefetch was issued.
    pub fn prefetch_if_fits(&self) -> bool {
        if self.len as u64 > available_ram_bytes() / 2 {
            return false;
        }
        let range = WIN32_MEMORY_RANGE_ENTRY {
            VirtualAddress: self.ptr as *mut c_void,
            NumberOfBytes: self.len,
        };
        unsafe { PrefetchVirtualMemory(GetCurrentProcess(), &[range], 0).is_ok() }
    }

    /// Asks Windows to start reading these `(offset, len)` ranges in the background with
    /// large I/Os, instead of 4 KiB page faults later. Pages go to the standby cache, not
    /// our working set, so this never pushes other apps out. A hint only: errors are ignored.
    pub fn prefetch_ranges(&self, ranges: &[(usize, usize)]) {
        prefetch_raw(self.ptr as usize, self.len, ranges);
    }

    /// Locks `(offset, len)` ranges into our working set (reading them in first). Locked pages
    /// are never trimmed, so they can't be the victims of the scan-order thrash described in
    /// `SmolLm2Engine::pin_hot_weights`. Returns how many bytes were locked. The lock lasts
    /// until the view is unmapped.
    pub fn lock_ranges(&self, ranges: &[(usize, usize)]) -> usize {
        // Read ahead in big I/Os first; VirtualLock would otherwise fault page by page.
        self.prefetch_ranges(ranges);
        ranges
            .iter()
            .filter(|&&(off, len)| len > 0 && off + len <= self.len)
            .filter(|&&(off, len)| unsafe { VirtualLock(self.ptr.add(off) as *const c_void, len).is_ok() })
            .map(|&(_, len)| len)
            .sum()
    }
}

fn prefetch_raw(base: usize, file_len: usize, ranges: &[(usize, usize)]) {
    let entries: Vec<WIN32_MEMORY_RANGE_ENTRY> = ranges
        .iter()
        .filter(|&&(off, len)| len > 0 && off + len <= file_len)
        .map(|&(off, len)| WIN32_MEMORY_RANGE_ENTRY { VirtualAddress: (base + off) as *mut c_void, NumberOfBytes: len })
        .collect();
    if !entries.is_empty() {
        unsafe {
            let _ = PrefetchVirtualMemory(GetCurrentProcess(), &entries, 0);
        }
    }
}

/// Issues prefetch hints from a background thread. `PrefetchVirtualMemory` walks every page
/// in the range before returning, which costs real time even when the pages are already
/// resident; doing it here keeps that off the compute thread. Requests that pile up while
/// the thread is busy are merged into one call.
pub struct Prefetcher {
    tx: Option<std::sync::mpsc::Sender<Vec<(usize, usize)>>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Prefetcher {
    /// # Safety
    /// `file` must stay mapped until this `Prefetcher` is dropped.
    pub unsafe fn new(file: &MappedFile) -> Self {
        let (base, len) = (file.ptr as usize, file.len);
        let (tx, rx) = std::sync::mpsc::channel::<Vec<(usize, usize)>>();
        let handle = std::thread::Builder::new()
            .name("prefetch".into())
            .spawn(move || {
                while let Ok(mut ranges) = rx.recv() {
                    while let Ok(more) = rx.try_recv() {
                        ranges.extend(more);
                    }
                    prefetch_raw(base, len, &ranges);
                }
            })
            .ok();
        Self { tx: handle.as_ref().map(|_| tx), handle }
    }

    pub fn hint(&self, ranges: Vec<(usize, usize)>) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(ranges);
        }
    }
}

impl Drop for Prefetcher {
    fn drop(&mut self) {
        drop(self.tx.take());
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Deref for MappedFile {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        // SAFETY: `ptr..ptr+len` is a live read-only view until `Drop`.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for MappedFile {
    fn drop(&mut self) {
        unsafe {
            let _ = UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: self.ptr as *mut c_void });
            let _ = CloseHandle(self.mapping);
            let _ = CloseHandle(self.file);
        }
    }
}

/// Physical RAM currently available to new allocations, in bytes.
pub fn available_ram_bytes() -> u64 {
    unsafe {
        let mut st = MEMORYSTATUSEX {
            dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32,
            ..Default::default()
        };
        if GlobalMemoryStatusEx(&mut st).is_ok() {
            st.ullAvailPhys
        } else {
            0
        }
    }
}

/// This process's resident set (working set) and private committed bytes, in MiB.
/// Mapped model pages count toward the working set but not the private bytes.
pub fn process_memory_mib() -> (f64, f64) {
    unsafe {
        let mut pmc = PROCESS_MEMORY_COUNTERS {
            cb: std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
            ..Default::default()
        };
        let ok = K32GetProcessMemoryInfo(GetCurrentProcess(), &mut pmc, pmc.cb).as_bool();
        if !ok {
            return (0.0, 0.0);
        }
        let mib = |b: usize| b as f64 / (1024.0 * 1024.0);
        (mib(pmc.WorkingSetSize), mib(pmc.PagefileUsage))
    }
}
