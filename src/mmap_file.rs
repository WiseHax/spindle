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
                CreateFileMappingW, MapViewOfFile, PrefetchVirtualMemory, UnmapViewOfFile,
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
