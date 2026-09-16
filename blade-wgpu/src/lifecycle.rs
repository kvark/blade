//! Encoder create/submit/destroy breakdown.
//!
//! Blade's native loop reuses one `CommandEncoder` (`start` → record →
//! `submit`). The wgpu backend was creating a fresh encoder (two 1 MiB
//! scratch buffers plus descriptor pools) per wgpu `CommandEncoder` and
//! destroying it inside `Queue::submit`. These counters separate that
//! allocator traffic from `vkQueueSubmit`.

use std::sync::atomic::{AtomicU64, Ordering};

static CREATE_NS: AtomicU64 = AtomicU64::new(0);
static CREATE_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static RECYCLE_COUNT: AtomicU64 = AtomicU64::new(0);
static VK_SUBMIT_NS: AtomicU64 = AtomicU64::new(0);
static DESTROY_NS: AtomicU64 = AtomicU64::new(0);
static CB_COUNT: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Default)]
pub struct EncoderLifecycle {
    pub create_ns: u64,
    pub create_count: u64,
    pub alloc_count: u64,
    pub recycle_count: u64,
    pub vk_submit_ns: u64,
    pub destroy_ns: u64,
    pub cb_count: u64,
}

pub fn reset() {
    CREATE_NS.store(0, Ordering::Relaxed);
    CREATE_COUNT.store(0, Ordering::Relaxed);
    ALLOC_COUNT.store(0, Ordering::Relaxed);
    RECYCLE_COUNT.store(0, Ordering::Relaxed);
    VK_SUBMIT_NS.store(0, Ordering::Relaxed);
    DESTROY_NS.store(0, Ordering::Relaxed);
    CB_COUNT.store(0, Ordering::Relaxed);
}

pub fn snapshot() -> EncoderLifecycle {
    EncoderLifecycle {
        create_ns: CREATE_NS.load(Ordering::Relaxed),
        create_count: CREATE_COUNT.load(Ordering::Relaxed),
        alloc_count: ALLOC_COUNT.load(Ordering::Relaxed),
        recycle_count: RECYCLE_COUNT.load(Ordering::Relaxed),
        vk_submit_ns: VK_SUBMIT_NS.load(Ordering::Relaxed),
        destroy_ns: DESTROY_NS.load(Ordering::Relaxed),
        cb_count: CB_COUNT.load(Ordering::Relaxed),
    }
}

pub fn take() -> EncoderLifecycle {
    let stats = snapshot();
    reset();
    stats
}

pub(crate) fn note_create(ns: u64, allocated: bool) {
    CREATE_NS.fetch_add(ns, Ordering::Relaxed);
    CREATE_COUNT.fetch_add(1, Ordering::Relaxed);
    if allocated {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
    } else {
        RECYCLE_COUNT.fetch_add(1, Ordering::Relaxed);
    }
}

pub(crate) fn note_vk_submit(ns: u64) {
    VK_SUBMIT_NS.fetch_add(ns, Ordering::Relaxed);
    CB_COUNT.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn note_destroy(ns: u64) {
    DESTROY_NS.fetch_add(ns, Ordering::Relaxed);
}

pub(crate) fn elapsed_ns(start: std::time::Instant) -> u64 {
    start.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64
}
