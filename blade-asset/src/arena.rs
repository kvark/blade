use std::{
    marker::PhantomData,
    num::NonZeroU32,
    ops,
    sync::{
        atomic::{AtomicPtr, Ordering},
        Mutex,
    },
};

#[repr(C)]
struct Address {
    index: NonZeroU32,
    chunk: u8,
}

pub struct Handle<T>(Address, PhantomData<T>);

const MAX_CHUNKS: usize = 30;

#[derive(Default)]
struct FreeManager<T> {
    chunk_bases: Vec<*mut [T]>,
    free_list: Vec<Address>,
}

pub struct Arena<T> {
    init_size: usize,
    chunks: [AtomicPtr<T>; MAX_CHUNKS],
    freeman: Mutex<FreeManager<T>>,
}

impl<T> ops::Index<Handle<T>> for Arena<T> {
    type Output = T;
    fn index(&self, handle: Handle<T>) -> &T {
        let first_ptr = &self.chunks[handle.0.chunk as usize].load(Ordering::Acquire);
        unsafe { &*first_ptr.add(handle.0.index.get() as usize) }
    }
}

impl<T: Default> Arena<T> {
    pub fn new(init_size: usize) -> Self {
        Self {
            init_size,
            chunks: Default::default(),
            freeman: Mutex::default(),
        }
    }

    pub fn alloc(&self) -> Handle<T> {
        let mut freeman = self.freeman.lock().unwrap();
        let address = match freeman.free_list.pop() {
            Some(address) => address,
            None => {
                let address = Address {
                    index: NonZeroU32::new(1).unwrap(),
                    chunk: freeman.chunk_bases.len() as _,
                };
                let size = self.init_size << freeman.chunk_bases.len();
                let mut data = (0..size).map(|_| T::default()).collect::<Box<[T]>>();
                self.chunks[address.chunk as usize]
                    .store(data.first_mut().unwrap(), Ordering::Release);
                freeman.chunk_bases.push(Box::into_raw(data));
                address
            }
        };
        Handle(address, PhantomData)
    }
}

// Ensure the `Option<Handle>` doesn't have any overhead
#[cfg(test)]
unsafe fn test_option_handle<A>(handle: Handle<A>) -> Option<Handle<A>> {
    std::mem::transmute(handle)
}
