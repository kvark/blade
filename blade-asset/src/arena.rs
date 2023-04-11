use std::{
    marker::PhantomData,
    num::NonZeroU8,
    ops, ptr,
    sync::{
        atomic::{AtomicPtr, Ordering},
        Mutex,
    },
};

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Hash)]
struct Address {
    index: u32,
    chunk: NonZeroU8,
}

pub struct Handle<T>(Address, PhantomData<T>);
impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        Handle(self.0, PhantomData)
    }
}
impl<T> Copy for Handle<T> {}

const MAX_CHUNKS: usize = 30;

#[derive(Default)]
struct FreeManager<T> {
    chunk_bases: Vec<*mut [T]>,
    free_list: Vec<Address>,
}

pub struct Arena<T> {
    min_size: usize,
    chunks: [AtomicPtr<T>; MAX_CHUNKS],
    freeman: Mutex<FreeManager<T>>,
}

impl<T> ops::Index<Handle<T>> for Arena<T> {
    type Output = T;
    fn index(&self, handle: Handle<T>) -> &T {
        let first_ptr = &self.chunks[handle.0.chunk.get() as usize].load(Ordering::Acquire);
        unsafe { &*first_ptr.add(handle.0.index as usize) }
    }
}

impl<T: Default> Arena<T> {
    pub fn new(min_size: usize) -> Self {
        assert_ne!(min_size, 0);
        let dummy_data = Some(T::default()).into_iter().collect::<Box<[T]>>();
        Self {
            min_size,
            chunks: Default::default(),
            freeman: Mutex::new(FreeManager {
                chunk_bases: vec![Box::into_raw(dummy_data)],
                free_list: Vec::new(),
            }),
        }
    }

    pub fn alloc(&self, value: T) -> Handle<T> {
        let mut freeman = self.freeman.lock().unwrap();
        let (address, chunk_start) = match freeman.free_list.pop() {
            Some(address) => {
                let chunk_start = self.chunks[address.chunk.get() as usize].load(Ordering::Acquire);
                (address, chunk_start)
            }
            None => {
                let address = Address {
                    index: 0,
                    chunk: NonZeroU8::new(freeman.chunk_bases.len() as _).unwrap(),
                };
                let size = self.min_size << freeman.chunk_bases.len();
                let mut data = (0..size).map(|_| T::default()).collect::<Box<[T]>>();
                let chunk_start = data.first_mut().unwrap() as *mut T;
                self.chunks[address.chunk.get() as usize].store(chunk_start, Ordering::Release);
                freeman.chunk_bases.push(Box::into_raw(data));
                (address, chunk_start)
            }
        };
        unsafe {
            ptr::write(chunk_start.add(address.index as usize), value);
        }
        Handle(address, PhantomData)
    }
}

impl<T> Drop for FreeManager<T> {
    fn drop(&mut self) {
        for base in self.chunk_bases.drain(..) {
            let _ = unsafe { Box::from_raw(base) };
        }
    }
}

// Ensure the `Option<Handle>` doesn't have any overhead
#[cfg(test)]
unsafe fn _test_option_handle<A>(handle: Handle<A>) -> Option<Handle<A>> {
    std::mem::transmute(handle)
}

#[test]
fn test_single_thread() {
    let arena = Arena::<usize>::new(1);
    let _ = arena.alloc(3);
    let _ = arena.alloc(4);
}
