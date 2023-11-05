use std::{
    fmt, hash,
    marker::PhantomData,
    mem,
    num::NonZeroU8,
    ops, ptr,
    sync::{
        atomic::{AtomicPtr, Ordering},
        Mutex,
    },
};

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Hash, Ord)]
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
impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl<T> hash::Hash for Handle<T> {
    fn hash<H: hash::Hasher>(&self, hasher: &mut H) {
        self.0.hash(hasher);
        self.1.hash(hasher);
    }
}
impl<T> fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

const MAX_CHUNKS: usize = 30;

#[derive(Default)]
struct FreeManager<T> {
    chunk_bases: Vec<*mut [T]>,
    free_list: Vec<Address>,
}

// These are safe beacuse members aren't exposed
unsafe impl<T> Send for FreeManager<T> {}
unsafe impl<T> Sync for FreeManager<T> {}

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

    fn chunk_size(&self, chunk: NonZeroU8) -> usize {
        self.min_size << chunk.get()
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
                let size = self.chunk_size(address.chunk);
                let mut data = (0..size).map(|_| T::default()).collect::<Box<[T]>>();
                let chunk_start: *mut T = data.first_mut().unwrap();
                self.chunks[address.chunk.get() as usize].store(chunk_start, Ordering::Release);
                freeman.chunk_bases.push(Box::into_raw(data));
                freeman
                    .free_list
                    .extend((1..size as u32).map(|index| Address { index, ..address }));
                (address, chunk_start)
            }
        };
        unsafe {
            ptr::write(chunk_start.add(address.index as usize), value);
        }
        Handle(address, PhantomData)
    }

    pub fn alloc_default(&self) -> (Handle<T>, *mut T) {
        let handle = self.alloc(T::default());
        (handle, self.get_mut_ptr(handle))
    }

    pub fn get_mut_ptr(&self, handle: Handle<T>) -> *mut T {
        let first_ptr = &self.chunks[handle.0.chunk.get() as usize].load(Ordering::Acquire);
        unsafe { first_ptr.add(handle.0.index as usize) }
    }

    pub fn _dealloc(&self, handle: Handle<T>) -> T {
        let mut freeman = self.freeman.lock().unwrap();
        freeman.free_list.push(handle.0);
        let ptr = self.get_mut_ptr(handle);
        mem::take(unsafe { &mut *ptr })
    }

    fn for_internal(&self, mut fun: impl FnMut(Address, *mut T)) {
        let mut freeman = self.freeman.lock().unwrap();
        freeman.free_list.sort(); // enables fast search
        for (chunk_index, chunk_start) in self.chunks[..freeman.chunk_bases.len()]
            .iter()
            .enumerate()
            .skip(1)
        {
            let first_ptr = chunk_start.load(Ordering::Acquire);
            let chunk = NonZeroU8::new(chunk_index as _).unwrap();
            for index in 0..self.chunk_size(chunk) {
                let address = Address {
                    index: index as u32,
                    chunk,
                };
                if freeman.free_list.binary_search(&address).is_err() {
                    //Note: accessing this is only safe if `get_mut_ptr` isn't called
                    // for example, during hot reloading.
                    fun(address, unsafe { first_ptr.add(index) });
                }
            }
        }
    }

    pub fn for_each(&self, mut fun: impl FnMut(Handle<T>, &T)) {
        self.for_internal(|address, ptr| fun(Handle(address, PhantomData), unsafe { &*ptr }))
    }

    pub fn dealloc_each(&self, mut fun: impl FnMut(Handle<T>, T)) {
        self.for_internal(|address, ptr| {
            fun(
                Handle(address, PhantomData),
                mem::take(unsafe { &mut *ptr }),
            )
        })
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
