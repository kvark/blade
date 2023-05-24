use std::{borrow::Cow, mem, num::NonZeroUsize, ptr, slice};

pub trait Flat {
    /// Type alignment, must be a power of two.
    const ALIGNMENT: usize;
    /// Size of the type, only if it's fixed.
    const FIXED_SIZE: Option<NonZeroUsize>;
    /// Size of the object.
    fn size(&self) -> usize {
        Self::FIXED_SIZE.unwrap().get()
    }
    /// Write self at the specified pointer.
    /// The pointer is guaranteed to be valid and aligned accordingly.
    ///
    /// # Safety
    /// Only safe if the available space in `ptr` is at least `self.size()`
    unsafe fn write(&self, ptr: *mut u8);
    /// Read self from the specified pointer.
    /// The pointer is guaranteed to be valid and aligned accordingly.
    ///
    /// # Safety
    /// Only safe when the `ptr` points to the same data as previously
    /// was written with [`Flat::write`].
    unsafe fn read(ptr: *const u8) -> Self;
}

macro_rules! impl_basic {
    ($ty: ident) => {
        impl Flat for $ty {
            const ALIGNMENT: usize = mem::align_of::<Self>();
            const FIXED_SIZE: Option<NonZeroUsize> = NonZeroUsize::new(mem::size_of::<Self>());
            unsafe fn write(&self, ptr: *mut u8) {
                ptr::write(ptr as *mut Self, *self);
            }
            unsafe fn read(ptr: *const u8) -> Self {
                ptr::read(ptr as *const Self)
            }
        }
    };
}

impl_basic!(u32);
impl_basic!(u64);
impl_basic!(f32);

/*
impl<T: bytemuck::Pod> Flat for T {
    const ALIGNMENT: usize = mem::align_of::<T>();
    const FIXED_SIZE: Option<NonZeroUsize> = NonZeroUsize::new(mem::size_of::<T>());
    unsafe fn write(&self, ptr: *mut u8) {
        ptr::write(ptr as *mut T, *self);
    }
    unsafe fn read(ptr: *const u8) {
        ptr::read(ptr as *const T)
    }
}*/

pub fn round_up(size: usize, alignment: usize) -> usize {
    (size + alignment - 1) & !(alignment - 1)
}

impl<T: Flat> Flat for Vec<T> {
    const ALIGNMENT: usize = T::ALIGNMENT;
    const FIXED_SIZE: Option<NonZeroUsize> = None;
    fn size(&self) -> usize {
        self.iter().fold(mem::size_of::<usize>(), |offset, item| {
            round_up(offset, T::ALIGNMENT) + item.size()
        })
    }
    unsafe fn write(&self, ptr: *mut u8) {
        ptr::write(ptr as *mut usize, self.len());
        let mut offset = mem::size_of::<usize>();
        for item in self.iter() {
            offset = round_up(offset, T::ALIGNMENT);
            item.write(ptr.add(offset));
            offset += item.size();
        }
    }
    unsafe fn read(ptr: *const u8) -> Self {
        let counter = ptr::read(ptr as *const usize);
        let mut offset = mem::size_of::<usize>();
        (0..counter)
            .map(|_| {
                offset = round_up(offset, T::ALIGNMENT);
                let value = T::read(ptr.add(offset));
                offset += value.size();
                value
            })
            .collect()
    }
}

impl<T: bytemuck::Pod, const C: usize> Flat for [T; C] {
    const ALIGNMENT: usize = mem::align_of::<T>();
    const FIXED_SIZE: Option<NonZeroUsize> = NonZeroUsize::new(mem::size_of::<Self>());
    unsafe fn write(&self, ptr: *mut u8) {
        ptr::copy_nonoverlapping(self.as_ptr(), ptr as *mut T, C);
    }
    unsafe fn read(ptr: *const u8) -> Self {
        ptr::read(ptr as *const Self)
    }
}

impl<'a, T: bytemuck::Pod> Flat for &'a [T] {
    const ALIGNMENT: usize = mem::align_of::<T>();
    const FIXED_SIZE: Option<NonZeroUsize> = None;
    fn size(&self) -> usize {
        let elem_size = round_up(mem::size_of::<T>(), mem::align_of::<T>());
        round_up(mem::size_of::<usize>(), mem::align_of::<T>()) + elem_size * self.len()
    }
    unsafe fn write(&self, ptr: *mut u8) {
        ptr::write(ptr as *mut usize, self.len());
        if !self.is_empty() {
            let offset = round_up(mem::size_of::<usize>(), mem::align_of::<T>());
            ptr::copy_nonoverlapping(self.as_ptr(), ptr.add(offset) as *mut T, self.len());
        }
    }
    unsafe fn read(ptr: *const u8) -> Self {
        let counter = ptr::read(ptr as *const usize);
        if counter != 0 {
            let offset = round_up(mem::size_of::<usize>(), mem::align_of::<T>());
            slice::from_raw_parts(ptr.add(offset) as *const T, counter)
        } else {
            &[]
        }
    }
}

impl<'a, T: bytemuck::Pod> Flat for Cow<'a, [T]> {
    const ALIGNMENT: usize = mem::align_of::<T>();
    const FIXED_SIZE: Option<NonZeroUsize> = None;
    fn size(&self) -> usize {
        self.as_ref().size()
    }
    unsafe fn write(&self, ptr: *mut u8) {
        self.as_ref().write(ptr)
    }
    unsafe fn read(ptr: *const u8) -> Self {
        Cow::Borrowed(<&'a [T] as Flat>::read(ptr))
    }
}
