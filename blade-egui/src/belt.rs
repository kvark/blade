struct ReusableBuffer {
    raw: blade_graphics::Buffer,
    size: u64,
}

pub struct BeltDescriptor {
    pub memory: blade_graphics::Memory,
    pub min_chunk_size: u64,
}

/// A belt of buffers, used by the EguiPainter to cheaply
/// find staging space for uploads.
pub struct BufferBelt {
    desc: BeltDescriptor,
    buffers: Vec<(ReusableBuffer, blade_graphics::SyncPoint)>,
    active: Vec<(ReusableBuffer, u64)>,
}

impl BufferBelt {
    pub fn new(desc: BeltDescriptor) -> Self {
        Self {
            desc,
            buffers: Vec::new(),
            active: Vec::new(),
        }
    }

    pub fn destroy(&mut self, context: &blade_graphics::Context) {
        for (buffer, _) in self.buffers.drain(..) {
            context.destroy_buffer(buffer.raw);
        }
        for (buffer, _) in self.active.drain(..) {
            context.destroy_buffer(buffer.raw);
        }
    }

    pub fn alloc(
        &mut self,
        size: u64,
        context: &blade_graphics::Context,
    ) -> blade_graphics::BufferPiece {
        for &mut (ref rb, ref mut offset) in self.active.iter_mut() {
            if *offset + size <= rb.size {
                let piece = rb.raw.at(*offset);
                *offset += size;
                return piece;
            }
        }

        let index_maybe = self
            .buffers
            .iter()
            .position(|&(ref rb, ref sp)| size <= rb.size && context.wait_for(sp, 0));
        if let Some(index) = index_maybe {
            let (rb, _) = self.buffers.remove(index);
            let piece = rb.raw.into();
            self.active.push((rb, size));
            return piece;
        }

        let chunk_index = self.buffers.len() + self.active.len();
        let chunk_size = size.max(self.desc.min_chunk_size);
        let chunk = context.create_buffer(blade_graphics::BufferDesc {
            name: &format!("chunk-{}", chunk_index),
            size: chunk_size,
            memory: self.desc.memory,
        });
        let rb = ReusableBuffer {
            raw: chunk,
            size: chunk_size,
        };
        self.active.push((rb, size));
        chunk.into()
    }

    pub fn alloc_data<T: bytemuck::Pod>(
        &mut self,
        data: &[T],
        context: &blade_graphics::Context,
    ) -> blade_graphics::BufferPiece {
        let bp = self.alloc((data.len() * std::mem::size_of::<T>()) as u64, context);
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), bp.data() as *mut T, data.len());
        }
        bp
    }

    pub fn flush(&mut self, sp: &blade_graphics::SyncPoint) {
        self.buffers
            .extend(self.active.drain(..).map(|(rb, _)| (rb, sp.clone())));
    }
}
