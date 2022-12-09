use blade::BufferPiece;

struct ReusableBuffer {
    raw: blade::Buffer,
    size: u64,
}

pub struct BeltDescriptor {
    pub memory: blade::Memory,
    pub min_chunk_size: u64,
}

pub struct BufferBelt {
    desc: BeltDescriptor,
    buffers: Vec<(ReusableBuffer, blade::SyncPoint)>,
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

    pub fn alloc(&mut self, size: u64, context: &blade::Context) -> blade::BufferPiece {
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
        let chunk = context.create_buffer(blade::BufferDesc {
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

    pub fn flush(&mut self, sp: blade::SyncPoint) {
        self.buffers
            .extend(self.active.drain(..).map(|(rb, _)| (rb, sp.clone())));
    }
}
