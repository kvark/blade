use std::{mem, path::PathBuf};

mod arena;

pub use arena::Handle;

struct Slot<A> {
    load_task: Option<choir::RunningTask>,
    iteration: usize,
    asset: A,
}

struct AssetManager<A> {
    root: PathBuf,
    slots: arena::Arena<Slot<A>>,
}

impl<A> AssetManager<A> {}
