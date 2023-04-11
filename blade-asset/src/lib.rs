use std::{
    collections::hash_map::{Entry, HashMap},
    ops,
    path::{Path, PathBuf},
    sync::Mutex,
};

mod arena;

pub struct Handle<A> {
    inner: arena::Handle<Slot<A>>,
    generation: u32,
}
impl<A> Clone for Handle<A> {
    fn clone(&self) -> Self {
        Handle {
            inner: self.inner,
            generation: self.generation,
        }
    }
}
impl<A> Copy for Handle<A> {}

#[derive(Default)]
struct Slot<A> {
    load_task: Option<choir::RunningTask>,
    generation: u32,
    asset: A,
}

struct AssetManager<A> {
    root: PathBuf,
    slots: arena::Arena<Slot<A>>,
    paths: Mutex<HashMap<PathBuf, Handle<A>>>,
}

impl<A> ops::Index<Handle<A>> for AssetManager<A> {
    type Output = A;
    fn index(&self, handle: Handle<A>) -> &A {
        let slot = &self.slots[handle.inner];
        assert_eq!(handle.generation, slot.generation);
        &slot.asset
    }
}

impl<A: Default> AssetManager<A> {
    pub fn new(root: &Path) -> Self {
        Self {
            root: root.to_path_buf(),
            slots: arena::Arena::new(64),
            paths: Mutex::default(),
        }
    }

    pub fn load(&self, path: &Path, choir: &choir::Choir) -> (Handle<A>, &choir::RunningTask) {
        let mut paths = self.paths.lock().unwrap();
        let handle = match paths.entry(path.to_path_buf()) {
            Entry::Occupied(e) => *e.get(),
            Entry::Vacant(e) => {
                let generation = 0;
                let handle = self.slots.alloc(Slot {
                    load_task: None, //TODO
                    generation,
                    asset: A::default(),
                });
                *e.insert(Handle {
                    inner: handle,
                    generation,
                })
            }
        };
        let task = self.slots[handle.inner].load_task.as_ref().unwrap();
        (handle, task)
    }
}
