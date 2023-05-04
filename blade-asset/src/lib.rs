use std::{
    collections::hash_map::{DefaultHasher, Entry, HashMap},
    fmt, fs,
    hash::Hash,
    ops,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

mod arena;
mod flat;

pub use flat::{round_up, Flat};

type Version = u32;

pub struct Handle<T> {
    inner: arena::Handle<Slot<T>>,
    version: Version,
}
impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        Handle {
            inner: self.inner,
            version: self.version,
        }
    }
}
impl<T> Copy for Handle<T> {}

struct DataRef<T>(*mut T, *mut Version);
unsafe impl<T> Send for DataRef<T> {}

#[derive(Default)]
struct Slot<T> {
    load_task: Option<choir::RunningTask>,
    version: Version,
    data: T,
}

pub trait Baker: Send + Sync + 'static {
    type Meta: Clone + Eq + fmt::Debug + Hash + Send;
    type Output: Default + Send;
    fn cook(
        &self,
        src_path: &Path,
        meta: Self::Meta,
        dst_path: &Path,
        exe_context: choir::ExecutionContext,
    );
    fn serve(&self, data: &[u8]) -> Self::Output;
}

pub struct AssetManager<B: Baker> {
    root: PathBuf,
    target: PathBuf,
    slots: arena::Arena<Slot<B::Output>>,
    paths: Mutex<HashMap<(PathBuf, B::Meta), Handle<B::Output>>>,
    choir: Arc<choir::Choir>,
    baker: Arc<B>,
}

impl<B: Baker> ops::Index<Handle<B::Output>> for AssetManager<B> {
    type Output = B::Output;
    fn index(&self, handle: Handle<B::Output>) -> &Self::Output {
        let slot = &self.slots[handle.inner];
        assert_eq!(handle.version, slot.version);
        &slot.data
    }
}

impl<B: Baker> AssetManager<B> {
    pub fn new(root: &Path, target: &Path, choir: &Arc<choir::Choir>, baker: B) -> Self {
        if !target.is_dir() {
            log::info!("Creating target {}", target.display());
            fs::create_dir_all(target).unwrap();
        }
        Self {
            root: root.to_path_buf(),
            target: target.to_path_buf(),
            slots: arena::Arena::new(64),
            paths: Mutex::default(),
            choir: Arc::clone(choir),
            baker: Arc::new(baker),
        }
    }

    fn create(&self, relative_path: &Path, meta: B::Meta) -> Handle<B::Output> {
        use base64::engine::{general_purpose::URL_SAFE as ENCODING_ENGINE, Engine as _};
        use std::hash::Hasher as _;

        let source_path = self.root.join(relative_path);
        let metadata = fs::metadata(&source_path).unwrap();
        assert!(metadata.is_file());
        let mut hasher = DefaultHasher::new();
        metadata.modified().unwrap().hash(&mut hasher);
        let target_path = {
            let hash = hasher.finish().to_le_bytes();
            let mut file_name = format!("{}-", relative_path.display());
            ENCODING_ENGINE.encode_string(hash, &mut file_name);
            file_name += ".raw";
            self.target.join(file_name)
        };

        let (handle, slot_ptr) = self.slots.alloc_default();
        let (task_option, data_ref) = unsafe {
            let slot = &mut *slot_ptr;
            (
                &mut slot.load_task,
                DataRef(&mut slot.data, &mut slot.version),
            )
        };
        let version = 1;

        let mut load_task = {
            let baker = Arc::clone(&self.baker);
            let target_path = target_path.clone();
            self.choir
                .spawn(&format!("load {} with {:?}", relative_path.display(), meta))
                .init(move |_| {
                    let cooked = fs::read(target_path).unwrap();
                    let target = baker.serve(&cooked);
                    let dr = data_ref;
                    unsafe {
                        *dr.0 = target;
                        *dr.1 = version;
                    }
                })
        };

        if !target_path.is_file() {
            let baker = Arc::clone(&self.baker);
            let cook_task = self
                .choir
                .spawn(&format!("cook {} with {:?}", relative_path.display(), meta))
                .init(move |exe_context| {
                    baker.cook(&source_path, meta, &target_path, exe_context);
                });
            load_task.depend_on(&cook_task);
        };

        *task_option = Some(load_task.run());
        Handle {
            inner: handle,
            version,
        }
    }

    pub fn load(&self, path: &Path, meta: B::Meta) -> (Handle<B::Output>, &choir::RunningTask) {
        let mut paths = self.paths.lock().unwrap();
        let handle = match paths.entry((path.to_path_buf(), meta)) {
            Entry::Occupied(e) => *e.get(),
            Entry::Vacant(e) => {
                let handle = self.create(&e.key().0, e.key().1.clone());
                *e.insert(handle)
            }
        };
        let task = self.slots[handle.inner].load_task.as_ref().unwrap();
        (handle, task)
    }
}
