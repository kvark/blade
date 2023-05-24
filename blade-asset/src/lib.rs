#![allow(clippy::new_without_default)]
#![warn(
    trivial_casts,
    trivial_numeric_casts,
    unused_extern_crates,
    unused_qualifications,
    // We don't match on a reference, unless required.
    clippy::pattern_type_mismatch,
)]

use std::{
    any::TypeId,
    collections::hash_map::{DefaultHasher, Entry, HashMap},
    fmt, fs,
    hash::{Hash, Hasher},
    io::{Read, Seek as _, SeekFrom},
    marker::PhantomData,
    ops,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

mod arena;
mod flat;

pub use flat::{round_up, Flat};
use syncell::SynCell;

type Version = u32;

/// Handle representing an asset.
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
impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner && self.version == other.version
    }
}
impl<T> Eq for Handle<T> {}
impl<T> Hash for Handle<T> {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.inner.hash(hasher);
    }
}
impl<T> fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Handle")
            .field("inner", &self.inner)
            .field("version", &self.version)
            .finish()
    }
}

struct DataRef<T>(*mut Option<T>, *mut Version);
unsafe impl<T> Send for DataRef<T> {}

struct Slot<T> {
    load_task: Option<choir::RunningTask>,
    version: Version,
    data: Option<T>,
}

impl<T> Default for Slot<T> {
    fn default() -> Self {
        Self {
            load_task: None,
            version: 0,
            data: None,
        }
    }
}

/// A container for storing the result of cooking.
///
/// It's meant to live only temporarily during an asset loading.
/// It receives the result of cooking and then delivers it to
/// a task that writes the data to disk.
///
/// Here `T` is the cooked asset type.
pub struct Cooked<T> {
    //Note: we aren't using the full power of `SynCell` here.
    cell: SynCell<Vec<u8>>,
    _phantom: PhantomData<T>,
}
unsafe impl<T> Send for Cooked<T> {}
unsafe impl<T> Sync for Cooked<T> {}

impl<T: Flat> Cooked<T> {
    /// Create a new container with no data.
    pub fn new() -> Self {
        Self {
            cell: SynCell::new(Vec::new()),
            _phantom: PhantomData,
        }
    }

    /// Put the data into it.
    pub fn put(&self, value: T) {
        let mut data = vec![0u8; value.size()];
        unsafe { value.write(data.as_mut_ptr()) };
        *self.cell.borrow_mut() = data;
    }
}

/// Baker class abstracts over asset-specific logic.
pub trait Baker: Send + Sync + 'static {
    /// Metadata used for loading assets.
    type Meta: Clone + Eq + Hash + Send + fmt::Display;
    /// Intermediate data that is cached, which comes out as a result of cooking.
    type Data<'a>: Flat;
    /// Output type that is produced for the client.
    type Output: Send;
    /// Cook an asset represented by a slice of bytes.
    ///
    /// This method is called within a task within the `exe_context` execution context.
    /// It may fork out other tasks if necessary.
    /// It must put the result into `result` at some point during execution.
    fn cook(
        &self,
        source: &[u8],
        extension: &str,
        meta: Self::Meta,
        result: Arc<Cooked<Self::Data<'_>>>,
        exe_context: choir::ExecutionContext,
    );
    /// Produce the output bsed on a cooked asset.
    ///
    /// This method is also called within a task `exe_context`.
    fn serve(&self, cooked: Self::Data<'_>, exe_context: choir::ExecutionContext) -> Self::Output;
    /// Delete the output of an asset.
    fn delete(&self, output: Self::Output);
}

/// Manager of assets.
///
/// Contains common logic for tracking the `Handle` associations,
/// caching the results of cooking by the path,
/// and scheduling tasks for cooking and serving assets.
pub struct AssetManager<B: Baker> {
    /// Root path where all assets are searched from.
    pub root: PathBuf,
    target: PathBuf,
    slots: arena::Arena<Slot<B::Output>>,
    #[allow(clippy::type_complexity)]
    paths: Mutex<HashMap<(PathBuf, B::Meta), Handle<B::Output>>>,
    choir: Arc<choir::Choir>,
    /// Asset-specific implementation.
    pub baker: Arc<B>,
}

impl<B: Baker> ops::Index<Handle<B::Output>> for AssetManager<B> {
    type Output = B::Output;
    fn index(&self, handle: Handle<B::Output>) -> &Self::Output {
        let slot = &self.slots[handle.inner];
        assert_eq!(handle.version, slot.version, "Outdated {:?}", handle);
        slot.data.as_ref().unwrap()
    }
}

impl<B: Baker> AssetManager<B> {
    /// Create a new asset manager.
    ///
    /// The `root` points to the base path for source data.
    /// The `target` points to the folder to store cooked assets in.
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
        use std::{hash::Hasher as _, io::Write as _};

        let source_path = self.root.join(relative_path);
        let metadata = match fs::metadata(&source_path) {
            Ok(metadata) => {
                assert!(
                    metadata.is_file(),
                    "Source '{}' is not a file",
                    source_path.display()
                );
                metadata
            }
            Err(e) => panic!(
                "Unable to get metadata for '{}': {}",
                source_path.display(),
                e
            ),
        };
        let target_path = {
            let mut hasher = DefaultHasher::new();
            metadata.modified().unwrap().hash(&mut hasher);
            let hash = hasher.finish().to_le_bytes();
            let mut file_name = format!("{}-", relative_path.display());
            ENCODING_ENGINE.encode_string(hash, &mut file_name);
            file_name += ".raw";
            self.target.join(file_name)
        };

        let (handle, slot_ptr) = self.slots.alloc_default();
        let (task_option, output_ref) = unsafe {
            let slot = &mut *slot_ptr;
            (
                &mut slot.load_task,
                DataRef(&mut slot.data, &mut slot.version),
            )
        };
        let version = 1;
        let expected_hash = {
            let mut hasher = DefaultHasher::new();
            TypeId::of::<B::Data<'static>>().hash(&mut hasher);
            hasher.finish()
        };

        let mut load_task = {
            let baker = Arc::clone(&self.baker);
            let target_path = target_path.clone();
            self.choir
                .spawn(format!("load {} with {}", relative_path.display(), meta))
                .init(move |exe_context| {
                    let mut file = fs::File::open(target_path).unwrap();
                    let mut bytes = [0u8; 8];
                    file.read_exact(&mut bytes).unwrap();
                    assert_eq!(u64::from_le_bytes(bytes), expected_hash);
                    let mut data = Vec::new();
                    file.read_to_end(&mut data).unwrap();
                    let cooked = unsafe { <B::Data<'_> as Flat>::read(data.as_ptr()) };
                    let target = baker.serve(cooked, exe_context);
                    let or = output_ref;
                    unsafe {
                        *or.0 = Some(target);
                        *or.1 = version;
                    }
                })
        };

        let current_hash = match fs::File::open(&target_path) {
            Ok(mut file) => {
                let mut bytes = [0u8; 8];
                match file.read_exact(&mut bytes) {
                    Ok(()) => u64::from_le_bytes(bytes),
                    Err(_) => 0,
                }
            }
            Err(_) => 0,
        };

        if current_hash != expected_hash {
            let op_str = if current_hash == 0 {
                "Cooking"
            } else {
                "Recooking"
            };
            log::info!("{} {}", op_str, relative_path.display());

            let result = Arc::new(Cooked::new());
            let result_finish = Arc::clone(&result);
            let mut cook_finish_task = self
                .choir
                .spawn(format!("cook finish for {}", relative_path.display()))
                .init(move |_| {
                    let mut file = fs::File::create(&target_path).unwrap();
                    file.write_all(&[0; 8]).unwrap(); // write zero hash first
                    let data_ref = result_finish.cell.borrow();
                    file.write_all(&data_ref).unwrap();
                    file.seek(SeekFrom::Start(0)).unwrap();
                    // Write the real hash last, so that the cached file is not valid
                    // unless everything went smooth.
                    file.write_all(&expected_hash.to_le_bytes()).unwrap();
                });

            let baker = Arc::clone(&self.baker);
            let cook_task = self
                .choir
                .spawn(format!("cook {} as {}", relative_path.display(), meta))
                .init(move |exe_context| {
                    let source = fs::read(&source_path).unwrap();
                    let extension = source_path.extension().unwrap().to_str().unwrap();
                    baker.cook(&source, extension, meta, result, exe_context);
                });

            cook_finish_task.depend_on(&cook_task);
            load_task.depend_on(&cook_finish_task);
        };

        *task_option = Some(load_task.run());
        Handle {
            inner: handle,
            version,
        }
    }

    /// Load an asset given the relative path from `self.root`.
    ///
    /// Metadata is an asset-specific piece of information that determines how the asset is processed.
    /// Each pair of (path, meta) is cached indepedently.
    ///
    /// This function produces a handle for the asset, and also returns the load task.
    /// It's only valid to access the asset once the load task is completed.
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

    /// Clear the asset manager by deleting all the stored assets.
    ///
    /// Invalidates all handles produced from loading assets.
    pub fn clear(&self) {
        for (_key, handle) in self.paths.lock().unwrap().drain() {
            let slot = self.slots.dealloc(handle.inner);
            if let Some(task) = slot.load_task {
                task.join();
            }
            if let Some(data) = slot.data {
                self.baker.delete(data);
            }
        }
    }
}
