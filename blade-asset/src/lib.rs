#![allow(clippy::new_without_default)]
#![warn(
    trivial_casts,
    trivial_numeric_casts,
    unused_extern_crates,
    //TODO: re-enable. Currently doesn't like "mem::size_of" on newer Rust
    //unused_qualifications,
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
    mem, ops,
    path::{Path, PathBuf},
    ptr, str,
    sync::{Arc, Mutex},
};

mod arena;
mod flat;

pub use flat::{round_up, Flat};

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

struct DataRef<T> {
    data: *mut Option<T>,
    version: *mut Version,
    sources: *mut Vec<PathBuf>,
}
unsafe impl<T> Send for DataRef<T> {}

struct Slot<T> {
    load_task: Option<choir::RunningTask>,
    version: Version,
    base_path: PathBuf,
    sources: Vec<PathBuf>,
    // Boxed erased type of metadata
    meta: *const (),
    data: Option<T>,
}
unsafe impl<T> Send for Slot<T> {}
unsafe impl<T> Sync for Slot<T> {}

impl<T> Default for Slot<T> {
    fn default() -> Self {
        Self {
            load_task: None,
            version: 0,
            base_path: PathBuf::default(),
            sources: Vec::new(),
            meta: ptr::null(),
            data: None,
        }
    }
}

#[derive(Default)]
struct Inner {
    result: Vec<u8>,
    dependencies: Vec<PathBuf>,
    hasher: DefaultHasher,
}

#[allow(unused)]
struct CachedSourceDependency {
    relative_path_length: usize,
    relative_path: *const u8,
}
#[allow(unused)]
struct CachedAssetHeader {
    hash: u64,
    data_offset: u64,
    source_dependency_count: usize,
    source_dependencies: *const CachedSourceDependency,
}

/// A container for storing the result of cooking.
///
/// It's meant to live only temporarily during an asset loading.
/// It receives the result of cooking and then delivers it to
/// a task that writes the data to disk.
///
/// Here `T` is the cooked asset type.
pub struct Cooker<B> {
    inner: Mutex<Inner>,
    base_path: PathBuf,
    _phantom: PhantomData<B>,
}
// T doesn't matter for Send/Sync, since we aren't storing it here.
unsafe impl<B> Send for Cooker<B> {}
unsafe impl<B> Sync for Cooker<B> {}

impl<B: Baker> Cooker<B> {
    /// Create a new container with no data.
    pub fn new(base_path: &Path, hasher: DefaultHasher) -> Self {
        Self {
            inner: Mutex::new(Inner {
                result: Vec::new(),
                dependencies: Vec::new(),
                hasher,
            }),
            base_path: base_path.to_path_buf(),
            _phantom: PhantomData,
        }
    }

    /// Create a new container with no data, no path, and no hasher.
    pub fn new_embedded() -> Self {
        Self {
            inner: Mutex::new(Inner::default()),
            base_path: Default::default(),
            _phantom: PhantomData,
        }
    }

    pub fn extract_embedded(&self) -> Vec<u8> {
        let mut inner = self.inner.lock().unwrap();
        assert!(inner.dependencies.is_empty());
        mem::take(&mut inner.result)
    }

    /// Return the base path of the asset.
    pub fn base_path(&self) -> &Path {
        &self.base_path
    }

    /// Put the data into it.
    pub fn finish(&self, value: B::Data<'_>) {
        let mut inner = self.inner.lock().unwrap();
        inner.result = vec![0u8; value.size()];
        unsafe { value.write(inner.result.as_mut_ptr()) };
    }

    /// Read another file as a dependency.
    pub fn add_dependency(&self, relative_path: &Path) -> Vec<u8> {
        let mut inner = self.inner.lock().unwrap();
        inner.dependencies.push(relative_path.to_path_buf());
        let full_path = self.base_path.join(relative_path);
        match fs::File::open(&full_path) {
            Ok(mut file) => {
                // Read the file at the same time as we include the hash
                // of its modification time in the header.
                let mut buf = Vec::new();
                file.metadata()
                    .unwrap()
                    .modified()
                    .unwrap()
                    .hash(&mut inner.hasher);
                file.read_to_end(&mut buf).unwrap();
                buf
            }
            Err(e) => panic!("Unable to read {}: {:?}", full_path.display(), e),
        }
    }
}

/// Baker class abstracts over asset-specific logic.
pub trait Baker: Sized + Send + Sync + 'static {
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
        cooker: Arc<Cooker<Self>>,
        exe_context: &choir::ExecutionContext,
    );
    /// Produce the output bsed on a cooked asset.
    ///
    /// This method is also called within a task `exe_context`.
    fn serve(&self, cooked: Self::Data<'_>, exe_context: &choir::ExecutionContext) -> Self::Output;
    /// Delete the output of an asset.
    fn delete(&self, output: Self::Output);
}

#[derive(Debug)]
enum InvalidDependency {
    MalformedPath,
    DoesntExist,
    NotFile,
}

#[derive(Debug)]
enum CookReason {
    NoTarget,
    BadHeader,
    #[allow(dead_code)]
    TooManyDependencies(usize),
    #[allow(dead_code)]
    Dependency(usize, InvalidDependency),
    Outdated,
    WrongDataOffset,
}

#[profiling::function]
#[allow(clippy::read_zero_byte_vec)] // bad warning?
fn check_target_relevancy(
    target_path: &Path,
    base_path: &Path,
    mut hasher: DefaultHasher,
) -> Result<(), CookReason> {
    let mut file = fs::File::open(target_path).map_err(|_| CookReason::NoTarget)?;
    let mut hash_bytes = [0u8; 8];
    file.read_exact(&mut hash_bytes)
        .map_err(|_| CookReason::BadHeader)?;
    let current_hash = u64::from_le_bytes(hash_bytes);
    file.read_exact(&mut hash_bytes)
        .map_err(|_| CookReason::BadHeader)?;
    let data_offset = u64::from_le_bytes(hash_bytes);

    let mut temp_bytes = [0u8; mem::size_of::<usize>()];
    file.read_exact(&mut temp_bytes)
        .map_err(|_| CookReason::BadHeader)?;
    let num_deps = usize::from_le_bytes(temp_bytes);
    if num_deps > 100 {
        return Err(CookReason::TooManyDependencies(num_deps));
    }
    let mut dep_str = Vec::new();
    for i in 0..num_deps {
        file.read_exact(&mut temp_bytes)
            .map_err(|_| CookReason::BadHeader)?;
        let str_len = usize::from_le_bytes(temp_bytes);
        dep_str.resize(str_len, 0u8);
        file.read_exact(&mut dep_str)
            .map_err(|_| CookReason::BadHeader)?;
        let dep_path = base_path.join(
            str::from_utf8(&dep_str)
                .map_err(|_| CookReason::Dependency(i, InvalidDependency::MalformedPath))?,
        );
        let metadata = fs::metadata(dep_path)
            .map_err(|_| CookReason::Dependency(i, InvalidDependency::DoesntExist))?;
        if !metadata.is_file() {
            return Err(CookReason::Dependency(i, InvalidDependency::NotFile));
        }
        metadata.modified().unwrap().hash(&mut hasher);
    }

    if hasher.finish() != current_hash {
        Err(CookReason::Outdated)
    } else if file.stream_position().unwrap() != data_offset {
        Err(CookReason::WrongDataOffset)
    } else {
        Ok(())
    }
}

/// Manager of assets.
///
/// Contains common logic for tracking the `Handle` associations,
/// caching the results of cooking by the path,
/// and scheduling tasks for cooking and serving assets.
pub struct AssetManager<B: Baker> {
    target: PathBuf,
    slots: arena::Arena<Slot<B::Output>>,
    #[allow(clippy::type_complexity)]
    paths: Mutex<HashMap<(PathBuf, B::Meta), Handle<B::Output>>>,
    pub choir: Arc<choir::Choir>,
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
    /// The `target` points to the folder to store cooked assets in.
    pub fn new(target: &Path, choir: &Arc<choir::Choir>, baker: B) -> Self {
        if !target.is_dir() {
            log::info!("Creating target {}", target.display());
            fs::create_dir_all(target).unwrap();
        }
        Self {
            target: target.to_path_buf(),
            slots: arena::Arena::new(64),
            paths: Mutex::default(),
            choir: Arc::clone(choir),
            baker: Arc::new(baker),
        }
    }

    pub fn get_main_source_path(&self, handle: Handle<B::Output>) -> Option<&PathBuf> {
        self.slots[handle.inner].sources.first()
    }

    fn make_target_path(&self, base_path: &Path, file_name: &Path, meta: &B::Meta) -> PathBuf {
        use base64::engine::{general_purpose::URL_SAFE as ENCODING_ENGINE, Engine as _};
        // The name hash includes the parent path and the metadata.
        let mut hasher = DefaultHasher::new();
        base_path.hash(&mut hasher);
        meta.hash(&mut hasher);
        let hash = hasher.finish().to_le_bytes();
        let mut file_name_str = format!("{}-", file_name.display());
        ENCODING_ENGINE.encode_string(hash, &mut file_name_str);
        file_name_str += ".raw";
        self.target.join(file_name_str)
    }

    fn create_impl<'a>(
        &self,
        slot: &'a mut Slot<B::Output>,
        file_name: &Path,
        content: Option<&[u8]>,
    ) -> Option<(u32, &'a choir::RunningTask)> {
        use std::{hash::Hasher as _, io::Write as _};

        let version = slot.version + 1;
        let (task_option, meta, data_ref) = (
            &mut slot.load_task,
            unsafe { &*(slot.meta as *const B::Meta) },
            DataRef {
                data: &mut slot.data,
                version: &mut slot.version,
                sources: &mut slot.sources,
            },
        );

        let target_path = self.make_target_path(&slot.base_path, file_name, meta);
        let file_name = file_name.to_owned();
        let content = content.map(Vec::from);
        let mut hasher = DefaultHasher::new();
        TypeId::of::<B::Data<'static>>().hash(&mut hasher);

        let load_task = if let Err(reason) =
            check_target_relevancy(&target_path, &slot.base_path, hasher.clone())
        {
            log::info!(
                "Cooking {:?}: {} version={}",
                reason,
                file_name.display(),
                version
            );
            let cooker = Arc::new(Cooker::new(&slot.base_path, hasher));
            let cooker_arg = Arc::clone(&cooker);
            let baker = Arc::clone(&self.baker);
            let mut load_task = self
                .choir
                .spawn(format!("cook finish for {}", file_name.display()))
                .init(move |exe_context| {
                    let mut inner = cooker.inner.lock().unwrap();
                    assert!(!inner.result.is_empty());
                    let mut file = fs::File::create(&target_path).unwrap_or_else(|e| {
                        panic!("Unable to create {}: {}", target_path.display(), e)
                    });
                    file.write_all(&[0; 8]).unwrap(); // write zero hash first
                    file.write_all(&[0; 8]).unwrap(); // write zero data offset
                                                      // write down the dependencies
                    file.write_all(&inner.dependencies.len().to_le_bytes())
                        .unwrap();
                    for dep in inner.dependencies.iter() {
                        let dep_bytes = dep.to_str().unwrap().as_bytes();
                        file.write_all(&dep_bytes.len().to_le_bytes()).unwrap();
                        file.write_all(dep_bytes).unwrap();
                    }
                    let data_offset = file.stream_position().unwrap();
                    file.write_all(&inner.result).unwrap();
                    // Write the real hash last, so that the cached file is not valid
                    // unless everything went smooth.
                    file.seek(SeekFrom::Start(0)).unwrap();
                    let hash = inner.hasher.finish();
                    file.write_all(&hash.to_le_bytes()).unwrap();
                    file.write_all(&data_offset.to_le_bytes()).unwrap();

                    let dr = data_ref;
                    if let Some(data) = unsafe { (*dr.data).take() } {
                        baker.delete(data);
                    }
                    let cooked = unsafe { <B::Data<'_> as Flat>::read(inner.result.as_ptr()) };
                    let target = baker.serve(cooked, &exe_context);
                    unsafe {
                        *dr.data = Some(target);
                        *dr.version = version;
                        *dr.sources = mem::take(&mut inner.dependencies);
                    }
                });

            // Note: this task is separate, because it may spawn sub-tasks.
            let baker = Arc::clone(&self.baker);
            let meta = meta.clone();
            let cook_task = self
                .choir
                .spawn(format!("cook {} as {}", file_name.display(), meta))
                .init(move |exe_context| {
                    // Read the source file through the same mechanism as the
                    // dependencies, so that its modified time makes it into the hash.
                    let extension = file_name.extension().unwrap().to_str().unwrap();
                    let source = match content {
                        Some(data) => data,
                        None => cooker_arg.add_dependency(&file_name),
                    };
                    baker.cook(&source, extension, meta, cooker_arg, &exe_context);
                });

            load_task.depend_on(&cook_task);
            load_task
        } else if task_option.is_none() {
            let baker = Arc::clone(&self.baker);
            self.choir
                .spawn(format!("load {} with {}", file_name.display(), meta))
                .init(move |exe_context| {
                    let mut file = fs::File::open(target_path).unwrap();
                    let mut bytes = [0u8; 8];
                    file.read_exact(&mut bytes).unwrap();
                    let _hash = u64::from_le_bytes(bytes);
                    file.read_exact(&mut bytes).unwrap();
                    let offset = u64::from_le_bytes(bytes);
                    file.seek(SeekFrom::Start(offset)).unwrap();
                    let mut data = Vec::new();
                    file.read_to_end(&mut data).unwrap();
                    let cooked = unsafe { <B::Data<'_> as Flat>::read(data.as_ptr()) };
                    let target = baker.serve(cooked, &exe_context);
                    let dr = data_ref;
                    unsafe {
                        *dr.data = Some(target);
                        *dr.version = version;
                        (*dr.sources).push(file_name);
                    }
                })
        } else {
            return None;
        };

        let running_task = task_option.insert(load_task.run());
        Some((version, running_task))
    }

    fn create(&self, source_path: &Path, meta: B::Meta) -> Handle<B::Output> {
        let (handle, slot_ptr) = self.slots.alloc_default();
        let slot = unsafe { &mut *slot_ptr };
        assert_eq!(slot.version, 0);
        *slot = Slot {
            base_path: source_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_owned(),
            meta: Box::into_raw(Box::new(meta)) as *const _,
            ..Default::default()
        };

        let file_name = Path::new(source_path.file_name().unwrap());
        let (version, _) = self.create_impl(slot, file_name, None).unwrap();
        Handle {
            inner: handle,
            version,
        }
    }

    /// Load an asset given the data directly.
    ///
    /// The `name` must be a pretend file name, with a proper extension.
    ///
    /// Doesn't get cached by the manager, always goes through the cooking process.
    pub fn load_data(
        &self,
        name: &Path,
        data: &[u8],
        meta: B::Meta,
    ) -> (Handle<B::Output>, &choir::RunningTask) {
        let (handle, slot_ptr) = self.slots.alloc_default();
        let slot = unsafe { &mut *slot_ptr };
        assert_eq!(slot.version, 0);
        *slot = Slot {
            meta: Box::into_raw(Box::new(meta)) as *const _,
            ..Default::default()
        };

        let (version, _) = self.create_impl(slot, name, Some(data)).unwrap();

        let task = self.slots[handle].load_task.as_ref().unwrap();
        let out_handle = Handle {
            inner: handle,
            version,
        };
        (out_handle, task)
    }

    /// Load an asset given the relative path.
    ///
    /// Metadata is an asset-specific piece of information that determines how the asset is processed.
    /// Each pair of (path, meta) is cached indepedently.
    ///
    /// This function produces a handle for the asset, and also returns the load task.
    /// It's only valid to access the asset once the load task is completed.
    pub fn load(
        &self,
        path: impl AsRef<Path>,
        meta: B::Meta,
    ) -> (Handle<B::Output>, &choir::RunningTask) {
        let path_buf = path.as_ref().to_path_buf();
        let mut paths = self.paths.lock().unwrap();
        let handle = match paths.entry((path_buf, meta)) {
            Entry::Occupied(e) => *e.get(),
            Entry::Vacant(e) => {
                let handle = self.create(&e.key().0, e.key().1.clone());
                *e.insert(handle)
            }
        };
        let task = self.slots[handle.inner].load_task.as_ref().unwrap();
        (handle, task)
    }

    /// Load an asset that has been pre-cooked already.
    ///
    /// Expected to run inside a task.
    pub fn load_cooked_inside_task(
        &self,
        cooked: B::Data<'_>,
        exe_context: &choir::ExecutionContext,
    ) -> Handle<B::Output> {
        let value = self.baker.serve(cooked, exe_context);
        let (handle, slot_ptr) = self.slots.alloc_default();
        let slot = unsafe { &mut *slot_ptr };
        assert_eq!(slot.version, 0);
        *slot = Slot {
            version: 1,
            data: Some(value),
            ..Slot::default()
        };
        Handle {
            inner: handle,
            version: slot.version,
        }
    }

    /// Clear the asset manager by deleting all the stored assets.
    ///
    /// Invalidates all handles produced from loading assets.
    pub fn clear(&self) {
        self.paths.lock().unwrap().clear();
        self.slots.dealloc_each(|_handle, slot| {
            if let Some(task) = slot.load_task {
                task.join();
            }
            if let Some(data) = slot.data {
                self.baker.delete(data);
            }
            if !slot.meta.is_null() {
                unsafe {
                    let _ = Box::from_raw(slot.meta as *mut B::Meta);
                }
            }
        })
    }

    /// Hot reload a changed asset.
    pub fn hot_reload(&self, handle: &mut Handle<B::Output>) -> Option<&choir::RunningTask> {
        let slot = unsafe { &mut *self.slots.get_mut_ptr(handle.inner) };
        let file_name = slot.sources.first().unwrap().to_owned();
        self.create_impl(slot, &file_name, None)
            .map(|(version, task)| {
                handle.version = version;
                task
            })
    }

    pub fn list_running_tasks(&self, list: &mut Vec<choir::RunningTask>) {
        self.slots.for_each(|_, slot| {
            if let Some(ref task) = slot.load_task {
                if !task.is_done() {
                    list.push(task.clone());
                }
            }
        });
    }
}
