use std::{
    collections::HashMap,
    path::{Component, Path, PathBuf},
    sync::Mutex,
};

fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

static FILES: Mutex<Option<HashMap<PathBuf, Vec<u8>>>> = Mutex::new(None);

/// Mount a file into the in-memory asset VFS.
///
/// Used on WASM (no real filesystem) and for generated assets that never
/// touch disk. Paths are normalized (`foo/./bar` == `foo/bar`).
pub fn mount(path: impl AsRef<Path>, data: Vec<u8>) {
    let mut guard = FILES.lock().unwrap();
    let map = guard.get_or_insert_with(HashMap::new);
    map.insert(normalize(path.as_ref()), data);
}

/// Read a previously mounted file. Returns `None` if the path is not in the VFS.
pub fn read(path: &Path) -> Option<Vec<u8>> {
    let guard = FILES.lock().unwrap();
    guard.as_ref()?.get(&normalize(path)).cloned()
}
