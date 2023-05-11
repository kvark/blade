use std::{
    fmt,
    path::{Path, PathBuf},
    sync::Arc,
};
use syncell::SynCell;

struct Baker;
impl blade_asset::Baker for Baker {
    type Meta = ();
    type Format = ();
    type Output = usize;
    fn cook(
        &self,
        _source: &[u8],
        _extension: &str,
        _meta: (),
        result: Arc<SynCell<Vec<u8>>>,
        _exe_context: choir::ExecutionContext,
    ) {
        *result.borrow_mut() = vec![1, 2, 3];
    }
    fn serve(&self, cooked: &[u8]) -> usize {
        cooked.len()
    }
}

#[test]
fn test_asset() {
    let choir = choir::Choir::new();
    let _w1 = choir.add_worker("main");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let am = blade_asset::AssetManager::<Baker>::new(&root, &root.join("cooked"), &choir, Baker);
    let (handle, task) = am.load(Path::new("Cargo.toml"), ());
    task.clone().join();
    assert_eq!(am[handle], 3);
}

fn flat_roundtrip<F: blade_asset::Flat + PartialEq + fmt::Debug>(data: F) {
    let mut vec = vec![0u8; data.size()];
    unsafe { data.write(vec.as_mut_ptr()) };
    let other = unsafe { F::read(vec.as_ptr()) };
    assert_eq!(data, other);
}

#[test]
fn test_flatten() {
    flat_roundtrip([0u32, 1u32, 2u32]);
    flat_roundtrip(&[2u32, 4u32, 6u32][..]);
}
