use std::{
    fmt,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

struct Baker {
    allow_cooking: AtomicBool,
}
impl blade_asset::Baker for Baker {
    type Meta = u32;
    type Data<'a> = u32;
    type Output = usize;
    fn cook(
        &self,
        _source: &[u8],
        _extension: &str,
        meta: u32,
        cooker: Arc<blade_asset::Cooker<Self>>,
        _exe_context: &choir::ExecutionContext,
    ) {
        assert!(self.allow_cooking.load(Ordering::SeqCst));
        let _ = cooker.add_dependency("README.md".as_ref());
        cooker.finish(meta);
    }
    fn serve(&self, cooked: u32, _exe_context: &choir::ExecutionContext) -> usize {
        cooked as usize
    }
    fn delete(&self, _output: usize) {}
}

#[test]
fn test_asset() {
    let choir = choir::Choir::new();
    let _w1 = choir.add_worker("main");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let am = blade_asset::AssetManager::<Baker>::new(
        &root.join("cooked"),
        &choir,
        Baker {
            allow_cooking: AtomicBool::new(true),
        },
    );
    let meta = 5;
    let path = root.join("Cargo.toml");
    let (handle, task) = am.load(&path, meta);
    task.join();
    assert_eq!(am[handle], meta as usize);

    // now try to load it again and check that we aren't re-cooking
    am.baker.allow_cooking.store(false, Ordering::SeqCst);
    let (h, t) = am.load(&path, meta);
    assert_eq!(h, handle);
    t.join();
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
    flat_roundtrip(vec![1u32, 2, 3]);
}
