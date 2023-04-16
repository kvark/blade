use std::path::{Path, PathBuf};

struct Baker;
impl blade_asset::Baker for Baker {
    type Meal = usize;
    fn cook(&self, _input: &[u8]) -> Box<[u8]> {
        (0..1).map(|_| 2u8).collect()
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
    let (handle, task) = am.load(Path::new("Cargo.toml"));
    task.clone().join();
    assert_eq!(am[handle], 1);
}
