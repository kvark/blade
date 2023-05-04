mod gltf_loader;

use std::path::Path;

pub struct Baker;
impl blade_asset::Baker for Baker {
    type Meta = ();
    type Output = usize;
    fn cook(
        &self,
        _src_path: &Path,
        _meta: (),
        _dst_path: &Path,
        _exe_context: choir::ExecutionContext,
    ) {
        unimplemented!()
    }
    fn serve(&self, _cooked: &[u8]) -> Self::Output {
        unimplemented!()
    }
}
