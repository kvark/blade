use std::sync::Arc;

mod gltf_loader;

pub struct Baker;
impl blade_asset::Baker for Baker {
    type Meta = ();
    type Format = ();
    type Output = usize;
    fn cook(
        &self,
        _source: &[u8],
        _extension: &str,
        _meta: (),
        _result: Arc<blade_asset::SynCell<Vec<u8>>>,
        _exe_context: choir::ExecutionContext,
    ) {
        unimplemented!()
    }
    fn serve(&self, _cooked: &[u8]) -> Self::Output {
        unimplemented!()
    }
}
