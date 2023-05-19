use std::sync::Arc;

//#[cfg(feature = "asset")]
mod gltf_loader;

pub struct Geometry {
    pub vertex_buf: blade::Buffer,
    pub vertex_count: u32,
    pub index_buf: blade::Buffer,
    pub index_type: Option<blade::IndexType>,
    pub triangle_count: u32,
    pub transform: blade::Transform,
    //pub material: blade_asset::Handle<Material>,
}

pub struct Model {
    pub name: String,
    pub geometries: Vec<Geometry>,
    pub acceleration_structure: blade::AccelerationStructure,
}

#[derive(blade_macros::Flat)]
pub struct CookedModel<'a> {
    name: &'a [u8],
}

pub struct Baker;

impl blade_asset::Baker for Baker {
    type Meta = ();
    type Data<'a> = CookedModel<'a>;
    type Output = Model;
    fn cook(
        &self,
        _source: &[u8],
        extension: &str,
        _meta: (),
        result: Arc<blade_asset::Cooked<CookedModel<'_>>>,
        _exe_context: choir::ExecutionContext,
    ) {
        match extension {
            #[cfg(feature = "asset")]
            "gltf" => {
                result.put(CookedModel { name: &[] });
            }
            other => panic!("Unknown texture extension: {}", other),
        }
    }
    fn serve(&self, _model: CookedModel<'_>) -> Self::Output {
        unimplemented!()
    }
}
