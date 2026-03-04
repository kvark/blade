use std::unimplemented;

#[derive(Default)]
pub struct TriMesh {
    pub points: Vec<nalgebra::Point3<f32>>,
    pub triangles: Vec<[u32; 3]>,
}

impl TriMesh {
    fn populate_from_gltf(
        &mut self,
        g_node: gltf::Node,
        parent_transform: nalgebra::Matrix4<f32>,
        data_buffers: &[Vec<u8>],
    ) {
        let name = g_node.name().unwrap_or("");
        let transform = parent_transform * nalgebra::Matrix4::from(g_node.transform().matrix());

        for child in g_node.children() {
            self.populate_from_gltf(child, transform, data_buffers);
        }

        let g_mesh = match g_node.mesh() {
            Some(mesh) => mesh,
            None => return,
        };

        for (prim_index, g_primitive) in g_mesh.primitives().enumerate() {
            if g_primitive.mode() != gltf::mesh::Mode::Triangles {
                log::warn!(
                    "Skipping primitive '{}'[{}] for having mesh mode {:?}",
                    name,
                    prim_index,
                    g_primitive.mode()
                );
                continue;
            }

            let reader = g_primitive.reader(|buffer| Some(&data_buffers[buffer.index()]));

            // Read the vertices into memory
            profiling::scope!("Read data");
            let base_vertex = self.points.len() as u32;

            match reader.read_indices() {
                Some(read) => {
                    let mut read_u32 = read.into_u32();
                    let tri_count = read_u32.len() / 3;
                    for _ in 0..tri_count {
                        let mut tri = [0u32; 3];
                        for index in tri.iter_mut() {
                            *index = base_vertex + read_u32.next().unwrap();
                        }
                        self.triangles.push(tri);
                    }
                }
                None => {
                    log::warn!("Missing index buffer for '{name}'");
                    continue;
                }
            }

            for pos in reader.read_positions().unwrap() {
                let point = transform.transform_point(&pos.into());
                self.points.push(point);
            }
        }
    }
}

pub fn load(path: &str) -> TriMesh {
    use base64::engine::{general_purpose::URL_SAFE as ENCODING_ENGINE, Engine as _};

    let gltf::Gltf { document, mut blob } = gltf::Gltf::open(path).unwrap();
    // extract buffers
    let mut data_buffers = Vec::new();
    for buffer in document.buffers() {
        let mut data = match buffer.source() {
            gltf::buffer::Source::Uri(uri) => {
                if let Some(rest) = uri.strip_prefix("data:") {
                    let (_before, after) = rest.split_once(";base64,").unwrap();
                    ENCODING_ENGINE.decode(after).unwrap()
                } else {
                    unimplemented!("Unexpected reference to external file: {uri}");
                }
            }
            gltf::buffer::Source::Bin => blob.take().unwrap(),
        };
        assert!(data.len() >= buffer.length());
        while data.len() % 4 != 0 {
            data.push(0);
        }
        data_buffers.push(data);
    }

    let scene = document.scenes().next().expect("Document has no scenes?");
    let mut trimesh = TriMesh::default();
    for g_node in scene.nodes() {
        trimesh.populate_from_gltf(g_node, nalgebra::Matrix4::identity(), &data_buffers);
    }
    trimesh
}
