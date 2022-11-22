use std::{collections::HashMap, fmt::Write as _};

fn map_view_dimension(dimension: super::TextureViewDimension) -> &'static str {
    use super::TextureViewDimension as Tvd;
    match dimension {
        Tvd::D1 => "1d",
        Tvd::D2 => "2d",
        Tvd::D2Array => "2d_array",
        Tvd::Cube => "cube",
        Tvd::CubeArray => "cube_array",
        Tvd::D3 => "3d",
    }
}

fn map_storage_access(access: super::StorageAccess) -> &'static str {
    if access == super::StorageAccess::LOAD {
        "read"
    } else if access == super::StorageAccess::STORE {
        "write"
    } else {
        "read_write"
    }
}

fn map_storage_format(format: super::TextureFormat) -> &'static str {
    use super::TextureFormat as Tf;
    match format {
        Tf::Rgba8Unorm => "rgba8unorm",
    }
}

impl super::Context {
    pub fn create_shader(&self, desc: super::ShaderDesc) -> super::Shader {
        const UNIFORM_NAME: &str = "_uniforms";
        struct Substitute {
            group_index: usize,
            is_uniform: bool,
        }

        let mut substitutions = HashMap::<&str, Substitute>::default();
        let mut header = String::new();
        
        for (group_index, layout_maybe) in desc.data_layouts.iter().enumerate() {
            let layout = match layout_maybe {
                Some(layout) => layout,
                None => continue,
            };

            //Note: the binding scheme is implicit:
            // Uniform buffer is at 0, and the rest are resources.
            let mut binding_index = 1;
            let mut has_uniforms = false;
            for &(ref name, binding) in layout.bindings.iter() {
                let old_binding_index = binding_index;
                match binding {
                    super::ShaderBinding::Texture { dimension } => {
                        let dim_str = map_view_dimension(dimension);
                        writeln!(header, "@group({}) @binding({}) var {}: texture_{}<f32>;",
                            group_index, binding_index, name, dim_str).unwrap();
                        binding_index += 1;
                    }
                    super::ShaderBinding::TextureStorage { dimension, format, access } => {
                        let dim_str = map_view_dimension(dimension);
                        let format_str = map_storage_format(format);
                        let access_str = map_storage_access(access);
                        writeln!(header, "@group({}) @binding({}) var {}: texture_storage_{}<{},{}>;",
                            group_index, binding_index, name, dim_str, format_str, access_str).unwrap();
                        binding_index += 1;
                    }
                    super::ShaderBinding::Sampler { comparison } => {
                        let suffix = if comparison { "_comparison" } else { "" };
                        writeln!(header, "@group({}) @binding({}) var {}: sampler{};",
                            group_index, binding_index, name, suffix).unwrap();
                        binding_index += 1;
                    }
                    super::ShaderBinding::Buffer { type_name, access } => {
                        let access_str = map_storage_access(access);
                        writeln!(header, "@group({}) @binding({}) var<storage, {}> {}: {};",
                            group_index, binding_index, access_str, name, type_name).unwrap();
                        binding_index += 1;
                    }
                    super::ShaderBinding::Plain { .. } => {
                    }
                };

                let is_uniform = binding_index == old_binding_index;
                has_uniforms |= is_uniform;
                if let Some(old) = substitutions.insert(name.as_str(), Substitute { group_index, is_uniform }) {
                    panic!("Duplicate binding '{}' in groups {} and {}", name, old.group_index, group_index);
                }
            }

            if has_uniforms {
                writeln!(header, "struct _Uniforms{} {{", group_index).unwrap();
                for &(ref name, binding) in layout.bindings.iter() {
                    match binding {
                        super::ShaderBinding::Texture { .. } |
                        super::ShaderBinding::TextureStorage { .. } |
                        super::ShaderBinding::Sampler { .. } |
                        super::ShaderBinding::Buffer { .. } => continue,
                        super::ShaderBinding::Plain { ty, container } => {
                            let scalar_name = match ty {
                                super::PlainType::U32 => "u32",
                                super::PlainType::I32 => "i32",
                                super::PlainType::F32 => "f32",
                            };
                            let ty_name = match container {
                                super::PlainContainer::Scalar => scalar_name.to_string(),
                                super::PlainContainer::Vector(size) => format!("vec{}<{}>", size as u32, scalar_name),
                                super::PlainContainer::Matrix(rows, cols) => format!("mat{}x{}<{}>", rows as u32, cols as u32, scalar_name),
                            };
                            writeln!(header, "\t{}: {},", name, ty_name).unwrap();
                        }
                    }
                }
                writeln!(header, "}}").unwrap();
                writeln!(header, "@group({}) @binding(0) var<uniform> {}{}: _Uniforms{};", group_index, UNIFORM_NAME, group_index, group_index).unwrap();
            }
        }
        log::debug!("Generated header:\n{}", header);

        let mut text = String::new();
        for line in desc.source.lines() {
            if line.starts_with("#") {
                if &line[1..] == "header" {
                    text.push_str(&header);
                }
                //TODO: handle includes
            } else {
                let mut remain = line;
                while let Some(pos) = remain.find('$') {
                    text.push_str(&remain[..pos]);
                    remain = &remain[pos+1..];
                    let (name, tail) = match remain.find(|c: char| !c.is_alphanumeric()) {
                        Some(end) => remain.split_at(end),
                        None => (remain, ""),
                    };
                    match substitutions.get(name) {
                        Some(sub) => {
                            if sub.is_uniform {
                                write!(text, "{}{}.", UNIFORM_NAME, sub.group_index).unwrap();
                            }
                            text.push_str(name);
                        }
                        None => panic!("Unable to substitute binding '{}'", name),
                    }
                    remain = tail;
                }
                text.push_str(remain);
            }
            text.push_str("\n");
        }

        let module = match naga::front::wgsl::parse_str(&text) {
            Ok(module) => module,
            Err(ref e) => {
                e.emit_to_stderr_with_path(&text, "");
                panic!("Shader compilation failed");
            }
        };

        let caps = naga::valid::Capabilities::empty();
        let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), caps)
            .validate(&module)
            .unwrap();

        super::Shader {
            module,
            info,
            bind_groups: desc.data_layouts.iter().map(|opt| opt.cloned()).collect(),
        }
    }
}
