use std::{collections::HashMap, fmt::Write as _};

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
                        let dim_name = match dimension {
                            super::TextureViewDimension::D1 => "1d",
                            super::TextureViewDimension::D2 => "2d",
                            super::TextureViewDimension::D2Array => "2d_array",
                            super::TextureViewDimension::Cube => "cube",
                            super::TextureViewDimension::CubeArray => "cube_array",
                            super::TextureViewDimension::D3 => "3d",
                        };
                        writeln!(header, "@group({}) @binding({}) var {}: texture_{}<f32>;",
                            group_index, binding_index, name, dim_name).unwrap();
                        binding_index += 1;
                    }
                    super::ShaderBinding::Sampler { comparison } => {
                        let suffix = if comparison { "_comparison" } else { "" };
                        writeln!(header, "@group({}) @binding({}) var {}: sampler{};",
                            group_index, binding_index, name, suffix).unwrap();
                        binding_index += 1;
                    }
                    super::ShaderBinding::Buffer { type_name, access } => {
                        let access_str = if access == super::StorageAccess::LOAD {
                            "read"
                        } else if access == super::StorageAccess::STORE {
                            "write"
                        } else {
                            "read_write"
                        };
                        writeln!(header, "@group({}) @binding({}) var<storage, {}> {}: {};",
                            group_index, binding_index, access_str, name, type_name).unwrap();
                        binding_index += 1;
                    }
                    super::ShaderBinding::Plain { .. } => {
                    }
                };

                let is_uniform = binding_index != old_binding_index;
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
                        super::ShaderBinding::Sampler { .. } |
                        super::ShaderBinding::Buffer { .. } => continue,
                        super::ShaderBinding::Plain { ty, container } => {
                            let scalar_name = match ty {
                                super::PlainType::F32 => "f32",
                            };
                            let ty_name = match container {
                                super::PlainContainer::Scalar => scalar_name.to_string(),
                                super::PlainContainer::Vector(size) => format!("vec{}<{}>", size as u32, scalar_name),
                            };
                            writeln!(header, "\t{}: {},", name, ty_name).unwrap();
                        }
                    }
                }
                writeln!(header, "}}").unwrap();
                writeln!(header, "@group({}) @binding(0) var<uniform> {}{}: _Uniforms{};", group_index, UNIFORM_NAME, group_index, group_index).unwrap();
            }
        }

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

        let module = naga::front::wgsl::parse_str(&text).unwrap();

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
