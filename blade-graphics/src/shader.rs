use std::{
    collections::{HashMap, HashSet},
    fmt::Write as _,
};

mod plain {
    use crate::{PlainContainer as Pc, PlainType as Pt, VectorSize as Vs};

    trait AsScalar {
        const TYPE: Pt;
    }
    impl AsScalar for f32 {
        const TYPE: Pt = Pt::F32;
    }
    impl AsScalar for u32 {
        const TYPE: Pt = Pt::U32;
    }
    impl AsScalar for i32 {
        const TYPE: Pt = Pt::I32;
    }

    impl<S: AsScalar> crate::AsPlain for S {
        const TYPE: Pt = S::TYPE;
        const CONTAINER: Pc = Pc::Scalar;
    }
    impl<S: AsScalar> crate::AsPlain for [S; 2] {
        const TYPE: Pt = S::TYPE;
        const CONTAINER: Pc = Pc::Vector(Vs::Bi);
    }
    impl<S: AsScalar> crate::AsPlain for [S; 3] {
        const TYPE: Pt = S::TYPE;
        const CONTAINER: Pc = Pc::Vector(Vs::Tri);
    }
    impl<S: AsScalar> crate::AsPlain for [S; 4] {
        const TYPE: Pt = S::TYPE;
        const CONTAINER: Pc = Pc::Vector(Vs::Quad);
    }

    impl crate::AsPlain for [[f32; 2]; 2] {
        const TYPE: Pt = Pt::F32;
        const CONTAINER: Pc = Pc::Matrix(Vs::Bi, Vs::Bi);
    }
    impl crate::AsPlain for [[f32; 4]; 4] {
        const TYPE: Pt = Pt::F32;
        const CONTAINER: Pc = Pc::Matrix(Vs::Quad, Vs::Quad);
    }

    #[cfg(feature = "mint")]
    impl<S: AsScalar> crate::AsPlain for mint::Vector2<S> {
        const TYPE: Pt = S::TYPE;
        const CONTAINER: Pc = Pc::Vector(Vs::Bi);
    }
    #[cfg(feature = "mint")]
    impl<S: AsScalar> crate::AsPlain for mint::Vector3<S> {
        const TYPE: Pt = S::TYPE;
        const CONTAINER: Pc = Pc::Vector(Vs::Tri);
    }
    #[cfg(feature = "mint")]
    impl<S: AsScalar> crate::AsPlain for mint::Vector4<S> {
        const TYPE: Pt = S::TYPE;
        const CONTAINER: Pc = Pc::Vector(Vs::Quad);
    }
    #[cfg(feature = "mint")]
    impl<S: AsScalar> crate::AsPlain for mint::Quaternion<S> {
        const TYPE: Pt = S::TYPE;
        const CONTAINER: Pc = Pc::Vector(Vs::Quad);
    }
}

fn map_view_dimension(dimension: super::TextureViewDimension) -> &'static str {
    use super::TextureViewDimension as Tvd;
    match dimension {
        Tvd::D1 => "1d",
        Tvd::D1Array => "1d_array",
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
        Tf::Depth32Float | Tf::Bgra8UnormSrgb => panic!("Unsupported storage format"),
    }
}

fn map_plain_type(ty: super::PlainType) -> &'static str {
    match ty {
        super::PlainType::U32 => "u32",
        super::PlainType::I32 => "i32",
        super::PlainType::F32 => "f32",
    }
}

pub(crate) fn merge_layouts<'a>(
    multi_layouts: &[(super::ShaderFunction<'a>, super::ShaderVisibility)],
) -> Vec<(&'a super::ShaderDataLayout, super::ShaderVisibility)> {
    let group_count = multi_layouts
        .iter()
        .map(|(sf, _)| sf.shader.bind_groups.len())
        .max()
        .unwrap_or_default();
    (0..group_count)
        .map(|group_index| {
            let mut layout_maybe = None;
            let mut visibility = super::ShaderVisibility::empty();
            for &(sf, shader_visibility) in multi_layouts {
                let ep_index = sf.entry_point_index();
                let ep_info = sf.shader.info.get_entry_point(ep_index);
                if let Some(ref bind_group) = sf.shader.bind_groups.get(group_index) {
                    // Check if any of the bindings are actually used by the entry point
                    if bind_group
                        .used_globals
                        .iter()
                        .all(|&var| ep_info[var].is_empty())
                    {
                        continue;
                    }
                    visibility |= shader_visibility;
                    if let Some(layout) = layout_maybe {
                        assert_eq!(&bind_group.layout, layout);
                    } else {
                        layout_maybe = Some(&bind_group.layout);
                    }
                }
            }
            match layout_maybe {
                Some(layout) => (layout, visibility),
                None => (
                    super::ShaderDataLayout::EMPTY,
                    super::ShaderVisibility::empty(),
                ),
            }
        })
        .collect()
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

        for (group_index, layout) in desc.data_layouts.iter().enumerate() {
            //Note: the binding scheme is implicit:
            // Uniform buffer is at 0, and the rest are resources.
            let mut binding_index = 1;
            let mut has_uniforms = false;
            for &(ref name, binding) in layout.bindings.iter() {
                let old_binding_index = binding_index;
                match binding {
                    super::ShaderBinding::Texture { dimension, ty } => {
                        let dim_str = map_view_dimension(dimension);
                        write!(
                            header,
                            "@group({}) @binding({}) var {}: ",
                            group_index, binding_index, name,
                        )
                        .unwrap();
                        match ty {
                            super::TextureBindingType::Plain(pty) => {
                                let scalar_name = map_plain_type(pty);
                                write!(header, "texture_{}<{}>", dim_str, scalar_name)
                                .unwrap();
                            }
                            super::TextureBindingType::Depth => {
                                write!(header, "texture_depth_{}", dim_str)
                                .unwrap();
                            }
                        }
                        write!(header, ";").unwrap();
                        binding_index += 1;
                    }
                    super::ShaderBinding::TextureStorage {
                        dimension,
                        format,
                        access,
                    } => {
                        let dim_str = map_view_dimension(dimension);
                        let format_str = map_storage_format(format);
                        let access_str = map_storage_access(access);
                        writeln!(
                            header,
                            "@group({}) @binding({}) var {}: texture_storage_{}<{},{}>;",
                            group_index, binding_index, name, dim_str, format_str, access_str
                        )
                        .unwrap();
                        binding_index += 1;
                    }
                    super::ShaderBinding::Sampler { comparison } => {
                        let suffix = if comparison { "_comparison" } else { "" };
                        writeln!(
                            header,
                            "@group({}) @binding({}) var {}: sampler{};",
                            group_index, binding_index, name, suffix
                        )
                        .unwrap();
                        binding_index += 1;
                    }
                    super::ShaderBinding::Buffer { type_name, access } => {
                        let access_str = map_storage_access(access);
                        writeln!(
                            header,
                            "@group({}) @binding({}) var<storage, {}> {}: {};",
                            group_index, binding_index, access_str, name, type_name
                        )
                        .unwrap();
                        binding_index += 1;
                    }
                    super::ShaderBinding::Plain { .. } => {}
                };

                let is_uniform = binding_index == old_binding_index;
                has_uniforms |= is_uniform;
                if let Some(old) = substitutions.insert(
                    name,
                    Substitute {
                        group_index,
                        is_uniform,
                    },
                ) {
                    panic!(
                        "Duplicate binding '{}' in groups {} and {}",
                        name, old.group_index, group_index
                    );
                }
            }

            if has_uniforms {
                writeln!(header, "struct _Uniforms{} {{", group_index).unwrap();
                for &(ref name, binding) in layout.bindings.iter() {
                    match binding {
                        super::ShaderBinding::Texture { .. }
                        | super::ShaderBinding::TextureStorage { .. }
                        | super::ShaderBinding::Sampler { .. }
                        | super::ShaderBinding::Buffer { .. } => continue,
                        super::ShaderBinding::Plain { ty, container } => {
                            let scalar_name = map_plain_type(ty);
                            let ty_name = match container {
                                super::PlainContainer::Scalar => scalar_name.to_string(),
                                super::PlainContainer::Vector(size) => {
                                    format!("vec{}<{}>", size as u32, scalar_name)
                                }
                                super::PlainContainer::Matrix(rows, cols) => {
                                    format!("mat{}x{}<{}>", rows as u32, cols as u32, scalar_name)
                                }
                            };
                            writeln!(header, "\t{}: {},", name, ty_name).unwrap();
                        }
                    }
                }
                writeln!(header, "}}").unwrap();
                writeln!(
                    header,
                    "@group({}) @binding(0) var<uniform> {}{}: _Uniforms{};",
                    group_index, UNIFORM_NAME, group_index, group_index
                )
                .unwrap();
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
                    remain = &remain[pos + 1..];
                    let (name, tail) = match remain.find(|c: char| !c.is_alphanumeric() && c != '_')
                    {
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
            .unwrap_or_else(|e| {
                crate::util::emit_annotated_error(&e, "", &text);
                crate::util::print_err(&e);
                panic!("Shader validation failed");
            });

        let mut bind_groups = Vec::with_capacity(desc.data_layouts.len());
        for (group_index, &data_layout) in desc.data_layouts.iter().enumerate() {
            let mut used_globals = HashSet::default();
            // cross-reference the bindings with globals to see which
            // ones are used by the shader.
            for (handle, var) in module.global_variables.iter() {
                // filter out the globals from other bind groups
                match var.binding {
                    Some(naga::ResourceBinding { group, binding: _ })
                        if group as usize == group_index => {}
                    _ => continue,
                };
                for &(name, binding) in data_layout.bindings.iter() {
                    let match_name = match binding {
                        // there can only be one uniform buffer per group in Blade
                        crate::ShaderBinding::Plain { .. } => {
                            var.space == naga::AddressSpace::Uniform
                        }
                        _ => var.name.as_ref().map(|s| s.as_str()) == Some(name),
                    };
                    if match_name {
                        used_globals.insert(handle);
                    }
                }
            }
            bind_groups.push(crate::ShaderBindGroup {
                layout: data_layout.clone(),
                used_globals,
            });
        }

        super::Shader {
            module,
            info,
            bind_groups: bind_groups.into_boxed_slice(),
        }
    }
}
