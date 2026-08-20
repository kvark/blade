use naga::{front::wgsl, valid::Validator};
use std::{collections::HashMap, fs, path::Path, path::PathBuf};

fn validate_shader(
    path: &Path,
    base_path: &Path,
    expansions: &HashMap<String, blade_render::shader::Expansion>,
) {
    println!("Validating {:?}", path);
    let shader_raw = fs::read(path).unwrap_or_default();
    let cooker = blade_asset::Cooker::new(base_path, Default::default());
    let mut text_out = blade_render::shader::parse_shader(&shader_raw, &cooker, expansions);

    // Substitute cooperative matrix template placeholders with defaults
    // so the shader parses as valid WGSL.
    text_out = text_out
        .replace("ENABLE_F16", "")
        .replace("COOP_MAT", "coop_mat8x8")
        .replace("INPUT_SCALAR", "f32")
        .replace("TILE_SIZE", "8u");

    let module = match wgsl::parse_str(&text_out) {
        Ok(module) => module,
        Err(e) => panic!("{}", e.emit_to_string(&text_out)),
    };
    //TODO: re-use the validator
    Validator::new(
        naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS,
        naga::valid::Capabilities::RAY_QUERY
            | naga::valid::Capabilities::ACCELERATION_STRUCTURE_BINDING_ARRAY
            | naga::valid::Capabilities::COOPERATIVE_MATRIX
            | naga::valid::Capabilities::STORAGE_BUFFER_BINDING_ARRAY
            | naga::valid::Capabilities::TEXTURE_AND_SAMPLER_BINDING_ARRAY
            | naga::valid::Capabilities::STORAGE_BUFFER_BINDING_ARRAY_NON_UNIFORM_INDEXING
            | naga::valid::Capabilities::TEXTURE_AND_SAMPLER_BINDING_ARRAY_NON_UNIFORM_INDEXING,
    )
    .validate(&module)
    .unwrap_or_else(|e| {
        blade_graphics::util::emit_annotated_error(&e, "", &text_out);
        blade_graphics::util::print_err(&e);
        panic!("Shader validation failed");
    });
}

/// Lists the standalone shaders in a directory.
///
/// The `*.inc.wgsl` includes are only valid in the context of their users.
fn list_shaders(dir: &Path) -> Vec<PathBuf> {
    let mut list = Vec::new();
    let read_dir = match dir.read_dir() {
        Ok(read_dir) => read_dir,
        Err(_) => return list,
    };
    for file in read_dir {
        let path = match file {
            Ok(entry) => entry.path(),
            Err(e) => {
                println!("Skipping file: {:?}", e);
                continue;
            }
        };
        let name = path.file_name().unwrap().to_str().unwrap();
        if name.ends_with(".inc.wgsl") || !name.ends_with(".wgsl") {
            continue;
        }
        list.push(path);
    }
    list
}

/// Runs through all pass shaders and ensures they are valid WGSL.
#[test]
fn parse_wgsl() {
    use blade_render::shader::Expansion;

    let mut expansions = HashMap::default();
    expansions.insert(
        "DebugMode".to_string(),
        Expansion::from_enum::<blade_render::DebugMode>(),
    );
    expansions.insert(
        "DebugDrawFlags".to_string(),
        Expansion::from_bitflags::<blade_render::DebugDrawFlags>(),
    );
    expansions.insert(
        "DebugTextureFlags".to_string(),
        Expansion::from_bitflags::<blade_render::DebugTextureFlags>(),
    );
    expansions.insert("DEBUG_MODE".to_string(), Expansion::Bool(true));

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut directories = vec![root.join("blade-render").join("code")];

    for sub_entry in root.join("examples").read_dir().unwrap() {
        match sub_entry {
            Ok(entry) => directories.push(entry.path()),
            Err(e) => println!("Skipping non-example: {:?}", e),
        }
    }

    for dir in directories {
        for path in list_shaders(&dir) {
            validate_shader(&path, &dir, &expansions);
        }
    }
}

/// Keep the portable renderer within the GLSL ES 3.00 feature set exposed by
/// WebGL2. WGSL validation alone does not catch backend-only resources such as
/// shader-storage buffers.
#[test]
fn raster_exports_to_webgl2() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let shader_dir = root.join("blade-render").join("code");
    let shader_raw = fs::read(shader_dir.join("raster.wgsl")).unwrap();
    let cooker = blade_asset::Cooker::new(&shader_dir, Default::default());
    let source = blade_render::shader::parse_shader(&shader_raw, &cooker, &HashMap::new());
    let mut module =
        wgsl::parse_str(&source).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
    let info = Validator::new(
        naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS,
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();

    let vertex_ep_index = module
        .entry_points
        .iter()
        .position(|ep| ep.name == "raster_vs")
        .unwrap();
    let vertex_ty = module.entry_points[vertex_ep_index].function.arguments[0].ty;
    let mut ty = module.types[vertex_ty].clone();
    let naga::TypeInner::Struct {
        ref mut members, ..
    } = ty.inner
    else {
        panic!("raster vertex input is not a struct");
    };
    for (location, member) in members.iter_mut().enumerate() {
        member.binding = Some(naga::Binding::Location {
            location: location as u32,
            interpolation: None,
            sampling: None,
            blend_src: None,
            per_primitive: false,
        });
    }
    module.types.replace(vertex_ty, ty);

    let mut stage_blocks: Vec<Vec<String>> = Vec::new();
    for entry_point in ["raster_vs", "raster_fs"] {
        let stage = module
            .entry_points
            .iter()
            .find(|ep| ep.name == entry_point)
            .unwrap()
            .stage;
        let options = naga::back::glsl::Options {
            version: naga::back::glsl::Version::Embedded {
                version: 300,
                is_webgl: true,
            },
            writer_flags: naga::back::glsl::WriterFlags::ADJUST_COORDINATE_SPACE,
            binding_map: Default::default(),
            zero_initialize_workgroup_memory: false,
        };
        let pipeline_options = naga::back::glsl::PipelineOptions {
            shader_stage: stage,
            entry_point: entry_point.to_string(),
            multiview: None,
        };
        let mut glsl = String::new();
        let mut writer = naga::back::glsl::Writer::new(
            &mut glsl,
            &module,
            &info,
            &options,
            &pipeline_options,
            Default::default(),
        )
        .unwrap();
        let mut reflection = writer.write().unwrap();
        // Mirror blade-graphics GLES ES 3.00: drop Naga's per-stage UBO suffix
        // so VS/FS share a block name. WebGL2 will not link otherwise.
        for name in reflection.uniforms.values_mut() {
            let stripped = name
                .strip_suffix("Vertex")
                .or_else(|| name.strip_suffix("Fragment"))
                .or_else(|| name.strip_suffix("Compute"));
            if let Some(stripped) = stripped {
                let stripped = stripped.to_string();
                glsl = glsl.replace(name.as_str(), &stripped);
                *name = stripped;
            }
        }
        assert!(glsl.starts_with("#version 300 es"));
        let mut blocks: Vec<String> = reflection.uniforms.values().cloned().collect();
        blocks.sort();
        stage_blocks.push(blocks);
    }
    assert_eq!(
        stage_blocks[0], stage_blocks[1],
        "WebGL2 requires matching uniform block names in vertex and fragment shaders"
    );
}
