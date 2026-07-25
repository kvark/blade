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
