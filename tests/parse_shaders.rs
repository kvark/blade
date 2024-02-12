use naga::{front::wgsl, valid::Validator};
use std::{collections::HashMap, fs, path::PathBuf};

/// Runs through all pass shaders and ensures they are valid WGSL.
#[test]
fn parse_wgsl() {
    let mut expansions = HashMap::default();
    expansions.insert(
        "DebugMode".to_string(),
        blade_render::shader::Expansion::from_enum::<blade_render::DebugMode>(),
    );

    let read_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .read_dir()
        .unwrap();

    for sub_entry in read_dir {
        let example = match sub_entry {
            Ok(entry) => entry.path(),
            Err(e) => {
                println!("Skipping non-example: {:?}", e);
                continue;
            }
        };
        let dir = match example.read_dir() {
            Ok(dir) => dir,
            Err(_) => continue,
        };

        for file in dir {
            let path = match file {
                Ok(entry) => entry.path(),
                Err(e) => {
                    println!("Skipping file: {:?}", e);
                    continue;
                }
            };
            let shader_raw = match path.extension() {
                Some(ostr) if &*ostr == "wgsl" => {
                    println!("Validating {:?}", path);
                    fs::read(&path).unwrap_or_default()
                }
                _ => continue,
            };

            let cooker = blade_asset::Cooker::new(&example, Default::default());
            let text_out = blade_render::shader::parse_shader(&shader_raw, &cooker, &expansions);

            let module = match wgsl::parse_str(&text_out) {
                Ok(module) => module,
                Err(e) => panic!("{}", e.emit_to_string(&text_out)),
            };
            //TODO: re-use the validator
            Validator::new(
                naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS,
                naga::valid::Capabilities::RAY_QUERY,
            )
            .validate(&module)
            .unwrap_or_else(|e| {
                blade_graphics::util::emit_annotated_error(&e, "", &text_out);
                blade_graphics::util::print_err(&e);
                panic!("Shader validation failed");
            });
        }
    }
}
