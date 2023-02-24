use naga::{front::wgsl, valid::Validator};
use std::{fs, path::PathBuf};

/// Runs through all pass shaders and ensures they are valid WGSL.
#[test]
fn parse_wgsl() {
    let read_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .read_dir()
        .unwrap();

    for sub_entry in read_dir {
        let example = match sub_entry {
            Ok(entry) => entry.path(),
            Err(e) => {
                println!("Skipping example: {:?}", e);
                continue;
            }
        };

        for file in example.read_dir().unwrap() {
            let path = match file {
                Ok(entry) => entry.path(),
                Err(e) => {
                    println!("Skipping file: {:?}", e);
                    continue;
                }
            };
            let shader = match path.extension() {
                Some(ostr) if &*ostr == "wgsl" => {
                    println!("Validating {:?}", path);
                    fs::read_to_string(path).unwrap_or_default()
                }
                _ => continue,
            };

            let module = match wgsl::parse_str(&shader) {
                Ok(module) => module,
                Err(e) => panic!("{}", e.emit_to_string(&shader)),
            };
            //TODO: re-use the validator
            Validator::new(
                naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS,
                naga::valid::Capabilities::RAY_QUERY,
            )
            .validate(&module)
            .unwrap_or_else(|e| {
                blade::util::emit_annotated_error(&e, "", &shader);
                blade::util::print_err(&e);
                panic!("Shader validation failed");
            });
        }
    }
}
