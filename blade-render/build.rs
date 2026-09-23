//! Turn `shaders/*.rs` into the WGSL files under `code/`.
//!
//! The same sources are compiled as Rust by the crate. This step is what the
//! GPU actually runs.

use std::path::{Path, PathBuf};

fn main() {
    let caps = blade_caps();
    let config = config_source();
    let stage = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("wgsl");
    std::fs::create_dir_all(&stage).unwrap();
    let mut failed = false;
    let mut ready: Vec<(&Shader, PathBuf)> = Vec::new();
    for shader in SHADERS {
        let preludes: Vec<String> = shader
            .preludes
            .iter()
            .map(|name| {
                if *name == "config" {
                    config.clone()
                } else {
                    std::fs::read_to_string(Path::new("shaders").join(format!("{name}.rs")))
                        .unwrap_or_else(|err| panic!("read shaders/{name}.rs: {err}"))
                }
            })
            .collect();
        let source_path = PathBuf::from("shaders").join(format!("{}.rs", shader.module));
        println!("cargo::rerun-if-changed={}", source_path.display());
        let source = std::fs::read_to_string(&source_path).expect("shader source");
        match transpile(&source_path, &preludes, &source, caps) {
            Ok(wgsl) => {
                let staged = stage.join(shader.file);
                std::fs::write(&staged, wgsl).expect("stage wgsl");
                ready.push((shader, staged));
            }
            Err(err) => {
                failed = true;
                for line in err.lines() {
                    println!("cargo::error={line}");
                }
            }
        }
    }
    if failed {
        std::process::exit(1);
    }
    // Publish only once every shader transpiles, so a failure cannot replace
    // a hand-written WGSL file with a partial result.
    for (shader, staged) in ready {
        let out = Path::new("code").join(shader.file);
        let wgsl = std::fs::read_to_string(&staged).unwrap();
        if std::fs::read_to_string(&out).ok().as_deref() != Some(wgsl.as_str()) {
            std::fs::write(&out, wgsl).expect("write wgsl");
        }
    }
}

struct Shader {
    module: &'static str,
    file: &'static str,
    preludes: &'static [&'static str],
}

const SHADERS: &[Shader] = &[
    Shader {
        module: "noop",
        file: "noop.wgsl",
        preludes: &[],
    },
    Shader {
        module: "env_prepare",
        file: "env-prepare.wgsl",
        preludes: &[],
    },
    Shader {
        module: "debug_blit",
        file: "debug-blit.wgsl",
        preludes: &[],
    },
    Shader {
        module: "debug_draw",
        file: "debug-draw.wgsl",
        preludes: &["config", "quaternion", "debug", "camera"], // quaternion before camera: camera calls qrot
    },
    Shader {
        module: "post_proc",
        file: "post-proc.wgsl",
        preludes: &["config", "debug", "color", "debug_param"],
    },
    Shader {
        module: "skin",
        file: "skin.wgsl",
        preludes: &["config", "vertex", "skin_inc"],
    },
    Shader {
        module: "raster",
        file: "raster.wgsl",
        preludes: &["config", "vertex", "brdf", "color", "skin_inc"],
    },
    Shader {
        module: "a_trous",
        file: "a-trous.wgsl",
        preludes: &["config", "quaternion", "camera", "gbuf", "surface"],
    },
    Shader {
        module: "fill_gbuf",
        file: "fill-gbuf.wgsl",
        preludes: &[
            "config",
            "vertex",
            "quaternion",
            "camera",
            "debug_param",
            "debug",
            "brdf",
            "gbuf",
            "hit",
        ],
    },
    Shader {
        module: "path_trace",
        file: "path-trace.wgsl",
        preludes: &[
            "config",
            "vertex",
            "quaternion",
            "random",
            "camera",
            "debug_param",
            "brdf",
            "sampling",
            "env_importance",
            "hit",
            "env_light",
        ],
    },
    Shader {
        module: "ray_trace",
        file: "ray-trace.wgsl",
        preludes: &[
            "config",
            "vertex",
            "quaternion",
            "random",
            "camera",
            "debug_param",
            "debug",
            "brdf",
            "sampling",
            "env_importance",
            "surface",
            "gbuf",
            "hit",
            "env_light",
        ],
    },
];

fn config_source() -> String {
    let debug = std::env::var("PROFILE").ok().as_deref() != Some("release");
    std::fs::read_to_string("shaders/config.rs")
        .expect("config")
        .replace(
            "cfg!(debug_assertions)",
            if debug { "true" } else { "false" },
        )
}

fn blade_caps() -> synaga::naga::valid::Capabilities {
    use synaga::naga::valid::Capabilities as C;
    C::RAY_QUERY
        | C::STORAGE_BUFFER_BINDING_ARRAY
        | C::STORAGE_BUFFER_BINDING_ARRAY_NON_UNIFORM_INDEXING
        | C::TEXTURE_AND_SAMPLER_BINDING_ARRAY
        | C::TEXTURE_AND_SAMPLER_BINDING_ARRAY_NON_UNIFORM_INDEXING
}

fn transpile(
    path: &Path,
    preludes: &[String],
    source: &str,
    caps: synaga::naga::valid::Capabilities,
) -> Result<String, String> {
    let texts: Vec<&str> = preludes
        .iter()
        .map(String::as_str)
        .chain(std::iter::once(source))
        .collect();
    let module = synaga::parse_all(texts).map_err(|err| {
        let file = preludes
            .get(err.index)
            .map(|_| "prelude")
            .unwrap_or("shader");
        format!("{}: {file}: {}", path.display(), err.error)
    })?;
    let flags = synaga::naga::valid::ValidationFlags::all()
        ^ synaga::naga::valid::ValidationFlags::BINDINGS;
    let info = synaga::validate_with(&module, flags, caps)
        .map_err(|err| format!("{}: {err}", path.display()))?;
    let wgsl =
        synaga::to_wgsl(&module, &info).map_err(|err| format!("{}: {err}", path.display()))?;
    for entry in synaga::build::entry_point_names(&module) {
        if entry.renamed() {
            println!(
                "cargo::warning={}: entry point `{}` is `{}` in the generated WGSL",
                path.display(),
                entry.name,
                entry.emitted_name
            );
        }
    }
    Ok(wgsl)
}
