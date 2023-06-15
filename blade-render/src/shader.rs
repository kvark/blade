use std::{fmt, fs, path::Path, str, sync::Arc};

const FAILURE_DUMP_NAME: &str = "_failure.wgsl";

#[derive(blade_macros::Flat)]
pub struct CookedShader<'a> {
    data: &'a [u8],
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Meta;
impl fmt::Display for Meta {
    fn fmt(&self, _f: &mut fmt::Formatter) -> fmt::Result {
        Ok(())
    }
}

pub struct Shader {
    pub raw: blade_graphics::Shader,
}

pub struct Baker {
    gpu_context: Arc<blade_graphics::Context>,
}

impl Baker {
    pub fn new(gpu_context: &Arc<blade_graphics::Context>) -> Self {
        Self {
            gpu_context: Arc::clone(gpu_context),
        }
    }
}

fn parse_impl(
    text_raw: &[u8],
    base_path: &Path,
    text_out: &mut String,
    cooker: &blade_asset::Cooker<CookedShader>,
) {
    let text_in = str::from_utf8(text_raw).unwrap();
    for line in text_in.lines() {
        if line.starts_with("#include") {
            let include_path = match line.split('"').nth(1) {
                Some(include) => base_path.join(include),
                None => panic!("Unable to extract the include path from: {line}"),
            };
            let include = cooker.add_dependency(&include_path);
            *text_out += "//";
            *text_out += line;
            *text_out += "\n";
            parse_impl(&include, include_path.parent().unwrap(), text_out, cooker);
        } else {
            *text_out += line;
        }
        *text_out += "\n";
    }
}

pub fn parse_shader(text_raw: &[u8], cooker: &blade_asset::Cooker<CookedShader>) -> String {
    let mut text_out = String::new();
    parse_impl(text_raw, ".".as_ref(), &mut text_out, cooker);
    text_out
}

impl blade_asset::Baker for Baker {
    type Meta = Meta;
    type Data<'a> = CookedShader<'a>;
    type Output = Shader;
    fn cook(
        &self,
        source: &[u8],
        extension: &str,
        _meta: Meta,
        cooker: Arc<blade_asset::Cooker<CookedShader>>,
        _exe_context: choir::ExecutionContext,
    ) {
        assert_eq!(extension, "wgsl");
        let text_out = parse_shader(source, &cooker);
        cooker.finish(CookedShader {
            data: text_out.as_bytes(),
        });
    }
    fn serve(&self, cooked: CookedShader, _exe_context: choir::ExecutionContext) -> Shader {
        let source = str::from_utf8(cooked.data).unwrap();
        match self
            .gpu_context
            .try_create_shader(blade_graphics::ShaderDesc { source })
        {
            Ok(raw) => Shader { raw },
            Err(e) => {
                let _ = fs::write(FAILURE_DUMP_NAME, source);
                panic!("Shader compilation failed: {e:?}, source dumped as '{FAILURE_DUMP_NAME}'.")
            }
        }
    }
    fn delete(&self, _output: Shader) {}
}
