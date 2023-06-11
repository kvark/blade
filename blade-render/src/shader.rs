use std::{fmt, fs, io::Read as _, path::Path, str, sync::Arc};

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

impl blade_asset::Baker for Baker {
    type Meta = Meta;
    type Data<'a> = CookedShader<'a>;
    type Output = Shader;
    fn cook(
        &self,
        source: &[u8],
        base_path: &Path,
        extension: &str,
        _meta: Meta,
        result: Arc<blade_asset::Cooked<CookedShader>>,
        _exe_context: choir::ExecutionContext,
    ) {
        assert_eq!(extension, "wgsl");
        let text_in = str::from_utf8(source).unwrap();

        let mut text_out = String::new();
        for line in text_in.lines() {
            if line.starts_with("#include") {
                let inc_path = match line.split('"').nth(1) {
                    Some(include) => base_path.join(include),
                    None => panic!("Unable to extract the include path from: {line}"),
                };
                match fs::File::open(&inc_path) {
                    Ok(mut include) => include.read_to_string(&mut text_out).unwrap(),
                    Err(e) => panic!("Unable to include {}: {:?}", inc_path.display(), e),
                };
            } else {
                text_out += line;
            };
            text_out += "\n";
        }

        result.put(CookedShader {
            data: text_out.as_bytes(),
        });
    }
    fn serve(&self, cooked: CookedShader, _exe_context: choir::ExecutionContext) -> Shader {
        let source = str::from_utf8(cooked.data).unwrap();
        Shader {
            raw: self
                .gpu_context
                .create_shader(blade_graphics::ShaderDesc { source }),
        }
    }
    fn delete(&self, _output: Shader) {}
}
