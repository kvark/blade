use std::{fmt, str, sync::Arc};

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
        extension: &str,
        _meta: Meta,
        cooker: Arc<blade_asset::Cooker<CookedShader>>,
        _exe_context: choir::ExecutionContext,
    ) {
        assert_eq!(extension, "wgsl");
        let text_in = str::from_utf8(source).unwrap();

        let mut text_out = String::new();
        for line in text_in.lines() {
            if line.starts_with("#include") {
                let include = match line.split('"').nth(1) {
                    Some(include) => cooker.add_dependency(include.as_ref()),
                    None => panic!("Unable to extract the include path from: {line}"),
                };
                text_out += str::from_utf8(&include).unwrap();
            } else {
                text_out += line;
            };
            text_out += "\n";
        }

        cooker.finish(CookedShader {
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
