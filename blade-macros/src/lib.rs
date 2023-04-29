mod flat;
mod shader_data;

use proc_macro::TokenStream;

#[proc_macro_derive(ShaderData)]
pub fn shader_data_derive(input: TokenStream) -> TokenStream {
    let stream = match shader_data::generate(input) {
        Ok(tokens) => tokens,
        Err(err) => err.into_compile_error(),
    };
    stream.into()
}

#[proc_macro_derive(Flat)]
pub fn flat_derive(input: TokenStream) -> TokenStream {
    let stream = match flat::generate(input) {
        Ok(tokens) => tokens,
        Err(err) => err.into_compile_error(),
    };
    stream.into()
}
