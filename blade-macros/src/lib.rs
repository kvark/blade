mod flat;
mod shader_data;

use proc_macro::TokenStream;

/// Derive the `ShaderData` trait for a struct
///
/// ## Example
///
/// ```rust
/// #[derive(blade_macros::ShaderData)]
/// struct Test {
///   sm: blade_graphics::Sampler,
/// }
/// ```
#[proc_macro_derive(ShaderData)]
pub fn shader_data_derive(input: TokenStream) -> TokenStream {
    let stream = match shader_data::generate(input) {
        Ok(tokens) => tokens,
        Err(err) => err.into_compile_error(),
    };
    stream.into()
}

/// Derive the `Flat` for a type.
///
/// Can either be used on a struct that has every field already implementing `blade_asset::Flat`:
///
/// ```rust
/// #[derive(blade_macros::Flat)]
/// struct FlatData<'a> {
///    array: [u32; 2],
///    single: f32,
///    slice: &'a [u16],
///}
/// ```
///
/// The struct may have a lifetime describing borrowed data members. Borrowing is
/// needed for zero-copy deserialization.
///
/// Alternatively, can be used on a transparent wrapper to force `blade_asset::Flat`
/// implementation even if the wrapped type doesn't implement it:
///
/// ```rust
/// #[derive(Clone, Copy)]
/// #[repr(u32)]
/// #[non_exhaustive]
/// enum Foo {
///     A,
///     B,
/// }
/// #[derive(blade_macros::Flat, Clone, Copy)]
/// #[repr(transparent)]
/// struct FooWrap(Foo);
/// ```
///
/// This can be particularly useful for types like `bytemuck::Pod` implementors,
/// or plain non-exhaustive enums from 3rd party crates.
#[proc_macro_derive(Flat)]
pub fn flat_derive(input: TokenStream) -> TokenStream {
    let stream = match flat::generate(input) {
        Ok(tokens) => tokens,
        Err(err) => err.into_compile_error(),
    };
    stream.into()
}
