//! Compile `shaders/` to Naga modules.
//!
//! The same sources are part of this crate, so `rustc` type-checks them. The
//! GPU runs the modules this build writes; `src/lib.rs` includes them.

fn main() {
    synaga::build::Shaders::new()
        .dir("shaders")
        .bindings(synaga::build::Bindings::Host)
        .capabilities(blade_caps())
        .run();
}

fn blade_caps() -> synaga::naga::valid::Capabilities {
    use synaga::naga::valid::Capabilities as C;
    C::RAY_QUERY
        | C::STORAGE_BUFFER_BINDING_ARRAY
        | C::STORAGE_BUFFER_BINDING_ARRAY_NON_UNIFORM_INDEXING
        | C::TEXTURE_AND_SAMPLER_BINDING_ARRAY
        | C::TEXTURE_AND_SAMPLER_BINDING_ARRAY_NON_UNIFORM_INDEXING
}
