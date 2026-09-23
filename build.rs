fn main() {
    synaga::build::Shaders::new()
        .dir("examples/bunnymark/shaders")
        .bindings(synaga::build::Bindings::Host)
        .module_name("bunnymark_shaders.rs")
        .run();
}
