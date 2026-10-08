fn main() {
    synaga::build::Shaders::new()
        .dir("examples/bunnymark/shaders")
        .bindings(synaga::build::Bindings::Host)
        .module_name("bunnymark_shaders.rs")
        // `examples/bunnymark/shaders/mod.rs` says `check_layout!(..)`.
        .layout_checks_included()
        .run();
}
