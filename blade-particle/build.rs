fn main() {
    synaga::build::Shaders::new()
        .bindings(synaga::build::Bindings::Host)
        // `src/shaders/mod.rs` says `check_layout!()`.
        .layout_checks_included()
        .run();
}
