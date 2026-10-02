fn main() {
    synaga::build::Shaders::new()
        .bindings(synaga::build::Bindings::Host)
        .run();
}
