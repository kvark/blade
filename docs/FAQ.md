# Frequency Asked Questions

### Why investing into this when there is `wgpu`?

`wgpu` is becoming a standard solution for GPU access in Rust and beyond. It's wonderful, and by any means just use it if you have any doubts. It's a strong local maxima in a chosen space of low-level portability. It may very well be the global maxima as well, but we don't know this until we explore the *other* local maximas. Blade is an attempt to strike where `wgpu` can't reach, it makes a lot of the opposite design solutions. Try it and see.

### How does Blade connect to a larger Rust ecosystem?

Blade happily uses Naga from `wgpu` project for shader translation. It also relies on `gpu-alloc`, `ash`, `metal`, and other crates.
