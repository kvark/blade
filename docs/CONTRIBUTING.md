# Code Style Guide

- keep dependencies and amount of code low
- simple is good, don't overcomplicate or anticipate
- one `use` per crate, prefer importing modules instead of concrete types/functions
- don't rely on implicit references via `match`, always use explicit `ref` instead

# Snapshot Tests

Reference image tests in `tests/gpu_examples.rs` render examples headlessly and compare against PNGs in `tests/reference/`.

Run:
```sh
cargo test --test gpu_examples -- --ignored
```

Update references after intentional rendering changes:
```sh
BLADE_UPDATE_SNAPSHOTS=1 cargo test --test gpu_examples -- --ignored
```
