//! The differentiable tracer of `blade-render`: its gradients against central
//! differences, and recoveries of scenes by optimizing them. These take about
//! an hour and a half on Lavapipe, so they are their own target, apart from
//! `gpu_examples`, and CI only runs a few of them:
//!
//! ```sh
//! cargo test --test diff_tracer -- --ignored --nocapture --test-threads=1
//! ```

mod diff;
mod pbr_harness;
mod pbr_scene;

const SHADER_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/blade-render/code");

use pbr_harness::PbrHarness;
