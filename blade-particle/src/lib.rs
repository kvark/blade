#![allow(
    // We don't use syntax sugar where it's not necessary.
    clippy::match_like_matches_macro,
    // Redundant matching is more explicit.
    clippy::redundant_pattern_matching,
    // Explicit lifetimes are often easier to reason about.
    clippy::needless_lifetimes,
    // No need for defaults in the internal types.
    clippy::new_without_default,
    // Matches are good and extendable, no need to make an exception here.
    clippy::single_match,
    // Push commands are more regular than macros.
    clippy::vec_init_then_push,
    // This is the land of unsafe.
    clippy::missing_safety_doc,
)]
#![warn(
    trivial_numeric_casts,
    unused_extern_crates,
    //TODO: re-enable. Currently doesn't like "mem::size_of" on newer Rust
    //unused_qualifications,
    // We don't match on a reference, unless required.
    clippy::pattern_type_mismatch,
)]

mod shaders;
mod system;

mod shader_ir {
    synaga_shader::include_ir!();
}

pub use shaders::particle::CameraParams;
pub use system::{ParticlePipeline, ParticleSystem, PipelineDesc};

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub enum EmitterShape {
    Point,
    Sphere { radius: f32 },
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct Emitter {
    /// Particles emitted per second (0 = burst-only).
    pub rate: f32,
    /// Number of particles per burst trigger.
    pub burst_count: u32,
    /// Shape from which particles originate.
    pub shape: EmitterShape,
    /// Half-angle of the emission cone in radians.
    /// 0 = emit only along direction, PI = full sphere (default).
    #[serde(default = "default_cone_angle")]
    pub cone_angle: f32,
}

fn default_cone_angle() -> f32 {
    std::f32::consts::PI
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub enum ColorConfig {
    /// Single solid color [r, g, b, a].
    Solid([u8; 4]),
    /// Random pick from a palette.
    Palette(Vec<[u8; 4]>),
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct ParticleConfig {
    /// Lifetime range in seconds [min, max].
    pub life: [f32; 2],
    /// Initial speed range [min, max].
    pub speed: [f32; 2],
    /// Scale range [min, max].
    pub scale: [f32; 2],
    /// Color configuration.
    pub color: ColorConfig,
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct ParticleEffect {
    /// Maximum number of live particles.
    pub capacity: u32,
    /// Emitter configuration.
    pub emitter: Emitter,
    /// Particle property ranges.
    pub particle: ParticleConfig,
}

impl ParticleEffect {
    pub fn load(source: &str) -> Result<Self, ron::error::SpannedError> {
        ron::from_str(source)
    }
}

#[cfg(test)]
mod tests {
    /// The module this crate embeds, read back with the Naga that will read it.
    ///
    /// The workspace's `[patch.crates-io]` makes the build script's Naga and
    /// this crate's one package, so the bytes were written by the Naga this
    /// reads them with. `decode` refusing trailing bytes is most of the proof
    /// of that, and reading the module into the Naga that will really read it
    /// is the rest of it. A GPU test would say the same thing where there is a
    /// GPU, which is why this is here: `blade-render`'s version of it runs on
    /// every runner, and these two modules did not.
    #[test]
    fn stock_ir_decodes_and_validates() {
        let flags = naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS;
        for (name, ir) in crate::shader_ir::ALL {
            let module: naga::Module = ir
                .decode()
                .unwrap_or_else(|err| panic!("{name} did not decode: {err}"));
            naga::valid::Validator::new(flags, naga::valid::Capabilities::empty())
                .validate(&module)
                .unwrap_or_else(|err| panic!("{name} failed validation: {err}"));
        }
    }
}
