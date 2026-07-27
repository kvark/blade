//! Headless benchmark for Blade's pass-boundary synchronization policy.
//!
//! This benchmark intentionally starts with one controlled question: what is
//! the cost of placing Blade's global barrier at every pass boundary instead
//! of only at application-declared hazards? It does not provide a precise
//! per-resource baseline or change image layouts.

use blade_graphics as gpu;
use gpu::ShaderData as _;
use std::{
    env, process,
    time::{Duration, Instant},
};

const COMPUTE_WORKGROUP_SIZE: u32 = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Workload {
    ComputeIndependent,
    ComputeChain,
    GraphicsIndependent,
    GraphicsChain,
    /// Alternating compute and render passes with no dependency between them.
    MixedIndependent,
    /// Alternating compute and render passes; each pass depends on the pass of
    /// the same kind two positions earlier. The benchmark's `HazardOnly`
    /// policy conservatively inserts at every boundary; this is not a minimal
    /// placement for the two-pass dependency distance.
    MixedChain,
}

impl Workload {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "compute-independent" => Ok(Self::ComputeIndependent),
            "compute-chain" => Ok(Self::ComputeChain),
            "graphics-independent" => Ok(Self::GraphicsIndependent),
            "graphics-chain" => Ok(Self::GraphicsChain),
            "mixed-independent" => Ok(Self::MixedIndependent),
            "mixed-chain" => Ok(Self::MixedChain),
            _ => Err(format!("unknown workload: {value}")),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::ComputeIndependent => "compute-independent",
            Self::ComputeChain => "compute-chain",
            Self::GraphicsIndependent => "graphics-independent",
            Self::GraphicsChain => "graphics-chain",
            Self::MixedIndependent => "mixed-independent",
            Self::MixedChain => "mixed-chain",
        }
    }

    fn needs_compute(self) -> bool {
        !matches!(self, Self::GraphicsIndependent | Self::GraphicsChain)
    }

    fn needs_graphics(self) -> bool {
        !matches!(self, Self::ComputeIndependent | Self::ComputeChain)
    }

    fn is_mixed(self) -> bool {
        matches!(self, Self::MixedIndependent | Self::MixedChain)
    }

    fn is_dependent(self) -> bool {
        matches!(
            self,
            Self::ComputeChain | Self::GraphicsChain | Self::MixedChain
        )
    }

    /// Whether pass `index` is a compute pass. Mixed workloads alternate,
    /// starting with compute.
    fn pass_is_compute(self, index: usize) -> bool {
        if self.is_mixed() {
            index.is_multiple_of(2)
        } else {
            self.needs_compute()
        }
    }

    /// Number of compute passes in a command buffer of `passes` passes.
    fn compute_pass_count(self, passes: usize) -> usize {
        if !self.needs_compute() {
            0
        } else if self.is_mixed() {
            passes.div_ceil(2)
        } else {
            passes
        }
    }

    /// Number of render passes in a command buffer of `passes` passes.
    fn graphics_pass_count(self, passes: usize) -> usize {
        if !self.needs_graphics() {
            0
        } else if self.is_mixed() {
            passes / 2
        } else {
            passes
        }
    }

    /// Whether each sub-workload uses distinct resources per pass.
    fn resources_are_independent(self) -> bool {
        !self.is_dependent()
    }
}

/// Where barriers go. Orthogonal to how wide they are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Placement {
    /// Blade places one at every pass boundary.
    Automatic,
    /// The application places barriers only in dependent workloads. This is
    /// minimal for the single-kind chains, but deliberately conservative for
    /// `MixedChain`, where same-kind dependencies are two passes apart.
    HazardOnly,
    /// The application places one before every pass. At global scope this is
    /// command-for-command equal to `Automatic` and is the instrumentation
    /// control. At pass-kind scope an explicit barrier has a wide destination
    /// because its consumer is unknown, so that combination is not a control.
    ExplicitAll,
}

/// The full barrier configuration: a placement crossed with a scope. Every
/// combination is measured. Automatic barriers can derive both source and
/// destination pass kinds; explicit barriers can derive only their source, an
/// intentional interaction rather than a fully symmetric factorial axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BarrierPolicy {
    placement: Placement,
    scope: gpu::BarrierScope,
}

impl BarrierPolicy {
    const ALL: [Self; 6] = [
        Self::new(Placement::Automatic, gpu::BarrierScope::Global),
        Self::new(Placement::Automatic, gpu::BarrierScope::PassKind),
        Self::new(Placement::HazardOnly, gpu::BarrierScope::Global),
        Self::new(Placement::HazardOnly, gpu::BarrierScope::PassKind),
        Self::new(Placement::ExplicitAll, gpu::BarrierScope::Global),
        Self::new(Placement::ExplicitAll, gpu::BarrierScope::PassKind),
    ];

    const fn new(placement: Placement, scope: gpu::BarrierScope) -> Self {
        Self { placement, scope }
    }

    fn parse(value: &str) -> Result<Self, String> {
        let (placement_name, scope) = match value.strip_suffix("-scoped") {
            Some(rest) => (rest, gpu::BarrierScope::PassKind),
            None => (value, gpu::BarrierScope::Global),
        };
        let placement = match placement_name {
            "automatic" => Placement::Automatic,
            "hazard-only" => Placement::HazardOnly,
            "explicit-all" => Placement::ExplicitAll,
            _ => return Err(format!("unknown barrier policy: {value}")),
        };
        Ok(Self::new(placement, scope))
    }

    fn as_str(self) -> &'static str {
        match (self.placement, self.scope) {
            (Placement::Automatic, gpu::BarrierScope::Global) => "automatic",
            (Placement::Automatic, gpu::BarrierScope::PassKind) => "automatic-scoped",
            (Placement::HazardOnly, gpu::BarrierScope::Global) => "hazard-only",
            (Placement::HazardOnly, gpu::BarrierScope::PassKind) => "hazard-only-scoped",
            (Placement::ExplicitAll, gpu::BarrierScope::Global) => "explicit-all",
            (Placement::ExplicitAll, gpu::BarrierScope::PassKind) => "explicit-all-scoped",
        }
    }

    fn uses_manual_mode(self) -> bool {
        self.placement != Placement::Automatic
    }

    fn inserts_before(self, workload: Workload, pass_index: usize) -> bool {
        match self.placement {
            Placement::Automatic => false,
            Placement::HazardOnly => pass_index != 0 && workload.is_dependent(),
            Placement::ExplicitAll => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BarrierPolicy, Placement, Workload, fnv1a64};
    use blade_graphics as gpu;

    fn policy(placement: Placement, scope: gpu::BarrierScope) -> BarrierPolicy {
        BarrierPolicy::new(placement, scope)
    }

    #[test]
    fn barrier_policies_match_the_experiment_contract() {
        for scope in [gpu::BarrierScope::Global, gpu::BarrierScope::PassKind] {
            for pass_index in 0..3 {
                assert!(
                    !policy(Placement::Automatic, scope)
                        .inserts_before(Workload::ComputeIndependent, pass_index)
                );
                assert!(
                    policy(Placement::ExplicitAll, scope)
                        .inserts_before(Workload::ComputeIndependent, pass_index)
                );
            }

            let hazard = policy(Placement::HazardOnly, scope);
            assert!(!hazard.inserts_before(Workload::ComputeChain, 0));
            assert!(hazard.inserts_before(Workload::ComputeChain, 1));
            assert!(!hazard.inserts_before(Workload::ComputeIndependent, 1));
            assert!(hazard.inserts_before(Workload::MixedChain, 1));
        }
    }

    #[test]
    fn every_policy_round_trips_through_its_name() {
        for expected in BarrierPolicy::ALL {
            let parsed = BarrierPolicy::parse(expected.as_str()).unwrap();
            assert_eq!(parsed, expected, "{}", expected.as_str());
        }
        assert_eq!(BarrierPolicy::ALL.len(), 6);
        assert!(BarrierPolicy::parse("nonsense").is_err());
    }

    #[test]
    fn mixed_workloads_alternate_pass_kinds() {
        let workload = Workload::MixedChain;
        assert!(workload.pass_is_compute(0));
        assert!(!workload.pass_is_compute(1));
        assert!(workload.pass_is_compute(2));
        assert_eq!(workload.compute_pass_count(16), 8);
        assert_eq!(workload.graphics_pass_count(16), 8);
        assert_eq!(workload.compute_pass_count(5), 3);
        assert_eq!(workload.graphics_pass_count(5), 2);

        // Non-mixed workloads keep every pass in one family.
        assert_eq!(Workload::ComputeChain.compute_pass_count(16), 16);
        assert_eq!(Workload::ComputeChain.graphics_pass_count(16), 0);
        assert_eq!(Workload::GraphicsChain.compute_pass_count(16), 0);
        assert_eq!(Workload::GraphicsChain.graphics_pass_count(16), 16);
    }

    #[test]
    fn validation_hash_is_fnv1a64() {
        // Published FNV-1a-64 test vector for the ASCII string "hello".
        assert_eq!(fnv1a64(b"hello"), 0xa430_d846_80aa_bd0b);
    }
}

#[derive(Clone, Debug)]
struct Config {
    workload: Workload,
    policy: BarrierPolicy,
    passes: usize,
    elements: u32,
    rounds: u32,
    width: u32,
    height: u32,
    warmups: usize,
    samples: usize,
    device_id: Option<u32>,
    validation: bool,
    allow_software: bool,
    gpu_timing: bool,
    list_adapters: bool,
    capture: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            workload: Workload::ComputeIndependent,
            policy: BarrierPolicy::new(Placement::Automatic, gpu::BarrierScope::Global),
            passes: 16,
            elements: 1 << 20,
            rounds: 8,
            width: 1024,
            height: 1024,
            warmups: 10,
            samples: 30,
            device_id: None,
            validation: false,
            allow_software: false,
            gpu_timing: true,
            list_adapters: false,
            capture: false,
        }
    }
}

impl Config {
    fn parse() -> Result<Self, String> {
        let mut config = Self::default();
        let mut args = env::args().skip(1);
        while let Some(argument) = args.next() {
            match argument.as_str() {
                "--workload" => {
                    config.workload = Workload::parse(&next_value(&mut args, "--workload")?)?
                }
                "--policy" => {
                    config.policy = BarrierPolicy::parse(&next_value(&mut args, "--policy")?)?
                }
                "--passes" => {
                    config.passes = parse_value(&next_value(&mut args, "--passes")?, "--passes")?
                }
                "--elements" => {
                    config.elements =
                        parse_value(&next_value(&mut args, "--elements")?, "--elements")?
                }
                "--rounds" => {
                    config.rounds = parse_value(&next_value(&mut args, "--rounds")?, "--rounds")?
                }
                "--width" => {
                    config.width = parse_value(&next_value(&mut args, "--width")?, "--width")?
                }
                "--height" => {
                    config.height = parse_value(&next_value(&mut args, "--height")?, "--height")?
                }
                "--warmups" => {
                    config.warmups = parse_value(&next_value(&mut args, "--warmups")?, "--warmups")?
                }
                "--samples" => {
                    config.samples = parse_value(&next_value(&mut args, "--samples")?, "--samples")?
                }
                "--device-id" => {
                    config.device_id = Some(parse_value(
                        &next_value(&mut args, "--device-id")?,
                        "--device-id",
                    )?)
                }
                "--validation" => config.validation = true,
                "--allow-software" => config.allow_software = true,
                "--no-gpu-timing" => config.gpu_timing = false,
                "--list-adapters" => config.list_adapters = true,
                "--capture" => config.capture = true,
                "--help" | "-h" => {
                    print_usage();
                    process::exit(0);
                }
                _ => return Err(format!("unknown argument: {argument}")),
            }
        }

        if config.passes == 0 || config.passes >= gpu::limits::PASS_COUNT {
            return Err(format!(
                "--passes must be in 1..{}",
                gpu::limits::PASS_COUNT
            ));
        }
        if config.elements == 0 {
            return Err("--elements must be nonzero".into());
        }
        if config.width == 0 || config.height == 0 {
            return Err("--width and --height must be nonzero".into());
        }
        if config.samples == 0 {
            return Err("--samples must be nonzero".into());
        }
        Ok(config)
    }
}

fn next_value(args: &mut impl Iterator<Item = String>, argument: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("missing value after {argument}"))
}

fn parse_value<T: std::str::FromStr>(value: &str, argument: &str) -> Result<T, String> {
    value
        .parse()
        .map_err(|_| format!("invalid value for {argument}: {value}"))
}

fn print_usage() {
    let policies = BarrierPolicy::ALL
        .iter()
        .map(|policy| policy.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    println!(
        "\
Blade synchronization benchmark

Usage:
  cargo run --release --example sync-bench -- [options]

Options:
  --workload <name>   compute-independent, compute-chain,
                      graphics-independent, graphics-chain,
                      mixed-independent, or mixed-chain
  --policy <name>     {policies}
  --passes <count>    passes per measured command buffer (default: 16)
  --elements <count>  u32 elements in compute buffers (default: 1048576)
  --rounds <count>    shader mixing rounds per invocation (default: 8)
  --width <pixels>    render-target width (default: 1024)
  --height <pixels>   render-target height (default: 1024)
  --warmups <count>   unreported warm-up iterations (default: 10)
  --samples <count>   reported iterations (default: 30)
  --device-id <id>    Blade/Vulkan physical device ID
  --validation        enable API and shader validation
  --allow-software    permit a software Vulkan device (correctness only)
  --no-gpu-timing     disable timestamp queries for CPU-only collection
  --list-adapters     list selectable adapters and exit
  --capture           wrap one measured iteration in a RenderDoc capture
                      (requires librenderdoc.so to be loaded, e.g. via
                      LD_PRELOAD; see paper/capture-streams.py)
  -h, --help          show this help
"
    );
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ComputeParams {
    element_count: u32,
    rounds: u32,
    seed: u32,
    _padding: u32,
}

#[derive(blade_macros::ShaderData)]
struct ComputeData {
    input_data: gpu::BufferPiece,
    output_data: gpu::BufferPiece,
    compute_params: ComputeParams,
}

struct ComputeBench {
    pipeline: gpu::ComputePipeline,
    buffers: Vec<gpu::Buffer>,
    element_count: u32,
    rounds: u32,
    independent: bool,
}

impl ComputeBench {
    fn new(context: &gpu::Context, config: &Config) -> Self {
        let shader = context.create_shader(gpu::ShaderDesc {
            source: include_str!("compute.wgsl"),
            naga_module: None,
        });
        let layout = ComputeData::layout();
        let pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "sync-bench-compute",
            data_layouts: &[&layout],
            compute: shader.at("cs_main"),
        });

        let independent = config.workload.resources_are_independent();
        let pass_count = config.workload.compute_pass_count(config.passes);
        let buffer_count = if independent { pass_count + 1 } else { 2 };
        let buffer_size = u64::from(config.elements) * 4;
        let buffers = (0..buffer_count)
            .map(|index| {
                context.create_buffer(gpu::BufferDesc {
                    name: &format!("sync-bench-buffer-{index}"),
                    size: buffer_size,
                    memory: gpu::Memory::Device,
                })
            })
            .collect::<Vec<_>>();

        let mut setup = context.create_command_encoder(gpu::CommandEncoderDesc {
            name: "sync-bench-compute-setup",
            buffer_count: 1,
            manual_barriers: false,
            barrier_scope: gpu::BarrierScope::Global,
        });
        setup.start();
        {
            let mut transfer = setup.transfer("initialize compute buffers");
            for (index, &buffer) in buffers.iter().enumerate() {
                transfer.fill_buffer(buffer.into(), buffer_size, (index + 1) as u8);
            }
        }
        let sync_point = context.submit(&mut setup);
        assert!(context.wait_for(&sync_point, !0).unwrap());
        context.destroy_command_encoder(&mut setup);

        Self {
            pipeline,
            buffers,
            element_count: config.elements,
            rounds: config.rounds,
            independent,
        }
    }

    fn record_pass(&self, encoder: &mut gpu::CommandEncoder, pass_index: usize, iteration: usize) {
        let (input, output) = if self.independent {
            (self.buffers[0], self.buffers[pass_index + 1])
        } else {
            (
                self.buffers[pass_index % 2],
                self.buffers[(pass_index + 1) % 2],
            )
        };
        let mut compute = encoder.compute("sync-bench-compute-pass");
        let mut pipeline = compute.with(&self.pipeline);
        pipeline.bind(
            0,
            &ComputeData {
                input_data: input.into(),
                output_data: output.into(),
                compute_params: ComputeParams {
                    element_count: self.element_count,
                    rounds: self.rounds,
                    seed: (iteration as u32)
                        .wrapping_mul(0x9E37_79B9)
                        .wrapping_add(pass_index as u32),
                    _padding: 0,
                },
            },
        );
        pipeline.dispatch([self.element_count.div_ceil(COMPUTE_WORKGROUP_SIZE), 1, 1]);
    }

    fn validate(&self, context: &gpu::Context, compute_passes: usize) -> u64 {
        let outputs: Vec<gpu::Buffer> = if self.independent {
            self.buffers[1..=compute_passes].to_vec()
        } else {
            vec![self.buffers[compute_passes % 2]]
        };
        let output_size = u64::from(self.element_count.min(1024)) * 4;
        let readback_size = output_size * outputs.len() as u64;
        let readback = context.create_buffer(gpu::BufferDesc {
            name: "sync-bench-compute-readback",
            size: readback_size,
            memory: gpu::Memory::Shared,
        });
        let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
            name: "sync-bench-compute-validate",
            buffer_count: 1,
            manual_barriers: false,
            barrier_scope: gpu::BarrierScope::Global,
        });
        encoder.start();
        {
            let mut transfer = encoder.transfer("read compute output");
            for (index, &output) in outputs.iter().enumerate() {
                transfer.copy_buffer_to_buffer(
                    output.into(),
                    gpu::BufferPiece {
                        buffer: readback,
                        offset: index as u64 * output_size,
                    },
                    output_size,
                );
            }
        }
        let sync_point = context.submit(&mut encoder);
        assert!(context.wait_for(&sync_point, !0).unwrap());

        let bytes = unsafe { std::slice::from_raw_parts(readback.data(), readback_size as usize) };
        for (index, output) in bytes.chunks_exact(output_size as usize).enumerate() {
            assert!(
                output.iter().any(|&byte| byte != 0),
                "compute output {index} validation produced only zero bytes"
            );
        }
        let hash = fnv1a64(bytes);
        context.destroy_command_encoder(&mut encoder);
        context.destroy_buffer(readback);
        hash
    }

    fn destroy(mut self, context: &gpu::Context) {
        context.destroy_compute_pipeline(&mut self.pipeline);
        for buffer in self.buffers {
            context.destroy_buffer(buffer);
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GraphicsParams {
    rounds: u32,
    seed: u32,
    _padding_a: u32,
    _padding_b: u32,
}

#[derive(blade_macros::ShaderData)]
struct GraphicsData {
    graphics_params: GraphicsParams,
}

struct GraphicsBench {
    pipeline: gpu::RenderPipeline,
    textures: Vec<gpu::Texture>,
    views: Vec<gpu::TextureView>,
    extent: gpu::Extent,
    rounds: u32,
    independent: bool,
}

impl GraphicsBench {
    fn new(context: &gpu::Context, config: &Config) -> Self {
        let shader = context.create_shader(gpu::ShaderDesc {
            source: include_str!("graphics.wgsl"),
            naga_module: None,
        });
        let layout = GraphicsData::layout();
        let color_targets = [gpu::ColorTargetState {
            format: gpu::TextureFormat::Rgba8Unorm,
            blend: Some(gpu::BlendState::ADDITIVE),
            write_mask: gpu::ColorWrites::ALL,
        }];
        let pipeline = context.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "sync-bench-graphics",
            data_layouts: &[&layout],
            vertex: shader.at("vs_main"),
            vertex_fetches: &[],
            primitive: gpu::PrimitiveState::default(),
            depth_stencil: None,
            fragment: Some(shader.at("fs_main")),
            color_targets: &color_targets,
            multisample_state: gpu::MultisampleState::default(),
        });

        let independent = config.workload.resources_are_independent();
        let pass_count = config.workload.graphics_pass_count(config.passes);
        // `max(1)` keeps a target available for validation when a sweep lands
        // on a mixed workload short enough to contain no render pass.
        let target_count = if independent { pass_count.max(1) } else { 1 };
        let extent = gpu::Extent {
            width: config.width,
            height: config.height,
            depth: 1,
        };
        let mut textures = Vec::with_capacity(target_count);
        let mut views = Vec::with_capacity(target_count);
        for index in 0..target_count {
            let texture = context.create_texture(gpu::TextureDesc {
                name: &format!("sync-bench-target-{index}"),
                format: gpu::TextureFormat::Rgba8Unorm,
                size: extent,
                dimension: gpu::TextureDimension::D2,
                array_layer_count: 1,
                mip_level_count: 1,
                usage: gpu::TextureUsage::TARGET | gpu::TextureUsage::COPY,
                sample_count: 1,
                external: None,
            });
            let view = context.create_texture_view(
                texture,
                gpu::TextureViewDesc {
                    name: &format!("sync-bench-target-view-{index}"),
                    format: gpu::TextureFormat::Rgba8Unorm,
                    dimension: gpu::ViewDimension::D2,
                    subresources: &gpu::TextureSubresources::default(),
                },
            );
            textures.push(texture);
            views.push(view);
        }

        let mut setup = context.create_command_encoder(gpu::CommandEncoderDesc {
            name: "sync-bench-graphics-setup",
            buffer_count: 1,
            manual_barriers: false,
            barrier_scope: gpu::BarrierScope::Global,
        });
        setup.start();
        for &texture in &textures {
            setup.init_texture(texture);
        }
        let sync_point = context.submit(&mut setup);
        assert!(context.wait_for(&sync_point, !0).unwrap());
        context.destroy_command_encoder(&mut setup);

        Self {
            pipeline,
            textures,
            views,
            extent,
            rounds: config.rounds,
            independent,
        }
    }

    fn record_pass(&self, encoder: &mut gpu::CommandEncoder, pass_index: usize, iteration: usize) {
        let view = if self.independent {
            self.views[pass_index]
        } else {
            self.views[0]
        };
        let init_op = if self.independent || pass_index == 0 {
            gpu::InitOp::Clear(gpu::TextureColor::TransparentBlack)
        } else {
            gpu::InitOp::Load
        };
        let colors = [gpu::RenderTarget {
            view,
            init_op,
            finish_op: gpu::FinishOp::Store,
        }];
        let mut render = encoder.render(
            "sync-bench-graphics-pass",
            gpu::RenderTargetSet {
                colors: &colors,
                depth_stencil: None,
            },
        );
        let mut pipeline = render.with(&self.pipeline);
        pipeline.bind(
            0,
            &GraphicsData {
                graphics_params: GraphicsParams {
                    rounds: self.rounds,
                    seed: (iteration as u32)
                        .wrapping_mul(0x9E37_79B9)
                        .wrapping_add(pass_index as u32),
                    _padding_a: 0,
                    _padding_b: 0,
                },
            },
        );
        pipeline.draw(0, 3, 0, 1);
    }

    fn validate(&self, context: &gpu::Context) -> u64 {
        let output_count = if self.independent {
            self.textures.len()
        } else {
            1
        };
        let valid_bytes_per_row = self.extent.width * 4;
        let bytes_per_row = valid_bytes_per_row.div_ceil(256) * 256;
        let readback_size = u64::from(bytes_per_row) * output_count as u64;
        let readback = context.create_buffer(gpu::BufferDesc {
            name: "sync-bench-graphics-readback",
            size: readback_size,
            memory: gpu::Memory::Shared,
        });
        let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
            name: "sync-bench-graphics-validate",
            buffer_count: 1,
            manual_barriers: false,
            barrier_scope: gpu::BarrierScope::Global,
        });
        encoder.start();
        {
            let mut transfer = encoder.transfer("read graphics output");
            for (index, &texture) in self.textures[..output_count].iter().enumerate() {
                transfer.copy_texture_to_buffer(
                    texture.into(),
                    gpu::BufferPiece {
                        buffer: readback,
                        offset: index as u64 * u64::from(bytes_per_row),
                    },
                    bytes_per_row,
                    gpu::Extent {
                        width: self.extent.width,
                        height: 1,
                        depth: 1,
                    },
                );
            }
        }
        let sync_point = context.submit(&mut encoder);
        assert!(context.wait_for(&sync_point, !0).unwrap());

        let bytes = unsafe { std::slice::from_raw_parts(readback.data(), readback_size as usize) };
        let mut logical_bytes = Vec::with_capacity(valid_bytes_per_row as usize * output_count);
        for (index, row) in bytes.chunks_exact(bytes_per_row as usize).enumerate() {
            let output = &row[..valid_bytes_per_row as usize];
            assert!(
                output.iter().any(|&byte| byte != 0),
                "graphics output {index} validation produced only zero bytes"
            );
            logical_bytes.extend_from_slice(output);
        }
        let hash = fnv1a64(&logical_bytes);
        context.destroy_command_encoder(&mut encoder);
        context.destroy_buffer(readback);
        hash
    }

    fn destroy(mut self, context: &gpu::Context) {
        context.destroy_render_pipeline(&mut self.pipeline);
        for view in self.views {
            context.destroy_texture_view(view);
        }
        for texture in self.textures {
            context.destroy_texture(texture);
        }
    }
}

enum Bench {
    Compute(ComputeBench),
    Graphics(GraphicsBench),
    Mixed(ComputeBench, GraphicsBench),
}

impl Bench {
    fn new(context: &gpu::Context, config: &Config) -> Self {
        match (
            config.workload.needs_compute(),
            config.workload.needs_graphics(),
        ) {
            (true, false) => Self::Compute(ComputeBench::new(context, config)),
            (false, true) => Self::Graphics(GraphicsBench::new(context, config)),
            _ => Self::Mixed(
                ComputeBench::new(context, config),
                GraphicsBench::new(context, config),
            ),
        }
    }

    fn record(&self, encoder: &mut gpu::CommandEncoder, config: &Config, iteration: usize) {
        let mut compute_index = 0;
        let mut graphics_index = 0;
        for pass_index in 0..config.passes {
            if config.policy.inserts_before(config.workload, pass_index) {
                encoder.barrier(config.policy.scope);
            }
            let compute = config.workload.pass_is_compute(pass_index);
            match *self {
                Self::Compute(ref bench) => {
                    bench.record_pass(encoder, compute_index, iteration);
                    compute_index += 1;
                }
                Self::Graphics(ref bench) => {
                    bench.record_pass(encoder, graphics_index, iteration);
                    graphics_index += 1;
                }
                Self::Mixed(ref compute_bench, ref graphics_bench) => {
                    if compute {
                        compute_bench.record_pass(encoder, compute_index, iteration);
                        compute_index += 1;
                    } else {
                        graphics_bench.record_pass(encoder, graphics_index, iteration);
                        graphics_index += 1;
                    }
                }
            }
        }
    }

    fn validate(&self, context: &gpu::Context, workload: Workload, passes: usize) -> u64 {
        let compute_passes = workload.compute_pass_count(passes);
        match *self {
            Self::Compute(ref bench) => bench.validate(context, compute_passes),
            Self::Graphics(ref bench) => bench.validate(context),
            Self::Mixed(ref compute_bench, ref graphics_bench) => {
                let compute_hash = compute_bench.validate(context, compute_passes);
                // A mixed workload short enough to contain no render pass has
                // an untouched target, which is not evidence of anything.
                if workload.graphics_pass_count(passes) == 0 {
                    return compute_hash;
                }
                let graphics_hash = graphics_bench.validate(context);
                fnv1a64(&[compute_hash.to_le_bytes(), graphics_hash.to_le_bytes()].concat())
            }
        }
    }

    fn destroy(self, context: &gpu::Context) {
        match self {
            Self::Compute(bench) => bench.destroy(context),
            Self::Graphics(bench) => bench.destroy(context),
            Self::Mixed(compute_bench, graphics_bench) => {
                compute_bench.destroy(context);
                graphics_bench.destroy(context);
            }
        }
    }
}

#[derive(Clone, Copy)]
struct HostTimes {
    start: Duration,
    record: Duration,
    submit: Duration,
    wait: Duration,
}

/// Wraps one measured iteration in a RenderDoc capture, when the library is
/// present in the process.
///
/// RenderDoc normally delimits captures at swapchain presents, and this
/// benchmark is headless, so the capture has to be requested explicitly. The
/// library has to be loaded already, which `paper/capture-streams.py` arranges
/// with `LD_PRELOAD`; without it `RenderDoc::new` fails and the run continues
/// uncaptured rather than aborting a measurement.
#[cfg(any(target_os = "linux", target_os = "windows"))]
struct Capture(Option<renderdoc::RenderDoc<renderdoc::V141>>);

#[cfg(any(target_os = "linux", target_os = "windows"))]
impl Capture {
    fn new(enabled: bool, template: &str) -> Self {
        if !enabled {
            return Self(None);
        }
        match renderdoc::RenderDoc::<renderdoc::V141>::new() {
            Ok(mut api) => {
                api.set_capture_file_path_template(template);
                Self(Some(api))
            }
            Err(error) => {
                eprintln!(
                    "warning: --capture requested but RenderDoc is not loaded ({error}); \
                     preload librenderdoc.so to capture"
                );
                Self(None)
            }
        }
    }

    fn begin(&mut self) {
        if let Some(ref mut api) = self.0 {
            api.start_frame_capture(std::ptr::null(), std::ptr::null());
        }
    }

    fn end(&mut self) {
        if let Some(ref mut api) = self.0 {
            api.end_frame_capture(std::ptr::null(), std::ptr::null());
        }
    }
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
struct Capture;

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
impl Capture {
    fn new(enabled: bool, _template: &str) -> Self {
        if enabled {
            eprintln!("warning: --capture is not supported on this platform");
        }
        Self
    }
    fn begin(&mut self) {}
    fn end(&mut self) {}
}

fn duration_ns(duration: Duration) -> u64 {
    duration.as_nanos().min(u128::from(u64::MAX)) as u64
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn csv_string(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\"\""))
}

fn backend_name() -> &'static str {
    #[cfg(gles)]
    {
        return "gles";
    }
    #[cfg(all(not(gles), any(target_os = "macos", target_os = "ios")))]
    {
        return "metal";
    }
    #[cfg(all(
        not(gles),
        not(any(target_os = "macos", target_os = "ios", target_arch = "wasm32"))
    ))]
    {
        "vulkan"
    }
    #[cfg(all(not(gles), target_arch = "wasm32"))]
    {
        "gles"
    }
}

fn main() {
    let config = Config::parse().unwrap_or_else(|error| {
        eprintln!("error: {error}\n");
        print_usage();
        process::exit(2);
    });

    if config.list_adapters {
        for report in gpu::Context::enumerate().unwrap_or_else(|error| {
            eprintln!("error: failed to enumerate adapters: {error}");
            process::exit(2);
        }) {
            println!(
                "0x{:04x}\t{}\tsoftware={}\t{:?}",
                report.device_id,
                report.information.device_name,
                report.information.is_software_emulated,
                report.status,
            );
        }
        return;
    }

    let vulkan_only_policy = BarrierPolicy::new(Placement::Automatic, gpu::BarrierScope::Global);
    if backend_name() != "vulkan" && config.policy != vulkan_only_policy {
        // Both `manual_barriers` and `barrier_scope` are Vulkan concepts; the
        // other backends would silently run something else.

        eprintln!(
            "error: {} only supports the automatic policy; selected policy is {}",
            backend_name(),
            config.policy.as_str(),
        );
        process::exit(2);
    }
    if backend_name() == "gles" {
        eprintln!(
            "error: sync-bench requires Vulkan or Metal; selected backend is {}",
            backend_name()
        );
        process::exit(2);
    }

    let context = unsafe {
        gpu::Context::init(gpu::ContextDesc {
            validation: config.validation,
            timing: config.gpu_timing,
            device_id: config.device_id,
            ..Default::default()
        })
        .unwrap_or_else(|error| {
            eprintln!("error: failed to initialize GPU context: {error}");
            process::exit(2);
        })
    };
    let device = context.device_information();
    if device.is_software_emulated && !config.allow_software {
        eprintln!(
            "error: {} is a software device; pass --allow-software for correctness-only runs",
            device.device_name
        );
        process::exit(2);
    }

    println!("# schema,blade-sync-bench-v1");
    println!("# implementation,blade");
    println!("# backend,{}", backend_name());
    println!("# device_name,{}", csv_string(&device.device_name));
    println!("# driver_name,{}", csv_string(&device.driver_name));
    println!("# driver_info,{}", csv_string(&device.driver_info));
    println!("# software_emulated,{}", device.is_software_emulated);
    println!("# validation,{}", config.validation);
    println!("# gpu_timing,{}", config.gpu_timing);
    println!(
        "sample,workload,policy,passes,elements,rounds,width,height,start_ns,record_ns,submit_ns,wait_ns,gpu_ns,gpu_pass_count"
    );

    let bench = Bench::new(&context, &config);
    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "sync-bench",
        buffer_count: 1,
        manual_barriers: config.policy.uses_manual_mode(),
        barrier_scope: config.policy.scope,
    });

    let iteration_count = config.warmups + config.samples;
    // Capture a single warmed iteration: the first one after the warmups, so
    // pipelines and descriptor pools are established and the command stream is
    // the steady-state one the timings describe.
    let capture_iteration = config.warmups;
    let mut capture = Capture::new(
        config.capture,
        &format!(
            "sync-bench__{}__{}",
            config.workload.as_str(),
            config.policy.as_str()
        ),
    );
    let mut previous_host_times: Option<HostTimes> = None;
    for iteration in 0..=iteration_count {
        if config.capture && iteration == capture_iteration {
            capture.begin();
        }
        let start_begin = Instant::now();
        encoder.start();
        let start_time = start_begin.elapsed();

        if let Some(host_times) = previous_host_times.take() {
            let gpu_timings = encoder.timings();
            let gpu_ns = gpu_timings
                .iter()
                .map(|(_, duration)| duration.as_nanos())
                .sum::<u128>()
                .min(u128::from(u64::MAX)) as u64;
            let completed_iteration = iteration - 1;
            if completed_iteration >= config.warmups {
                println!(
                    "{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                    completed_iteration - config.warmups,
                    config.workload.as_str(),
                    config.policy.as_str(),
                    config.passes,
                    config.elements,
                    config.rounds,
                    config.width,
                    config.height,
                    duration_ns(host_times.start),
                    duration_ns(host_times.record),
                    duration_ns(host_times.submit),
                    duration_ns(host_times.wait),
                    gpu_ns,
                    gpu_timings.len(),
                );
            }
        }

        if iteration == iteration_count {
            let sync_point = context.submit(&mut encoder);
            assert!(context.wait_for(&sync_point, !0).unwrap());
            break;
        }

        let record_begin = Instant::now();
        bench.record(&mut encoder, &config, iteration);
        let record_time = record_begin.elapsed();

        let submit_begin = Instant::now();
        let sync_point = context.submit(&mut encoder);
        let submit_time = submit_begin.elapsed();

        let wait_begin = Instant::now();
        assert!(context.wait_for(&sync_point, !0).unwrap());
        let wait_time = wait_begin.elapsed();

        previous_host_times = Some(HostTimes {
            start: start_time,
            record: record_time,
            submit: submit_time,
            wait: wait_time,
        });

        if config.capture && iteration == capture_iteration {
            capture.end();
        }
    }

    let validation_hash = bench.validate(&context, config.workload, config.passes);
    println!("# validation_hash,fnv1a64-standard:{validation_hash:016x}");
    context.destroy_command_encoder(&mut encoder);
    bench.destroy(&context);
}
