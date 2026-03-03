use blade_graphics as gpu;

fn main() {
    env_logger::init();
    let context = unsafe { gpu::Context::init(gpu::ContextDesc::default()).unwrap() };
    let info = context.device_information();
    let caps = context.capabilities();

    println!("GPU information");
    println!("  device_name: {}", info.device_name);
    println!("  driver_name: {}", info.driver_name);
    println!("  driver_info: {}", info.driver_info);
    println!("  software_emulated: {}", info.is_software_emulated);
    println!("GPU capabilities");
    println!("  ray_query: {:?}", caps.ray_query);
    println!("  sample_count_mask: 0x{:X}", caps.sample_count_mask);
    println!("  dual_source_blending: {}", caps.dual_source_blending);
}
