use blade_graphics as gpu;

fn main() {
    env_logger::init();
    for device in gpu::Context::enumerate().unwrap() {
        let (status, caps) = match &device.status {
            gpu::DeviceReportStatus::Available { is_default, caps } => {
                let label = if *is_default { "default" } else { "available" };
                (label.to_string(), Some(caps))
            }
            gpu::DeviceReportStatus::Rejected(reason) => (format!("rejected: {reason}"), None),
        };
        println!("Device 0x{:X}: {}", device.device_id, status);
        println!("  name: {}", device.information.device_name);
        println!(
            "  driver: {} ({})",
            device.information.driver_name, device.information.driver_info
        );
        println!(
            "  software_emulated: {}",
            device.information.is_software_emulated
        );
        if let Some(caps) = caps {
            println!("  ray_query: {:?}", caps.ray_query);
            println!("  sample_count_mask: 0x{:X}", caps.sample_count_mask);
            println!("  dual_source_blending: {}", caps.dual_source_blending);
        }
    }
}
