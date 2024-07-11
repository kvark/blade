pub fn populate_debug_hud(debug: &mut blade_render::DebugConfig, ui: &mut egui::Ui) {
    use strum::IntoEnumIterator as _;

    egui::ComboBox::from_label("View mode")
        .selected_text(format!("{:?}", debug.view_mode))
        .show_ui(ui, |ui| {
            for value in blade_render::DebugMode::iter() {
                ui.selectable_value(&mut debug.view_mode, value, format!("{value:?}"));
            }
        });

    ui.label("Draw debug:");
    for (name, bit) in blade_render::DebugDrawFlags::all().iter_names() {
        let mut enabled = debug.draw_flags.contains(bit);
        ui.checkbox(&mut enabled, name);
        debug.draw_flags.set(bit, enabled);
    }
    ui.label("Ignore textures:");
    for (name, bit) in blade_render::DebugTextureFlags::all().iter_names() {
        let mut enabled = debug.texture_flags.contains(bit);
        ui.checkbox(&mut enabled, name);
        debug.texture_flags.set(bit, enabled);
    }
}

pub fn populate_debug_selection(
    mouse_pos: &mut Option<[i32; 2]>,
    selection: &blade_render::SelectionInfo,
    asset_hub: &blade_render::AssetHub,
    ui: &mut egui::Ui,
) {
    let screen_pos = match *mouse_pos {
        Some(pos) => pos,
        None => return,
    };

    let style = ui.style();
    egui::Frame::group(style).show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label("Pixel:");
            ui.colored_label(
                egui::Color32::WHITE,
                format!("{}x{}", screen_pos[0], screen_pos[1]),
            );
            if ui.button("Unselect").clicked() {
                *mouse_pos = None;
            }
        });
        ui.horizontal(|ui| {
            let sd = &selection.std_deviation;
            ui.label("Std Deviation:");
            ui.colored_label(
                egui::Color32::WHITE,
                format!("{:.2} {:.2} {:.2}", sd.x, sd.y, sd.z),
            );
        });
        ui.horizontal(|ui| {
            ui.label("Samples:");
            let power = selection
                .std_deviation_history
                .next_power_of_two()
                .trailing_zeros();
            ui.colored_label(egui::Color32::WHITE, format!("2^{}", power));
        });
        ui.horizontal(|ui| {
            ui.label("Depth:");
            ui.colored_label(egui::Color32::WHITE, format!("{:.2}", selection.depth));
        });
        ui.horizontal(|ui| {
            let tc = &selection.tex_coords;
            ui.label("Texture coords:");
            ui.colored_label(egui::Color32::WHITE, format!("{:.2} {:.2}", tc.x, tc.y));
        });
        ui.horizontal(|ui| {
            let wp = &selection.position;
            ui.label("World pos:");
            ui.colored_label(
                egui::Color32::WHITE,
                format!("{:.2} {:.2} {:.2}", wp.x, wp.y, wp.z),
            );
        });
        ui.horizontal(|ui| {
            ui.label("Base color:");
            if let Some(handle) = selection.base_color_texture {
                let name = asset_hub
                    .textures
                    .get_main_source_path(handle)
                    .map(|path| path.display().to_string())
                    .unwrap_or_default();
                ui.colored_label(egui::Color32::WHITE, name);
            }
        });
        ui.horizontal(|ui| {
            ui.label("Normal:");
            if let Some(handle) = selection.normal_texture {
                let name = asset_hub
                    .textures
                    .get_main_source_path(handle)
                    .map(|path| path.display().to_string())
                    .unwrap_or_default();
                ui.colored_label(egui::Color32::WHITE, name);
            }
        });
    });
}
