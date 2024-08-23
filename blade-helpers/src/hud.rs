pub trait ExposeHud {
    fn populate_hud(&mut self, ui: &mut egui::Ui);
}

impl ExposeHud for blade_render::RayConfig {
    fn populate_hud(&mut self, ui: &mut egui::Ui) {
        ui.add(
            egui::Slider::new(&mut self.num_environment_samples, 1..=100u32)
                .text("Num env samples")
                .logarithmic(true),
        );
        ui.checkbox(
            &mut self.environment_importance_sampling,
            "Env importance sampling",
        );
        ui.checkbox(&mut self.temporal_tap, "Temporal tap");
        ui.add(
            egui::widgets::Slider::new(&mut self.temporal_history, 0..=50).text("Temporal history"),
        );
        ui.add(egui::widgets::Slider::new(&mut self.spatial_taps, 0..=10).text("Spatial taps"));
        ui.add(
            egui::widgets::Slider::new(&mut self.spatial_tap_history, 0..=50)
                .text("Spatial tap history"),
        );
        ui.add(egui::widgets::Slider::new(&mut self.group_mixer, 1..=10).text("Group mixer"));
        ui.add(
            egui::widgets::Slider::new(&mut self.spatial_min_distance, 1..=10)
                .text("Spatial minimum distance (px)"),
        );
        ui.add(
            egui::widgets::Slider::new(&mut self.t_start, 0.001..=0.5)
                .text("T min")
                .logarithmic(true),
        );
    }
}

impl ExposeHud for blade_render::DenoiserConfig {
    fn populate_hud(&mut self, ui: &mut egui::Ui) {
        ui.add(egui::Slider::new(&mut self.temporal_weight, 0.0..=1.0f32).text("Temporal weight"));
        ui.add(egui::Slider::new(&mut self.num_passes, 0..=5u32).text("A-trous passes"));
    }
}

impl ExposeHud for blade_render::PostProcConfig {
    fn populate_hud(&mut self, ui: &mut egui::Ui) {
        ui.add(
            egui::Slider::new(&mut self.average_luminocity, 0.1f32..=1_000f32)
                .text("Average luminocity")
                .logarithmic(true),
        );
        ui.add(
            egui::Slider::new(&mut self.exposure_key_value, 0.01f32..=10f32)
                .text("Key value")
                .logarithmic(true),
        );
        ui.add(egui::Slider::new(&mut self.white_level, 0.1f32..=2f32).text("White level"));
    }
}

impl ExposeHud for blade_render::DebugConfig {
    fn populate_hud(&mut self, ui: &mut egui::Ui) {
        use strum::IntoEnumIterator as _;

        egui::ComboBox::from_label("View mode")
            .selected_text(format!("{:?}", self.view_mode))
            .show_ui(ui, |ui| {
                for value in blade_render::DebugMode::iter() {
                    ui.selectable_value(&mut self.view_mode, value, format!("{value:?}"));
                }
            });

        ui.label("Draw debug:");
        for (name, bit) in blade_render::DebugDrawFlags::all().iter_names() {
            let mut enabled = self.draw_flags.contains(bit);
            ui.checkbox(&mut enabled, name);
            self.draw_flags.set(bit, enabled);
        }
        ui.label("Ignore textures:");
        for (name, bit) in blade_render::DebugTextureFlags::all().iter_names() {
            let mut enabled = self.texture_flags.contains(bit);
            ui.checkbox(&mut enabled, name);
            self.texture_flags.set(bit, enabled);
        }
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
