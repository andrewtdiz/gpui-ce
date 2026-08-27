//! Deterministic offscreen verification for a representative `gpui-component` suite.
//!
//! The host owns WGPU from adapter selection through command submission and readback. The same
//! retained GPUI view is rendered once with the light theme and once with the dark theme, then the
//! two frames are joined into `target/gpui-embed/headless-component-suite.png` for local review.

use gpui::{
    AppContext, Bounds, Context, DevicePixels, Entity, IntoElement, Pixels, Render, Window,
    block_on, div, prelude::*, px, size,
};
use gpui_component::{
    ActiveTheme as _, Disableable as _, Icon, IconName, IndexPath, Root, Sizable as _,
    StyledExt as _, Theme, ThemeMode,
    badge::Badge,
    button::{Button, ButtonVariants as _},
    checkbox::Checkbox,
    label::Label,
    progress::Progress,
    select::{SearchableVec, Select, SelectState},
    separator::Separator,
    switch::Switch,
};
use gpui_embed::{EmbeddedConfig, EmbeddedGpui, HostGpu, OffscreenTarget, WindowMetrics};
use gpui_wgpu::{WgpuSceneRenderer, wgpu};
use image::{ColorType, ImageFormat};
use std::{cell::RefCell, collections::HashSet, fs, path::PathBuf, rc::Rc, sync::Arc};

const WIDTH: u32 = 640;
const HEIGHT: u32 = 520;

struct ComponentSuite {
    select: Entity<SelectState<SearchableVec<String>>>,
    icon_bounds: Rc<RefCell<Option<Bounds<Pixels>>>>,
    select_bounds: Rc<RefCell<Option<Bounds<Pixels>>>>,
}

impl Render for ComponentSuite {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let mode = if cx.theme().is_dark() {
            "Dark theme"
        } else {
            "Light theme"
        };
        let icon_bounds = self.icon_bounds.clone();
        let select_bounds = self.select_bounds.clone();

        div()
            .size_full()
            .flex()
            .items_center()
            .justify_center()
            .bg(cx.theme().background)
            .text_color(cx.theme().foreground)
            .child(
                div()
                    .w(px(560.))
                    .p_7()
                    .rounded_xl()
                    .border_1()
                    .border_color(cx.theme().border)
                    .bg(cx.theme().popover)
                    .shadow_lg()
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .justify_between()
                            .child(
                                div()
                                    .child(
                                        div()
                                            .text_2xl()
                                            .font_semibold()
                                            .child("gpui-component"),
                                    )
                                    .child(
                                        div()
                                            .mt_1()
                                            .text_sm()
                                            .text_color(cx.theme().muted_foreground)
                                            .child("EmbeddedGpui · host-owned WGPU · pixel readback"),
                                    ),
                            )
                            .child(
                                Badge::new().count(7).child(
                                    div()
                                        .px_3()
                                        .py_1()
                                        .rounded_full()
                                        .bg(cx.theme().secondary)
                                        .text_sm()
                                        .child(mode),
                                    ),
                            ),
                    )
                    .child(
                        div()
                            .mt_5()
                            .flex()
                            .items_center()
                            .gap_3()
                            .child(
                                div()
                                    .size(px(36.))
                                    .flex()
                                    .items_center()
                                    .justify_center()
                                    .rounded_md()
                                    .bg(cx.theme().secondary)
                                    .on_children_prepainted(move |bounds, _, _| {
                                        *icon_bounds.borrow_mut() = bounds.first().copied();
                                    })
                                    .child(
                                        Icon::new(IconName::Plus)
                                            .size_5()
                                            .text_color(cx.theme().primary),
                                    ),
                            )
                            .child(
                                div()
                                    .text_sm()
                                    .text_color(cx.theme().muted_foreground)
                                    .child("Asset-backed IconName + Select caret"),
                            )
                            .child(
                                div()
                                    .w(px(180.))
                                    .on_children_prepainted(move |bounds, _, _| {
                                        *select_bounds.borrow_mut() = bounds.first().copied();
                                    })
                                    .child(
                                        Select::new(&self.select)
                                            .small()
                                            .placeholder("Choose a mode"),
                                    ),
                            ),
                    )
                    .child(Separator::horizontal().my_5())
                    .child(
                        Label::new("Solid-core component suite")
                            .secondary("real upstream controls")
                            .highlights("component"),
                    )
                    .child(
                        div()
                            .mt_5()
                            .flex()
                            .gap_3()
                            .child(Button::new("primary").primary().label("Primary action"))
                            .child(Button::new("success").success().label("Success"))
                            .child(Button::new("danger").danger().label("Danger"))
                            .child(
                                Button::new("disabled")
                                    .secondary()
                                    .label("Disabled")
                                    .disabled(true),
                            ),
                    )
                    .child(
                        div()
                            .mt_6()
                            .flex()
                            .gap_8()
                            .child(Checkbox::new("checked").checked(true).label("Checked"))
                            .child(Checkbox::new("unchecked").label("Unchecked"))
                            .child(Switch::new("switch-on").checked(true).label("Enabled"))
                            .child(Switch::new("switch-off").label("Disabled")),
                    )
                    .child(
                        div()
                            .mt_6()
                            .child(
                                div()
                                    .mb_2()
                                    .flex()
                                    .justify_between()
                                    .text_sm()
                                    .child("Deterministic render progress")
                                    .child("72%"),
                            )
                            .child(Progress::new("render-progress").value(72.)),
                    )
                    .child(
                        div()
                            .mt_6()
                            .p_4()
                            .rounded_lg()
                            .border_1()
                            .border_color(cx.theme().border)
                            .bg(cx.theme().background)
                            .child(
                                div()
                                    .font_semibold()
                                    .child("Headless verification is active"),
                            )
                            .child(
                                div()
                                    .mt_1()
                                    .text_sm()
                                    .text_color(cx.theme().muted_foreground)
                                    .child("Shapes, glyphs, component states, and theme colors reached the offscreen texture."),
                            ),
                    ),
            )
    }
}

fn create_host_gpu() -> gpui::Result<HostGpu> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        flags: wgpu::InstanceFlags::default(),
        backend_options: wgpu::BackendOptions::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        display: None,
    });
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .map_err(|error| std::io::Error::other(format!("failed to request adapter: {error}")))?;
    let requirements = WgpuSceneRenderer::requirements(&adapter);
    let (device, queue) = block_on(
        adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("gpui_embed_headless_component_suite"),
            required_features: requirements.features,
            required_limits: wgpu::Limits::downlevel_defaults()
                .using_resolution(adapter.limits())
                .using_alignment(adapter.limits()),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        }),
    )
    .map_err(|error| std::io::Error::other(format!("failed to request device: {error}")))?;

    Ok(HostGpu::new(
        Arc::new(adapter),
        Arc::new(device),
        Arc::new(queue),
        wgpu::TextureFormat::Rgba8UnormSrgb,
    ))
}

fn render_frame(ui: &EmbeddedGpui, gpu: &HostGpu) -> gpui::Result<Vec<u8>> {
    let _ = ui.poll();
    let frame = ui.prepare_frame()?;
    assert!(
        frame.scene_generation > 0,
        "GPUI produced no retained scene"
    );

    let target = OffscreenTarget::new(
        gpu,
        size(DevicePixels(WIDTH as i32), DevicePixels(HEIGHT as i32)),
    );
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpui_embed_headless_component_encoder"),
        });
    let stats = ui.encode_offscreen(&target, &mut encoder)?;
    assert!(
        stats.instance_bytes > 0,
        "the component suite encoded no GPU instances"
    );
    gpu.queue.submit(Some(encoder.finish()));
    assert!(ui.mark_presented(frame.scene_generation)?);

    readback_rgba(gpu, &target)
}

fn readback_rgba(gpu: &HostGpu, target: &OffscreenTarget) -> gpui::Result<Vec<u8>> {
    let unpadded_bytes_per_row = WIDTH * 4;
    let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(256) * 256;
    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpui_embed_headless_component_readback"),
        size: padded_bytes_per_row as u64 * HEIGHT as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpui_embed_headless_component_readback_encoder"),
        });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: target.texture(),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(HEIGHT),
            },
        },
        wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit(Some(encoder.finish()));

    let slice = readback.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    gpu.device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .map_err(|error| std::io::Error::other(format!("GPU readback poll failed: {error}")))?;
    receiver
        .recv()
        .map_err(|error| std::io::Error::other(format!("GPU readback callback failed: {error}")))?
        .map_err(|error| std::io::Error::other(format!("GPU readback mapping failed: {error}")))?;

    let mapped = slice.get_mapped_range()?;
    let mut rgba = Vec::with_capacity((WIDTH * HEIGHT * 4) as usize);
    for row in mapped.chunks_exact(padded_bytes_per_row as usize) {
        rgba.extend_from_slice(&row[..unpadded_bytes_per_row as usize]);
    }
    drop(mapped);
    readback.unmap();
    Ok(rgba)
}

fn assert_nontrivial(name: &str, rgba: &[u8]) {
    let pixels = rgba.as_chunks::<4>().0;
    let total = pixels.len();
    let opaque = pixels.iter().filter(|pixel| pixel[3] > 250).count();
    let dark = pixels
        .iter()
        .filter(|pixel| pixel[0] < 140 && pixel[1] < 140 && pixel[2] < 140)
        .count();
    let bright = pixels
        .iter()
        .filter(|pixel| pixel[0] > 200 && pixel[1] > 200 && pixel[2] > 200)
        .count();
    let colorful = pixels
        .iter()
        .filter(|pixel| {
            let min = pixel[..3].iter().copied().min().unwrap();
            let max = pixel[..3].iter().copied().max().unwrap();
            max - min > 35
        })
        .count();
    let quantized_colors = pixels
        .iter()
        .map(|pixel| [pixel[0] / 16, pixel[1] / 16, pixel[2] / 16])
        .collect::<HashSet<_>>();

    println!(
        "{name}: opaque={opaque}/{total}, dark={dark}, bright={bright}, colorful={colorful}, quantized_colors={}",
        quantized_colors.len()
    );
    assert!(opaque > total * 95 / 100, "{name} frame was mostly empty");
    assert!(dark > 100, "{name} frame had no dark detail or text");
    assert!(bright > 100, "{name} frame had no bright detail or text");
    assert!(
        colorful > 500,
        "{name} frame did not contain component accent colors"
    );
    assert!(
        quantized_colors.len() > 24,
        "{name} frame had too little visual diversity"
    );
}

fn assert_asset_bounds(
    name: &str,
    rgba: &[u8],
    icon_bounds: Bounds<Pixels>,
    select_bounds: Bounds<Pixels>,
) {
    let icon_x = icon_bounds.origin.x.as_f32().round().max(0.) as u32;
    let icon_y = icon_bounds.origin.y.as_f32().round().max(0.) as u32;
    let icon_width = icon_bounds.size.width.as_f32().round().max(1.) as u32;
    let icon_height = icon_bounds.size.height.as_f32().round().max(1.) as u32;
    let select_x = select_bounds.origin.x.as_f32().round().max(0.) as u32;
    let select_y = select_bounds.origin.y.as_f32().round().max(0.) as u32;
    let select_width = select_bounds.size.width.as_f32().round().max(1.) as u32;
    let select_height = select_bounds.size.height.as_f32().round().max(1.) as u32;

    assert!(
        icon_x < WIDTH && icon_y < HEIGHT,
        "{name} icon bounds are outside the frame"
    );
    assert!(
        select_x < WIDTH && select_y < HEIGHT,
        "{name} select bounds are outside the frame"
    );

    let mut icon_colors = HashSet::new();
    let mut icon_luminance = (u8::MAX, 0u8);
    for y in icon_y..(icon_y + icon_height).min(HEIGHT) {
        for x in icon_x..(icon_x + icon_width).min(WIDTH) {
            let index = ((y * WIDTH + x) * 4) as usize;
            let pixel = &rgba[index..index + 4];
            if pixel[3] > 200 {
                icon_colors.insert([pixel[0] / 16, pixel[1] / 16, pixel[2] / 16]);
                let luminance = ((u16::from(pixel[0]) * 54
                    + u16::from(pixel[1]) * 183
                    + u16::from(pixel[2]) * 19)
                    / 256) as u8;
                icon_luminance.0 = icon_luminance.0.min(luminance);
                icon_luminance.1 = icon_luminance.1.max(luminance);
            }
        }
    }

    let mut select_colors = HashSet::new();
    for y in select_y..(select_y + select_height).min(HEIGHT) {
        for x in select_x..(select_x + select_width).min(WIDTH) {
            let index = ((y * WIDTH + x) * 4) as usize;
            let pixel = &rgba[index..index + 4];
            if pixel[3] > 200 {
                select_colors.insert([pixel[0] / 16, pixel[1] / 16, pixel[2] / 16]);
            }
        }
    }

    println!(
        "{name}: icon_bounds={icon_bounds:?}, select_bounds={select_bounds:?}, icon_colors={}, icon_luminance={:?}, select_colors={}",
        icon_colors.len(),
        icon_luminance,
        select_colors.len()
    );
    assert!(
        icon_colors.len() > 1 && icon_luminance.1.saturating_sub(icon_luminance.0) > 20,
        "{name} had no visible IconName pixels"
    );
    assert!(
        select_colors.len() > 2,
        "{name} Select trigger/caret was not visible"
    );
}

fn stitch_frames(light: &[u8], dark: &[u8]) -> Vec<u8> {
    let row_bytes = (WIDTH * 4) as usize;
    let mut joined = Vec::with_capacity(light.len() + dark.len());
    for (light_row, dark_row) in light
        .chunks_exact(row_bytes)
        .zip(dark.chunks_exact(row_bytes))
    {
        joined.extend_from_slice(light_row);
        joined.extend_from_slice(dark_row);
    }
    joined
}

fn artifact_path() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|path| path.parent())
        .expect("gpui_embed must live under the workspace crates directory");
    workspace_root
        .join("target")
        .join("gpui-embed")
        .join("headless-component-suite.png")
}

fn main() -> gpui::Result<()> {
    let gpu = create_host_gpu()?;
    let metrics = WindowMetrics::new(size(px(WIDTH as f32), px(HEIGHT as f32)), 1.0);
    let icon_bounds = Rc::new(RefCell::new(None));
    let select_bounds = Rc::new(RefCell::new(None));
    let icon_bounds_for_root = icon_bounds.clone();
    let select_bounds_for_root = select_bounds.clone();
    let ui = EmbeddedGpui::new_with_root(
        EmbeddedConfig::new(gpu.clone())
            .with_metrics(metrics)
            .with_assets(gpui_component_assets::Assets),
        |window, cx| {
            // Stable verification images should not depend on wall-clock animation progress.
            cx.set_reduce_motion(true);
            gpui_component::init(cx);
            Theme::change(ThemeMode::Light, Some(window), cx);
            let select = cx.new(|cx| {
                SelectState::new(
                    SearchableVec::new(["Play", "Edit", "Preview"].map(str::to_owned)),
                    Some(IndexPath::default().row(0)),
                    window,
                    cx,
                )
            });
            let suite = cx.new(|_| ComponentSuite {
                select,
                icon_bounds: icon_bounds_for_root,
                select_bounds: select_bounds_for_root,
            });
            cx.new(|cx| Root::new(suite, window, cx).bordered(false))
        },
    )?;

    let _warmup = render_frame(&ui, &gpu)?;
    let light = render_frame(&ui, &gpu)?;
    let light_icon_bounds =
        (*icon_bounds.borrow()).expect("light IconName bounds were not recorded");
    let light_select_bounds =
        (*select_bounds.borrow()).expect("light Select bounds were not recorded");

    let window = ui.window_handle();
    ui.update(|cx| {
        window.update(cx, |_, window, cx| {
            Theme::change(ThemeMode::Dark, Some(window), cx);
        })
    })?;
    let dark = render_frame(&ui, &gpu)?;
    let dark_icon_bounds = (*icon_bounds.borrow()).expect("dark IconName bounds were not recorded");
    let dark_select_bounds =
        (*select_bounds.borrow()).expect("dark Select bounds were not recorded");

    // Persist successful GPU readbacks before analyzing them so a threshold failure still leaves
    // an artifact that explains what the renderer produced.
    let joined = stitch_frames(&light, &dark);
    let artifact = artifact_path();
    fs::create_dir_all(artifact.parent().expect("artifact has a parent directory"))?;
    image::save_buffer_with_format(
        &artifact,
        &joined,
        WIDTH * 2,
        HEIGHT,
        ColorType::Rgba8,
        ImageFormat::Png,
    )
    .map_err(|error| std::io::Error::other(format!("failed to write PNG: {error}")))?;

    assert_asset_bounds("light", &light, light_icon_bounds, light_select_bounds);
    assert_asset_bounds("dark", &dark, dark_icon_bounds, dark_select_bounds);
    assert_nontrivial("light", &light);
    assert_nontrivial("dark", &dark);

    let changed_pixels = light
        .as_chunks::<4>()
        .0
        .iter()
        .zip(dark.as_chunks::<4>().0)
        .filter(|(light, dark)| {
            light[..3]
                .iter()
                .zip(&dark[..3])
                .any(|(light, dark)| light.abs_diff(*dark) > 20)
        })
        .count();
    assert!(
        changed_pixels > (WIDTH * HEIGHT / 3) as usize,
        "light and dark component frames were unexpectedly similar"
    );
    println!(
        "theme_delta: changed_pixels={changed_pixels}/{}",
        WIDTH * HEIGHT
    );

    println!(
        "verified light and dark gpui-component frames; wrote {}",
        artifact.display()
    );
    Ok(())
}
