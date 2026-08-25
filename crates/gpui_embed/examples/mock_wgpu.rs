//! Offscreen host integration proof for `gpui_embed`.
//!
//! The host creates the instance, adapter, device, command encoder, target, submission, and
//! readback buffer. A small GPU triangle is rendered into a layout-sized viewport texture first,
//! then GPUI's retained UI scene samples that texture while encoding into the host target.

use gpui::{
    AppContext, Bounds, Context, DevicePixels, ExternalSurfaceId, ExternalSurfaceRegistry,
    IntoElement, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, Pixels, PlatformInput,
    Render, Window, block_on, div, external_surface, point, prelude::*, px, rgb, size,
};
use gpui_embed::{EmbeddedConfig, EmbeddedGpui, HostGpu, OffscreenTarget, WindowMetrics};
use gpui_wgpu::{WgpuExternalSurface, WgpuSceneRenderer, wgpu};
use std::{cell::RefCell, rc::Rc, sync::Arc};

const WIDTH: u32 = 320;
const HEIGHT: u32 = 200;

struct MockUi {
    clicked: bool,
    external_surface_id: ExternalSurfaceId,
    viewport_bounds: Rc<RefCell<Option<Bounds<Pixels>>>>,
}

impl Render for MockUi {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let button_label = if self.clicked { "Clicked" } else { "Click me" };
        let viewport_bounds = self.viewport_bounds.clone();
        div()
            .size_full()
            .flex()
            .flex_col()
            .items_center()
            .justify_center()
            .gap_3()
            .text_color(rgb(0xf8fafc))
            .child(
                div()
                    .w(px(260.))
                    .p_5()
                    .rounded_lg()
                    .bg(rgb(0x172554))
                    .child(div().text_xl().child("GPUI embedded UI"))
                    .child(
                        div()
                            .mt_2()
                            .text_sm()
                            .text_color(rgb(0xbfdbfe))
                            .child("Retained layout and text over a host GPU scene"),
                    )
                    .child(
                        div()
                            .id("mock-button")
                            .mt_4()
                            .px_4()
                            .py_2()
                            .rounded_md()
                            .cursor_pointer()
                            .on_mouse_down(MouseButton::Left, |_, _, cx| {
                                cx.stop_propagation();
                            })
                            .bg(rgb(0x2563eb))
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.clicked = true;
                                cx.notify();
                            }))
                            .child(button_label),
                    )
                    .child(
                        div()
                            .mt_3()
                            .w(px(120.))
                            .h(px(24.))
                            .overflow_hidden()
                            .on_children_prepainted(move |children, _, _| {
                                *viewport_bounds.borrow_mut() = children.into_iter().next();
                            })
                            .child(
                                external_surface(self.external_surface_id)
                                    .w(px(180.))
                                    .h(px(24.)),
                            ),
                    ),
            )
    }
}

fn create_gpu_scene_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mock_gpu_scene_shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
            r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-0.78, -0.55),
        vec2<f32>( 0.00,  0.78),
        vec2<f32>( 0.78, -0.55),
    );
    var colors = array<vec3<f32>, 3>(
        vec3<f32>(0.14, 0.65, 0.95),
        vec3<f32>(0.35, 0.95, 0.62),
        vec3<f32>(0.96, 0.52, 0.25),
    );
    var output: VertexOutput;
    output.position = vec4<f32>(positions[index], 0.0, 1.0);
    output.color = colors[index];
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(input.color, 1.0);
}
"#,
        )),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("mock_gpu_scene_pipeline_layout"),
        bind_group_layouts: &[],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("mock_gpu_scene_pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn encode_engine_viewport(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::RenderPipeline,
    target: &OffscreenTarget,
    clear: wgpu::Color,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("mock_engine_viewport_pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target.view(),
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(clear),
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: None,
        ..Default::default()
    });
    pass.set_pipeline(pipeline);
    pass.draw(0..3, 0..1);
}

fn readback_target(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    target: &OffscreenTarget,
) -> gpui::Result<(Vec<u8>, u32)> {
    let size = target.size();
    let width = size.width.0 as u32;
    let height = size.height.0 as u32;
    let unpadded_bytes_per_row = width * 4;
    let bytes_per_row = unpadded_bytes_per_row.div_ceil(256) * 256;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mock_wgpu_readback"),
        size: bytes_per_row as u64 * height as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("mock_wgpu_readback_encoder"),
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
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    let slice = readback.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .map_err(|error| std::io::Error::other(format!("GPU readback poll failed: {error}")))?;
    receiver
        .recv()
        .map_err(|error| std::io::Error::other(format!("GPU readback callback failed: {error}")))?
        .map_err(|error| std::io::Error::other(format!("GPU readback mapping failed: {error}")))?;
    let mapped = slice.get_mapped_range().to_vec();
    readback.unmap();
    Ok((mapped, bytes_per_row))
}

fn pixel(bytes: &[u8], bytes_per_row: u32, x: u32, y: u32) -> [u8; 4] {
    let offset = y as usize * bytes_per_row as usize + x as usize * 4;
    bytes[offset..offset + 4]
        .try_into()
        .expect("RGBA pixel is four bytes")
}

fn main() -> gpui::Result<()> {
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
    }))
    .map_err(|error| std::io::Error::other(format!("failed to request adapter: {error}")))?;
    let requirements = WgpuSceneRenderer::requirements(&adapter);
    let (device, queue) = block_on(
        adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("gpui_embed_mock_device"),
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

    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let host_gpu = HostGpu::new(Arc::new(adapter), Arc::new(device), Arc::new(queue), format);
    let device = host_gpu.device.clone();
    let queue = host_gpu.queue.clone();
    let metrics = WindowMetrics::new(size(px(WIDTH as f32), px(HEIGHT as f32)), 1.0);
    let placeholder_target =
        OffscreenTarget::new(&host_gpu, size(DevicePixels(1), DevicePixels(1)));
    let mut external_registry = ExternalSurfaceRegistry::default();
    let external_surface_id = external_registry.register(WgpuExternalSurface {
        view: Arc::new(placeholder_target.view().clone()),
    });
    let config = EmbeddedConfig::new(host_gpu.clone()).with_metrics(metrics);
    let root_entity = Rc::new(RefCell::new(None));
    let root_entity_for_builder = root_entity.clone();
    let viewport_bounds = Rc::new(RefCell::new(None));
    let viewport_bounds_for_builder = viewport_bounds.clone();
    let ui = EmbeddedGpui::new_with_root(config, move |_, cx| {
        let entity = cx.new(|_| MockUi {
            clicked: false,
            external_surface_id,
            viewport_bounds: viewport_bounds_for_builder,
        });
        *root_entity_for_builder.borrow_mut() = Some(entity.clone());
        entity
    })?;

    let gpu_scene_pipeline = create_gpu_scene_pipeline(device.as_ref(), format);
    let _ = ui.poll();
    let frame_status = ui.prepare_frame()?;
    assert!(frame_status.scene_generation > 0);
    let rendered_scene = ui
        .rendered_scene()
        .expect("the embedded GPUI window did not produce a retained scene");
    let viewport_logical_bounds = viewport_bounds
        .borrow()
        .expect("GPUI did not export the viewport child bounds");
    let viewport_device_bounds =
        viewport_logical_bounds.to_device_pixels(rendered_scene.metrics.scale_factor);
    assert!(
        viewport_device_bounds.size.width.0 > 0 && viewport_device_bounds.size.height.0 > 0,
        "GPUI exported an empty viewport"
    );

    // GPUI layout is the source of truth for the host viewport. Render the engine scene into a
    // texture sized from those device-pixel bounds, then replace the stable registry ID.
    let mut viewport_target = OffscreenTarget::new(&host_gpu, viewport_device_bounds.size);
    assert!(external_registry.replace(
        external_surface_id,
        WgpuExternalSurface {
            view: Arc::new(viewport_target.view().clone()),
        },
    ));
    assert_eq!(external_registry.generation(external_surface_id), Some(2));

    let target_size = size(DevicePixels(WIDTH as i32), DevicePixels(HEIGHT as i32));
    let target = OffscreenTarget::new(&host_gpu, target_size);

    // Both failure modes are checked before the renderer writes atlas/uniform data or opens a
    // render pass. A missing resource is never silently skipped.
    let missing_registry_target = OffscreenTarget::new(&host_gpu, target_size);
    let mut missing_registry_encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mock_missing_registry_encoder"),
        });
    let missing_registry_error = ui
        .encode(
            missing_registry_target.target(wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)),
            &mut missing_registry_encoder,
        )
        .expect_err("external surfaces must require a registry");
    assert!(missing_registry_error.to_string().contains("registry"));

    let missing_id_target = OffscreenTarget::new(&host_gpu, target_size);
    let mut missing_id_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("mock_missing_id_encoder"),
    });
    let missing_id_registry = ExternalSurfaceRegistry::<WgpuExternalSurface>::default();
    let missing_id_error = ui
        .encode_with_external_surfaces(
            missing_id_target.target(wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)),
            &mut missing_id_encoder,
            &missing_id_registry,
        )
        .expect_err("missing external IDs must fail encoding");
    assert!(
        missing_id_error
            .to_string()
            .contains(&format!("ID {}", external_surface_id.value()))
    );

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gpui_embed_mock_encoder"),
    });
    encode_engine_viewport(
        &mut encoder,
        &gpu_scene_pipeline,
        &viewport_target,
        wgpu::Color {
            r: 0.04,
            g: 0.72,
            b: 0.10,
            a: 1.0,
        },
    );
    let stats = ui.encode_with_external_surfaces(
        target.target(wgpu::LoadOp::Clear(wgpu::Color {
            r: 0.015,
            g: 0.027,
            b: 0.09,
            a: 1.0,
        })),
        &mut encoder,
        &external_registry,
    )?;
    assert!(
        stats.instance_bytes > 0,
        "the GPUI scene encoded no GPU instances"
    );
    queue.submit(Some(encoder.finish()));
    assert!(ui.mark_presented(frame_status.scene_generation)?);
    assert!(!ui.mark_presented(frame_status.scene_generation - 1)?);

    let (mapped, bytes_per_row) = readback_target(device.as_ref(), queue.as_ref(), &target)?;
    let unpadded_bytes_per_row = WIDTH * 4;
    let non_background_pixels = mapped
        .chunks(bytes_per_row as usize)
        .flat_map(|row| row[..unpadded_bytes_per_row as usize].chunks(4))
        .filter(|pixel| pixel[0] > 20 || pixel[1] > 20 || pixel[2] > 20)
        .count();
    let bright_text_pixels = mapped
        .chunks(bytes_per_row as usize)
        .flat_map(|row| row[..unpadded_bytes_per_row as usize].chunks(4))
        .filter(|pixel| pixel[0] > 180 && pixel[1] > 180 && pixel[2] > 180)
        .count();
    let viewport_readback = readback_target(device.as_ref(), queue.as_ref(), &viewport_target)?;
    let viewport_background = pixel(&viewport_readback.0, viewport_readback.1, 0, 0);
    let triangle_pixels = viewport_readback
        .0
        .chunks(viewport_readback.1 as usize)
        .flat_map(|row| row.chunks_exact(4))
        .filter(|pixel| {
            pixel
                .iter()
                .zip(viewport_background)
                .any(|(value, background)| value.abs_diff(background) > 10)
        })
        .count();
    let child_origin_x = viewport_device_bounds.origin.x.0.max(0) as u32;
    let child_origin_y = viewport_device_bounds.origin.y.0.max(0) as u32;
    let external_green_pixels = mapped
        .chunks(bytes_per_row as usize)
        .flat_map(|row| row[..unpadded_bytes_per_row as usize].chunks_exact(4))
        .filter(|pixel| pixel[1] > 150 && pixel[0] < 180 && pixel[2] < 180)
        .count();
    let inside_viewport = pixel(
        &mapped,
        bytes_per_row,
        (child_origin_x + 60).min(WIDTH - 1),
        (child_origin_y + 12).min(HEIGHT - 1),
    );
    let clipped_viewport = pixel(
        &mapped,
        bytes_per_row,
        (child_origin_x + 150).min(WIDTH - 1),
        (child_origin_y + 12).min(HEIGHT - 1),
    );
    assert!(
        non_background_pixels > 100,
        "GPU scene and GPUI overlay did not produce readable pixels"
    );
    assert!(
        bright_text_pixels > 10,
        "GPUI text did not reach the host target"
    );
    assert!(
        triangle_pixels > 20,
        "the host GPU triangle did not render into the engine viewport texture"
    );
    assert!(
        external_green_pixels > 100,
        "the registered external viewport did not reach GPUI scene order"
    );
    assert!(
        external_green_pixels < 180 * 24,
        "the external viewport was not clipped by its GPUI layout parent"
    );
    assert!(
        inside_viewport[1] > inside_viewport[0] + 20
            || inside_viewport[2] > inside_viewport[0] + 20,
        "the engine viewport did not reach the visible GPUI surface"
    );
    assert_ne!(
        inside_viewport, clipped_viewport,
        "the GPUI parent did not clip the external viewport"
    );

    // Normalized host events go to GPUI first. An unhandled event in the exported viewport is
    // then translated with the same logical bounds the host used to size the engine texture.
    let mut engine_events = Vec::new();
    let engine_position = point(
        viewport_logical_bounds.origin.x + px(8.),
        viewport_logical_bounds.origin.y + px(8.),
    );
    let engine_input = PlatformInput::MouseMove(MouseMoveEvent {
        position: engine_position,
        pressed_button: None,
        modifiers: Default::default(),
    });
    let engine_outcome = ui.dispatch_input(engine_input)?;
    if engine_outcome.propagate
        && !engine_outcome.pointer_capture
        && viewport_logical_bounds.contains(&engine_position)
    {
        engine_events.push(point(
            engine_position.x - viewport_logical_bounds.origin.x,
            engine_position.y - viewport_logical_bounds.origin.y,
        ));
    }
    assert_eq!(engine_events.len(), 1);
    assert_eq!(engine_events[0], point(px(8.), px(8.)));

    // The retained scene can be encoded into a fresh host target without rebuilding layout or
    // paint. This is the normal path when the host acquires a new swapchain image.
    let _ = ui.poll();
    let unchanged_status = ui.prepare_frame()?;
    assert_eq!(
        unchanged_status.scene_generation,
        frame_status.scene_generation
    );
    assert!(!unchanged_status.scene_changed);
    let unchanged_target = OffscreenTarget::new(&host_gpu, target_size);
    let mut unchanged_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gpui_embed_mock_unchanged_encoder"),
    });
    let unchanged_stats = ui.encode_with_external_surfaces(
        unchanged_target.target(wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)),
        &mut unchanged_encoder,
        &external_registry,
    )?;
    assert!(unchanged_stats.instance_bytes > 0);
    queue.submit(Some(unchanged_encoder.finish()));
    ui.mark_presented(unchanged_status.scene_generation)?;

    // Drive a real GPUI click through the host input seam. The small grid keeps the proof
    // independent of the exact font metrics used by the host while still exercising hit testing,
    // capture, and the button's `on_click` listener.
    let mut clicked = false;
    let mut button_consumed = false;
    'click_grid: for y in (40..=180).step_by(10) {
        for x in (40..=280).step_by(10) {
            let position = point(px(x as f32), px(y as f32));
            let down_outcome = ui.dispatch_input(PlatformInput::MouseDown(MouseDownEvent {
                button: MouseButton::Left,
                position,
                modifiers: Default::default(),
                click_count: 1,
                first_mouse: false,
            }))?;
            let up_outcome = ui.dispatch_input(PlatformInput::MouseUp(MouseUpEvent {
                button: MouseButton::Left,
                position,
                modifiers: Default::default(),
                click_count: 1,
            }))?;
            button_consumed |= !down_outcome.propagate || !up_outcome.propagate;
            clicked = ui.update(|cx| {
                cx.read_entity(
                    root_entity.borrow().as_ref().expect("mock root entity"),
                    |mock, _| mock.clicked,
                )
            });
            if clicked {
                break 'click_grid;
            }
        }
    }
    assert!(clicked, "host mouse input did not activate the GPUI button");
    assert!(
        button_consumed,
        "the GPUI button did not consume its mouse event"
    );
    let _ = ui.poll();
    let clicked_status = ui.prepare_frame()?;
    assert!(clicked_status.scene_changed);
    assert!(clicked_status.scene_generation > unchanged_status.scene_generation);
    let clicked_target = OffscreenTarget::new(&host_gpu, target_size);
    let mut clicked_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gpui_embed_mock_clicked_encoder"),
    });
    ui.encode_with_external_surfaces(
        clicked_target.target(wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)),
        &mut clicked_encoder,
        &external_registry,
    )?;
    queue.submit(Some(clicked_encoder.finish()));
    ui.mark_presented(clicked_status.scene_generation)?;

    // Exercise host-owned resize handling. The host skips encoding while minimized, then
    // restores a positive target; every positive cycle submits before the next encode.
    for (width, height) in [(96_u32, 64_u32), (256, 128), (WIDTH, HEIGHT)] {
        ui.set_window_metrics(WindowMetrics::new(
            size(px(width as f32), px(height as f32)),
            1.0,
        ));
        let _ = ui.poll();
        let resize_status = ui.prepare_frame()?;
        let resize_scene = ui.rendered_scene().expect("resize did not retain a scene");
        let resize_logical_bounds = viewport_bounds
            .borrow()
            .expect("resize did not export viewport bounds");
        let resize_device_bounds =
            resize_logical_bounds.to_device_pixels(resize_scene.metrics.scale_factor);
        if resize_device_bounds.size != viewport_target.size() {
            viewport_target = OffscreenTarget::new(&host_gpu, resize_device_bounds.size);
            assert!(external_registry.replace(
                external_surface_id,
                WgpuExternalSurface {
                    view: Arc::new(viewport_target.view().clone()),
                },
            ));
        }
        let resize_target = OffscreenTarget::new(
            &host_gpu,
            size(DevicePixels(width as i32), DevicePixels(height as i32)),
        );
        let mut resize_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mock_resize_encoder"),
        });
        encode_engine_viewport(
            &mut resize_encoder,
            &gpu_scene_pipeline,
            &viewport_target,
            wgpu::Color {
                r: 0.04,
                g: 0.72,
                b: 0.10,
                a: 1.0,
            },
        );
        ui.encode_with_external_surfaces(
            resize_target.target(wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)),
            &mut resize_encoder,
            &external_registry,
        )?;
        queue.submit(Some(resize_encoder.finish()));
        assert!(ui.mark_presented(resize_status.scene_generation)?);
    }

    let minimized_metrics = WindowMetrics::new(size(px(0.), px(0.)), 1.0);
    ui.set_window_metrics(minimized_metrics);
    let _ = ui.poll();
    let minimized_status = ui.prepare_frame()?;
    assert_eq!(
        ui.rendered_scene()
            .expect("minimized frame did not retain scene")
            .metrics
            .bounds
            .size,
        minimized_metrics.bounds.size
    );
    assert!(minimized_status.scene_generation > 0);

    // Recreate the positive host target after minimization and exercise the recovery seam. The
    // renderer keeps the atlas Arc, while the host recreates its viewport/final targets and
    // replaces the external view under the same stable ID.
    ui.set_window_metrics(metrics);
    let _ = ui.poll();
    let before_recovery = ui.prepare_frame()?;
    let (replacement_device, replacement_queue) = block_on(
        host_gpu.adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("gpui_embed_mock_recovery_device"),
            required_features: requirements.features,
            required_limits: wgpu::Limits::downlevel_defaults()
                .using_resolution(host_gpu.adapter.limits())
                .using_alignment(host_gpu.adapter.limits()),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        }),
    )
    .map_err(|error| {
        std::io::Error::other(format!("failed to request recovery device: {error}"))
    })?;
    let replacement_device = Arc::new(replacement_device);
    let replacement_queue = Arc::new(replacement_queue);
    let replacement_host_gpu = HostGpu::new(
        host_gpu.adapter.clone(),
        replacement_device.clone(),
        replacement_queue.clone(),
        format,
    );
    ui.replace_gpu_context(
        host_gpu.adapter.as_ref(),
        replacement_device.clone(),
        replacement_queue.clone(),
        Some(&external_registry),
    )?;
    let stale_recovery_target = OffscreenTarget::new(&replacement_host_gpu, target_size);
    let mut stale_recovery_encoder =
        replacement_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mock_recovery_stale_registry_encoder"),
        });
    let stale_registry_error = ui
        .encode_with_external_surfaces(
            stale_recovery_target.target(wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)),
            &mut stale_recovery_encoder,
            &external_registry,
        )
        .expect_err("the old external view must be rejected after GPU recovery");
    assert!(
        stale_registry_error
            .to_string()
            .contains("must be replaced after GPU recovery")
    );
    let _ = ui.poll();
    let recovery_status = ui.prepare_frame()?;
    assert!(
        recovery_status.scene_generation >= before_recovery.scene_generation,
        "recovery did not leave a retained frame available"
    );
    let recovery_scene = ui
        .rendered_scene()
        .expect("recovery did not retain a scene");
    let recovery_logical_bounds = viewport_bounds
        .borrow()
        .expect("recovery did not export viewport bounds");
    let recovery_device_bounds =
        recovery_logical_bounds.to_device_pixels(recovery_scene.metrics.scale_factor);
    viewport_target = OffscreenTarget::new(&replacement_host_gpu, recovery_device_bounds.size);
    let recovery_pipeline = create_gpu_scene_pipeline(replacement_device.as_ref(), format);
    let recovery_target = OffscreenTarget::new(&replacement_host_gpu, target_size);
    let mut recovery_encoder =
        replacement_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mock_recovery_encoder"),
        });
    encode_engine_viewport(
        &mut recovery_encoder,
        &recovery_pipeline,
        &viewport_target,
        wgpu::Color {
            r: 0.88,
            g: 0.03,
            b: 0.03,
            a: 1.0,
        },
    );
    assert!(external_registry.replace(
        external_surface_id,
        WgpuExternalSurface {
            view: Arc::new(viewport_target.view().clone()),
        },
    ));
    ui.encode_with_external_surfaces(
        recovery_target.target(wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)),
        &mut recovery_encoder,
        &external_registry,
    )?;
    replacement_queue.submit(Some(recovery_encoder.finish()));
    assert!(ui.mark_presented(recovery_status.scene_generation)?);
    let (recovery_pixels, recovery_stride) = readback_target(
        replacement_device.as_ref(),
        replacement_queue.as_ref(),
        &recovery_target,
    )?;
    let recovery_bright_text_pixels = recovery_pixels
        .chunks(recovery_stride as usize)
        .flat_map(|row| row[..(WIDTH * 4) as usize].chunks_exact(4))
        .filter(|pixel| pixel[0] > 180 && pixel[1] > 180 && pixel[2] > 180)
        .count();
    let recovery_red_pixels = recovery_pixels
        .chunks(recovery_stride as usize)
        .flat_map(|row| row[..(WIDTH * 4) as usize].chunks_exact(4))
        .filter(|pixel| pixel[0] > 180 && pixel[1] < 110 && pixel[2] < 110)
        .count();
    assert!(
        recovery_red_pixels > 100,
        "the replacement external view did not change the composed pixels"
    );
    assert!(
        recovery_bright_text_pixels > 10,
        "GPUI text did not render after device recovery"
    );

    println!(
        "mock_wgpu passed: {} GPUI instance bytes, {} non-background pixels, {} bright text pixels, {} engine triangle pixels, {} external viewport pixels, {} recovery pixels, {} recovery text pixels",
        stats.instance_bytes,
        non_background_pixels,
        bright_text_pixels,
        triangle_pixels,
        external_green_pixels,
        recovery_red_pixels,
        recovery_bright_text_pixels
    );
    Ok(())
}
