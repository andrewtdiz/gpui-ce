//! The gpui-component `hello_world` case hosted by `gpui_embed`.
//!
//! This example owns the window, WGPU surface, event loop, input translation, command encoder,
//! submission, and presentation. The button and root are gpui-component APIs; the animated
//! triangle is rendered by the arbitrary native WGPU host into an external surface.

use gpui::{
    AppContext, Bounds, Context, DevicePixels, ExternalSurfaceId, ExternalSurfaceRegistry,
    IntoElement, Modifiers, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, Pixels,
    PlatformInput, Render, Window, block_on, div, external_surface, point, prelude::*, px, size,
};
use gpui_component::{ActiveTheme as _, Root, button::*};
use gpui_embed::{EmbeddedConfig, EmbeddedGpui, HostGpu, OffscreenTarget, WindowMetrics};
use gpui_wgpu::{WgpuExternalSurface, WgpuSceneRenderer, wgpu};
use std::{cell::RefCell, rc::Rc, sync::Arc, time::Instant};
use winit::{
    application::ApplicationHandler,
    dpi::{LogicalSize, PhysicalPosition, PhysicalSize},
    event::{ElementState, MouseButton as WinitMouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window as WinitWindow, WindowId},
};

const WINDOW_WIDTH: f64 = 960.0;
const WINDOW_HEIGHT: f64 = 720.0;
const VIEWPORT_WIDTH: f32 = 520.0;
const VIEWPORT_HEIGHT: f32 = 240.0;

struct DemoUi {
    clicks: u32,
    external_surface_id: ExternalSurfaceId,
    viewport_bounds: Rc<RefCell<Option<Bounds<Pixels>>>>,
}

impl Render for DemoUi {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let button_label = format!(
            "Let's Go! · {} click{}",
            self.clicks,
            if self.clicks == 1 { "" } else { "s" }
        );
        let viewport_bounds = self.viewport_bounds.clone();
        div()
            .size_full()
            .flex()
            .items_center()
            .justify_center()
            .child(
                div()
                    .w(px(720.))
                    .p_6()
                    .rounded_lg()
                    .border_1()
                    .border_color(cx.theme().border)
                    .bg(cx.theme().surface)
                    .text_color(cx.theme().surface_foreground)
                    .child(div().text_xl().child("Hello, gpui-component!"))
                    .child(
                        div()
                            .mt_2()
                            .text_sm()
                            .text_color(cx.theme().muted_foreground)
                            .child("The controls are gpui-component; the triangle is rendered by the host's own WGPU pass."),
                    )
                    .child(
                        Button::new("component-hello-button")
                            .mt_4()
                            .primary()
                            .label(button_label)
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.clicks = this.clicks.saturating_add(1);
                                cx.notify();
                            })),
                    )
                    .child(
                        div()
                            .mt_5()
                            .w(px(VIEWPORT_WIDTH))
                            .h(px(VIEWPORT_HEIGHT))
                            .overflow_hidden()
                            .rounded_md()
                            .on_children_prepainted(move |children, _, _| {
                                *viewport_bounds.borrow_mut() = children.into_iter().next();
                            })
                            .child(
                                external_surface(self.external_surface_id)
                                    .w(px(VIEWPORT_WIDTH))
                                    .h(px(VIEWPORT_HEIGHT)),
                            ),
                    ),
            )
    }
}

fn create_gpu_scene_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
) -> (wgpu::RenderPipeline, wgpu::Buffer, wgpu::BindGroup) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("windowed_gpu_scene_shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
            r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

struct SceneUniforms {
    pointer: vec2<f32>,
    time: f32,
    hovered: f32,
};

@group(0) @binding(0)
var<uniform> scene: SceneUniforms;

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
    let angle = scene.time * 0.9;
    let rotation = mat2x2<f32>(
        cos(angle), -sin(angle),
        sin(angle),  cos(angle),
    );
    let pointer_offset = select(
        vec2<f32>(0.0),
        scene.pointer * vec2<f32>(0.12, 0.12),
        scene.hovered > 0.5,
    );
    var output: VertexOutput;
    output.position = vec4<f32>(rotation * positions[index] + pointer_offset, 0.0, 1.0);
    output.color = colors[index];
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let pulse = 0.85 + 0.15 * sin(scene.time * 3.0);
    let hover_boost = select(1.0, 1.18, scene.hovered > 0.5);
    return vec4<f32>(input.color * pulse * hover_boost, 1.0);
}
"#,
        )),
    });
    let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("windowed_gpu_scene_uniform_layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: Some(wgpu::BufferSize::new(16).unwrap()),
            },
            count: None,
        }],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("windowed_gpu_scene_pipeline_layout"),
        bind_group_layouts: &[Some(&uniform_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("windowed_gpu_scene_pipeline"),
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
    });
    let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("windowed_gpu_scene_uniform_buffer"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("windowed_gpu_scene_bind_group"),
        layout: &uniform_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    });
    (pipeline, uniform_buffer, bind_group)
}

fn encode_gpu_scene(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    target: &OffscreenTarget,
    clear: wgpu::Color,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("windowed_gpu_scene_pass"),
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
    pass.set_bind_group(0, bind_group, &[]);
    pass.draw(0..3, 0..1);
}

fn scene_uniform_bytes(pointer: [f32; 2], time: f32, hovered: bool) -> [u8; 16] {
    let values = [pointer[0], pointer[1], time, hovered as u32 as f32];
    let mut bytes = [0; 16];
    for (index, value) in values.into_iter().enumerate() {
        bytes[index * 4..(index + 1) * 4].copy_from_slice(&value.to_ne_bytes());
    }
    bytes
}

struct WindowedApp {
    window: Option<Arc<WinitWindow>>,
    instance: Option<wgpu::Instance>,
    surface: Option<wgpu::Surface<'static>>,
    surface_config: Option<wgpu::SurfaceConfiguration>,
    host_gpu: Option<HostGpu>,
    ui: Option<EmbeddedGpui>,
    external_registry: ExternalSurfaceRegistry<WgpuExternalSurface>,
    external_surface_id: Option<ExternalSurfaceId>,
    viewport_target: Option<OffscreenTarget>,
    viewport_pipeline: Option<wgpu::RenderPipeline>,
    viewport_uniform_buffer: Option<wgpu::Buffer>,
    viewport_bind_group: Option<wgpu::BindGroup>,
    viewport_bounds: Rc<RefCell<Option<Bounds<Pixels>>>>,
    animation_start: Instant,
    cursor_position: gpui::Point<Pixels>,
    modifiers: Modifiers,
    pressed_button: Option<MouseButton>,
}

impl Default for WindowedApp {
    fn default() -> Self {
        Self {
            window: None,
            instance: None,
            surface: None,
            surface_config: None,
            host_gpu: None,
            ui: None,
            external_registry: ExternalSurfaceRegistry::default(),
            external_surface_id: None,
            viewport_target: None,
            viewport_pipeline: None,
            viewport_uniform_buffer: None,
            viewport_bind_group: None,
            viewport_bounds: Rc::new(RefCell::new(None)),
            animation_start: Instant::now(),
            cursor_position: point(px(0.), px(0.)),
            modifiers: Modifiers::default(),
            pressed_button: None,
        }
    }
}

impl WindowedApp {
    fn fail(&mut self, event_loop: &ActiveEventLoop, error: impl std::fmt::Display) {
        eprintln!("gpui_component example failed: {error}");
        event_loop.exit();
    }

    fn window(&self) -> Option<&Arc<WinitWindow>> {
        self.window.as_ref()
    }

    fn update_metrics(&self) {
        let (Some(window), Some(ui)) = (self.window(), self.ui.as_ref()) else {
            return;
        };
        let scale_factor = window.scale_factor();
        let physical_size = window.inner_size();
        let logical_size = physical_size.to_logical::<f64>(scale_factor);
        ui.set_window_metrics(WindowMetrics::new(
            size(
                px(logical_size.width as f32),
                px(logical_size.height as f32),
            ),
            scale_factor as f32,
        ));
    }

    fn viewport_device_size(&self) -> gpui::Size<DevicePixels> {
        let scale_factor = self
            .window()
            .map(|window| window.scale_factor())
            .unwrap_or(1.0);
        size(
            DevicePixels((VIEWPORT_WIDTH as f64 * scale_factor).round() as i32),
            DevicePixels((VIEWPORT_HEIGHT as f64 * scale_factor).round() as i32),
        )
    }

    fn ensure_viewport_target(&mut self) {
        let (Some(gpu), Some(id)) = (self.host_gpu.as_ref(), self.external_surface_id) else {
            return;
        };
        let desired_size = self.viewport_device_size();
        let needs_replacement = self
            .viewport_target
            .as_ref()
            .is_none_or(|target| target.size() != desired_size);
        if !needs_replacement {
            return;
        }

        let target = OffscreenTarget::new(gpu, desired_size);
        if self.external_registry.get(id).is_some() {
            let replaced = self.external_registry.replace(
                id,
                WgpuExternalSurface {
                    view: Arc::new(target.view().clone()),
                },
            );
            assert!(replaced, "windowed external surface disappeared");
        }
        self.viewport_target = Some(target);
    }

    fn configure_surface(&mut self, physical_size: PhysicalSize<u32>) {
        let (Some(surface), Some(gpu), Some(config)) = (
            self.surface.as_ref(),
            self.host_gpu.as_ref(),
            self.surface_config.as_mut(),
        ) else {
            return;
        };
        config.width = physical_size.width.max(1);
        config.height = physical_size.height.max(1);
        surface.configure(&gpu.device, config);
    }

    fn dispatch_input(&self, input: PlatformInput) {
        let Some(ui) = self.ui.as_ref() else {
            return;
        };
        match ui.dispatch_input(input) {
            Ok(outcome) if outcome.redraw_requested => {
                if let Some(window) = self.window() {
                    window.request_redraw();
                }
            }
            Ok(_) => {}
            Err(error) => eprintln!("input dispatch failed: {error:#}"),
        }
    }

    fn logical_cursor_position(&self, position: PhysicalPosition<f64>) -> gpui::Point<Pixels> {
        let scale_factor = self
            .window()
            .map(|window| window.scale_factor())
            .unwrap_or(1.0);
        let logical = position.to_logical::<f64>(scale_factor);
        point(px(logical.x as f32), px(logical.y as f32))
    }

    fn viewport_pointer(&self) -> ([f32; 2], bool) {
        let Some(bounds) = self.viewport_bounds.borrow().as_ref().copied() else {
            return ([0.0, 0.0], false);
        };
        let local_x = f32::from(self.cursor_position.x) - f32::from(bounds.origin.x);
        let local_y = f32::from(self.cursor_position.y) - f32::from(bounds.origin.y);
        let width = f32::from(bounds.size.width).max(f32::EPSILON);
        let height = f32::from(bounds.size.height).max(f32::EPSILON);
        let hovered = local_x >= 0.0 && local_x <= width && local_y >= 0.0 && local_y <= height;
        (
            [
                (local_x / width).clamp(0.0, 1.0) * 2.0 - 1.0,
                (local_y / height).clamp(0.0, 1.0) * 2.0 - 1.0,
            ],
            hovered,
        )
    }

    fn gpui_mouse_button(button: WinitMouseButton) -> Option<MouseButton> {
        match button {
            WinitMouseButton::Left => Some(MouseButton::Left),
            WinitMouseButton::Right => Some(MouseButton::Right),
            WinitMouseButton::Middle => Some(MouseButton::Middle),
            WinitMouseButton::Back => Some(MouseButton::Navigate(gpui::NavigationDirection::Back)),
            WinitMouseButton::Forward => {
                Some(MouseButton::Navigate(gpui::NavigationDirection::Forward))
            }
            WinitMouseButton::Other(_) => None,
        }
    }

    fn render(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none()
            || self.surface.is_none()
            || self.host_gpu.is_none()
            || self.ui.is_none()
            || self.surface_config.is_none()
        {
            return;
        }

        self.ensure_viewport_target();
        let window = self.window.as_ref().unwrap().clone();
        let (width, height, format) = {
            let config = self.surface_config.as_ref().unwrap();
            (config.width, config.height, config.format)
        };
        if width == 0 || height == 0 {
            return;
        }

        let surface_result = self.surface.as_ref().unwrap().get_current_texture();
        let frame = match surface_result {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                self.configure_surface(window.inner_size());
                window.request_redraw();
                return;
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return;
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                self.configure_surface(window.inner_size());
                window.request_redraw();
                return;
            }
        };

        let gpu = self.host_gpu.as_ref().unwrap().clone();
        let ui = self.ui.as_ref().unwrap();
        let _ = ui.poll();
        let frame_status = match ui.prepare_frame() {
            Ok(status) => status,
            Err(error) => {
                eprintln!("windowed_wgpu failed to prepare a frame: {error}");
                event_loop.exit();
                return;
            }
        };
        let (pointer, hovered) = self.viewport_pointer();
        let uniform_bytes = scene_uniform_bytes(
            pointer,
            self.animation_start.elapsed().as_secs_f32(),
            hovered,
        );
        if let Some(uniform_buffer) = self.viewport_uniform_buffer.as_ref() {
            gpu.queue.write_buffer(uniform_buffer, 0, &uniform_bytes);
        }
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("windowed_wgpu_frame_encoder"),
            });
        if let (Some(viewport_target), Some(viewport_pipeline), Some(viewport_bind_group)) = (
            self.viewport_target.as_ref(),
            self.viewport_pipeline.as_ref(),
            self.viewport_bind_group.as_ref(),
        ) {
            encode_gpu_scene(
                &mut encoder,
                viewport_pipeline,
                viewport_bind_group,
                viewport_target,
                wgpu::Color {
                    r: 0.02,
                    g: 0.08,
                    b: 0.18,
                    a: 1.0,
                },
            );
        }
        let target = gpui_wgpu::WgpuRenderTarget {
            view: &view,
            size: size(DevicePixels(width as i32), DevicePixels(height as i32)),
            format,
            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        };
        if let Err(error) =
            ui.encode_with_external_surfaces(target, &mut encoder, &self.external_registry)
        {
            gpu.queue.present(frame);
            eprintln!("windowed_wgpu failed to encode a frame: {error}");
            event_loop.exit();
            return;
        }
        gpu.queue.submit(Some(encoder.finish()));
        gpu.queue.present(frame);
        if let Err(error) = ui.mark_presented(frame_status.scene_generation) {
            eprintln!("windowed_wgpu failed to acknowledge a frame: {error}");
            event_loop.exit();
            return;
        }
        window.request_redraw();
    }

    fn initialize(&mut self, event_loop: &ActiveEventLoop) -> Result<(), String> {
        let window = Arc::new(
            event_loop
                .create_window(
                    WinitWindow::default_attributes()
                        .with_title("gpui-component on embedded gpui-ce + WGPU")
                        .with_inner_size(LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT)),
                )
                .map_err(|error| error.to_string())?,
        );
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            display: None,
        });
        let surface = instance
            .create_surface(window.clone())
            .map_err(|error| format!("failed to create WGPU surface: {error}"))?;
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .map_err(|error| format!("failed to request WGPU adapter: {error}"))?;
        let surface_caps = surface.get_capabilities(&adapter);
        let format = surface_caps
            .formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb)
            .unwrap_or(surface_caps.formats[0]);
        let requirements = WgpuSceneRenderer::requirements(&adapter);
        let (device, queue) = block_on(
            adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("windowed_gpui_embed_device"),
                required_features: requirements.features,
                required_limits: wgpu::Limits::downlevel_defaults()
                    .using_resolution(adapter.limits())
                    .using_alignment(adapter.limits()),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            }),
        )
        .map_err(|error| format!("failed to request WGPU device: {error}"))?;
        let host_gpu = HostGpu::new(Arc::new(adapter), Arc::new(device), Arc::new(queue), format);
        let physical_size = window.inner_size();
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: physical_size.width.max(1),
            height: physical_size.height.max(1),
            present_mode: if surface_caps
                .present_modes
                .contains(&wgpu::PresentMode::Mailbox)
            {
                wgpu::PresentMode::Mailbox
            } else {
                wgpu::PresentMode::Fifo
            },
            desired_maximum_frame_latency: 2,
            alpha_mode: surface_caps.alpha_modes[0],
            color_space: wgpu::SurfaceColorSpace::Auto,
            view_formats: vec![],
        };
        surface.configure(&host_gpu.device, &surface_config);

        let scale_factor = window.scale_factor();
        let logical_size = physical_size.to_logical::<f64>(scale_factor);
        let metrics = WindowMetrics::new(
            size(
                px(logical_size.width as f32),
                px(logical_size.height as f32),
            ),
            scale_factor as f32,
        );
        let viewport_size = size(
            DevicePixels((VIEWPORT_WIDTH as f64 * scale_factor).round() as i32),
            DevicePixels((VIEWPORT_HEIGHT as f64 * scale_factor).round() as i32),
        );
        let viewport_target = OffscreenTarget::new(&host_gpu, viewport_size);
        let external_surface_id = self.external_registry.register(WgpuExternalSurface {
            view: Arc::new(viewport_target.view().clone()),
        });
        let config = EmbeddedConfig::new(host_gpu.clone()).with_metrics(metrics);
        let viewport_bounds = self.viewport_bounds.clone();
        let ui = EmbeddedGpui::new_with_root(config, move |window, cx| {
            gpui_component::init(cx);
            let view = cx.new(|_| DemoUi {
                clicks: 0,
                external_surface_id,
                viewport_bounds,
            });
            cx.new(|cx| Root::new(view, window, cx))
        })
        .map_err(|error| format!("failed to initialize embedded GPUI: {error}"))?;
        let (viewport_pipeline, viewport_uniform_buffer, viewport_bind_group) =
            create_gpu_scene_pipeline(&host_gpu.device, format);

        self.window = Some(window);
        self.instance = Some(instance);
        self.surface = Some(surface);
        self.surface_config = Some(surface_config);
        self.host_gpu = Some(host_gpu);
        self.ui = Some(ui);
        self.external_surface_id = Some(external_surface_id);
        self.viewport_target = Some(viewport_target);
        self.viewport_pipeline = Some(viewport_pipeline);
        self.viewport_uniform_buffer = Some(viewport_uniform_buffer);
        self.viewport_bind_group = Some(viewport_bind_group);
        Ok(())
    }
}

impl ApplicationHandler for WindowedApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        if let Err(error) = self.initialize(event_loop) {
            self.fail(event_loop, error);
            return;
        }
        self.update_metrics();
        if let Some(window) = self.window() {
            window.request_redraw();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if self.window().is_none_or(|window| window.id() != window_id) {
            return;
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => self.render(event_loop),
            WindowEvent::Resized(size) => {
                self.configure_surface(size);
                self.update_metrics();
                self.ensure_viewport_target();
                if let Some(window) = self.window() {
                    window.request_redraw();
                }
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                self.update_metrics();
                self.ensure_viewport_target();
                if let Some(window) = self.window().cloned() {
                    self.configure_surface(window.inner_size());
                    window.request_redraw();
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_position = self.logical_cursor_position(position);
                self.dispatch_input(PlatformInput::MouseMove(MouseMoveEvent {
                    position: self.cursor_position,
                    pressed_button: self.pressed_button,
                    modifiers: self.modifiers,
                }));
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let Some(button) = Self::gpui_mouse_button(button) else {
                    return;
                };
                match state {
                    ElementState::Pressed => {
                        self.pressed_button = Some(button);
                        self.dispatch_input(PlatformInput::MouseDown(MouseDownEvent {
                            button,
                            position: self.cursor_position,
                            modifiers: self.modifiers,
                            click_count: 1,
                            first_mouse: true,
                        }));
                    }
                    ElementState::Released => {
                        self.dispatch_input(PlatformInput::MouseUp(MouseUpEvent {
                            button,
                            position: self.cursor_position,
                            modifiers: self.modifiers,
                            click_count: 1,
                        }));
                        self.pressed_button = None;
                    }
                }
                if let Some(window) = self.window() {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window() {
            window.request_redraw();
        }
    }
}

fn main() -> gpui::Result<()> {
    let event_loop = EventLoop::new().map_err(|error| std::io::Error::other(error.to_string()))?;
    event_loop
        .run_app(&mut WindowedApp::default())
        .map_err(|error| std::io::Error::other(error.to_string()).into())
}
