use crate::{CompositorGpuHint, WgpuAtlas, WgpuContext, WgpuDeviceRequirements};
use bytemuck::{Pod, Zeroable};
use gpui::{
    AtlasTextureId, BackdropFilter, Background, Bounds, DevicePixels, ExternalSurface,
    ExternalSurfaceId, ExternalSurfaceRegistry, FilterBoundary, GpuSpecs, MonochromeSprite,
    PaintSurface, Path, Point, PolychromeSprite, PrimitiveBatch, Quad, ScaledFilter, ScaledPixels,
    Scene, Shadow, Size, SubpixelSprite, Underline, get_gamma_correction_ratios,
};
use log::warn;

/// The largest blur radius in a scene-space filter chain, in device pixels — used to size the
/// blur kernel and the dilated region the blur passes are scissored to.
///
/// The `match` is exhaustive on purpose: adding a [`ScaledFilter`] variant breaks it here,
/// forcing this backend to handle (or deliberately ignore) the new filter rather than silently
/// dropping it.
fn max_blur_radius(filters: &[ScaledFilter]) -> f32 {
    filters.iter().fold(0.0, |acc, filter| match filter {
        ScaledFilter::Blur(radius) => acc.max(radius.0),
    })
}
#[cfg(not(target_family = "wasm"))]
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::cell::RefCell;
use std::collections::HashMap;
use std::num::NonZeroU64;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GlobalParams {
    viewport_size: [f32; 2],
    premultiplied_alpha: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct PodBounds {
    origin: [f32; 2],
    size: [f32; 2],
}

impl From<Bounds<ScaledPixels>> for PodBounds {
    fn from(bounds: Bounds<ScaledPixels>) -> Self {
        Self {
            origin: [bounds.origin.x.0, bounds.origin.y.0],
            size: [bounds.size.width.0, bounds.size.height.0],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SurfaceParams {
    bounds: PodBounds,
    content_mask: PodBounds,
}

/// Uniform passed to the blur pipelines. The same struct drives the downsample, separable
/// gaussian, and composite passes; fields not relevant to a given pass are left zero.
#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct BlurParams {
    /// Composite target rectangle, in device pixels (composite pass only).
    bounds: PodBounds,
    /// Clip rectangle, in device pixels (composite pass only).
    content_mask: PodBounds,
    /// Rounded-corner radii (tl, tr, br, bl), in device pixels (composite pass only).
    corner_radii: [f32; 4],
    /// Per-tap sampling step in UV space (gaussian passes only): (1/width, 0) or (0, 1/height).
    direction: [f32; 2],
    /// Gaussian sigma, in the (half-resolution) blur texture's pixels.
    sigma: f32,
    /// Element opacity, multiplied into the composited result.
    opacity: f32,
    /// Number of taps to each side of center (gaussian passes only).
    tap_count: f32,
    /// Spacing between taps in pixels; >1 lets `tap_count` taps span very large radii without
    /// truncating the gaussian (see #6 in review).
    tap_step: f32,
    /// 1.0 to clip the composite to the rounded rect (backdrop — the panel has a defined shape),
    /// 0.0 to let the blurred result fade out on its own (content `filter` — it bleeds past the
    /// element bounds like CSS, so the fade isn't sharply truncated at the box edge).
    clip_rounded: f32,
    /// 1.0 = snapped 2:1 box downsample (anchor the half-res grid to a fixed 2px grid at the
    /// origin, so a stationary element blurs identically at every window size); 0.0 = 1:1 copy
    /// (the scene blit, which must not downsample). Downsample pass only.
    downsample: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GammaParams {
    gamma_ratios: [f32; 4],
    grayscale_enhanced_contrast: f32,
    subpixel_enhanced_contrast: f32,
    is_bgr: u32,
    _pad: u32,
}

#[derive(Clone, Debug)]
#[repr(C)]
struct PathSprite {
    bounds: Bounds<ScaledPixels>,
}

#[derive(Clone, Debug)]
#[repr(C)]
struct PathRasterizationVertex {
    xy_position: Point<ScaledPixels>,
    st_position: Point<f32>,
    color: Background,
    bounds: Bounds<ScaledPixels>,
}

pub struct WgpuSurfaceConfig {
    pub size: Size<DevicePixels>,
    pub transparent: bool,
    /// Preferred presentation mode. When `Some`, the renderer will use this
    /// mode if supported by the surface, falling back to `Fifo`.
    /// When `None`, defaults to `Fifo` (VSync).
    ///
    /// Mobile platforms may prefer `Mailbox` (triple-buffering) to avoid
    /// blocking in `get_current_texture()` during lifecycle transitions.
    pub preferred_present_mode: Option<wgpu::PresentMode>,
}

/// Configuration for a renderer that records GPUI scene commands into a host-owned target.
///
/// The renderer does not create or configure a surface. The host is responsible for creating a
/// single-sample texture view with this format and size and passing it to [`WgpuSceneRenderer::encode`].
#[derive(Clone, Copy, Debug)]
pub struct WgpuSceneRendererConfig {
    pub size: Size<DevicePixels>,
    pub format: wgpu::TextureFormat,
    pub transparent: bool,
}

/// A host-owned color target for [`WgpuSceneRenderer::encode`].
///
/// The target must be a single-sample view created with the format supplied to
/// [`WgpuSceneRendererConfig`]. The host owns the texture and decides when the command encoder is
/// submitted. Dimensions must be positive. GPUI always stores the final color attachment; the
/// host chooses only whether the first pass loads or clears the target. MSAA, arbitrary formats,
/// and embedded filter backgrounds are outside this contract.
pub struct WgpuRenderTarget<'a> {
    pub view: &'a wgpu::TextureView,
    pub size: Size<DevicePixels>,
    /// The format of `view`.
    pub format: wgpu::TextureFormat,
    /// The load operation for the target's first UI pass.
    pub load: wgpu::LoadOp<wgpu::Color>,
}

/// Basic information about the work recorded for a scene.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RenderStats {
    /// Bytes reserved in the renderer's instance buffer for this scene.
    pub instance_bytes: u64,
}

/// A WGPU view registered for a backend-neutral GPUI external surface.
#[derive(Clone)]
pub struct WgpuExternalSurface {
    /// The host-owned single-sample texture view sampled during GPUI encoding.
    ///
    /// The view must be recreated and replaced in the external-surface registry after host GPU
    /// device recovery; the surface ID remains stable while the registry generation advances.
    pub view: Arc<wgpu::TextureView>,
}

struct WgpuPipelines {
    quads: wgpu::RenderPipeline,
    shadows: wgpu::RenderPipeline,
    path_rasterization: wgpu::RenderPipeline,
    paths: wgpu::RenderPipeline,
    underlines: wgpu::RenderPipeline,
    mono_sprites: wgpu::RenderPipeline,
    subpixel_sprites: Option<wgpu::RenderPipeline>,
    poly_sprites: wgpu::RenderPipeline,
    #[allow(dead_code)]
    surfaces: wgpu::RenderPipeline,
    /// Copies a source texture into the (smaller) target with one bilinear tap. Used both to
    /// downsample the scene into the half-resolution blur texture and to blit the offscreen
    /// scene into the swapchain at the end of the frame.
    blur_downsample: wgpu::RenderPipeline,
    /// One axis of a separable gaussian blur; direction is supplied per draw via [`BlurParams`].
    blur: wgpu::RenderPipeline,
    /// Composites a blurred texture into a rounded rectangle (with clip + opacity).
    blur_composite: wgpu::RenderPipeline,
}

struct WgpuBindGroupLayouts {
    globals: wgpu::BindGroupLayout,
    instances: wgpu::BindGroupLayout,
    instances_with_texture: wgpu::BindGroupLayout,
    surfaces: wgpu::BindGroupLayout,
    blur: wgpu::BindGroupLayout,
}

/// Shared GPU context reference, used to coordinate device recovery across multiple windows.
pub type GpuContext = Rc<RefCell<Option<WgpuContext>>>;

/// GPU resources that must be dropped together during device recovery.
struct WgpuResources {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipelines: WgpuPipelines,
    bind_group_layouts: WgpuBindGroupLayouts,
    atlas_sampler: wgpu::Sampler,
    surface_sampler: wgpu::Sampler,
    #[allow(dead_code)]
    surface_uniform_buffer: wgpu::Buffer,
    /// One reused uniform buffer holding [`BlurParams`] for every blur pass in a frame, each at a
    /// distinct (alignment-strided) offset. Avoids allocating a buffer per pass; distinct offsets
    /// mean `write_buffer`'s last-write-at-submit semantics don't clobber earlier passes.
    blur_params_buffer: wgpu::Buffer,
    globals_buffer: wgpu::Buffer,
    globals_bind_group: wgpu::BindGroup,
    path_globals_bind_group: wgpu::BindGroup,
    instance_buffer: wgpu::Buffer,
    path_intermediate_texture: Option<wgpu::Texture>,
    path_intermediate_view: Option<wgpu::TextureView>,
    path_msaa_texture: Option<wgpu::Texture>,
    path_msaa_view: Option<wgpu::TextureView>,
    /// Blur offscreen targets. Allocated lazily (only when a frame actually uses a blur filter)
    /// so apps that never blur pay no extra VRAM. `None`/empty until first use.
    ///
    /// Full-resolution offscreen color target the scene is rendered into so that blur passes
    /// can sample already-painted content; blitted to the swapchain at the end of the frame.
    scene_color_texture: Option<wgpu::Texture>,
    scene_color_view: Option<wgpu::TextureView>,
    /// Half-resolution ping/pong targets for the downsample + separable gaussian passes.
    blur_ping_texture: Option<wgpu::Texture>,
    blur_ping_view: Option<wgpu::TextureView>,
    blur_pong_texture: Option<wgpu::Texture>,
    blur_pong_view: Option<wgpu::TextureView>,
    /// Full-resolution offscreen targets a content-filter (`filter`) group renders into before
    /// being blurred and composited back. One per nesting level (indexed by depth) so nested
    /// content blurs isolate correctly, up to [`MAX_FILTER_DEPTH`]; deeper nests render inline.
    group_textures: Vec<wgpu::Texture>,
    group_views: Vec<wgpu::TextureView>,
}

impl WgpuResources {
    fn invalidate_intermediate_textures(&mut self) {
        self.path_intermediate_texture = None;
        self.path_intermediate_view = None;
        self.path_msaa_texture = None;
        self.path_msaa_view = None;
        self.scene_color_texture = None;
        self.scene_color_view = None;
        self.blur_ping_texture = None;
        self.blur_ping_view = None;
        self.blur_pong_texture = None;
        self.blur_pong_view = None;
        self.group_textures.clear();
        self.group_views.clear();
    }
}

/// Number of content-filter (`filter`) nesting levels that get their own isolated group texture.
/// Two covers the realistic "a blurred element inside another blurred element" case; deeper nests
/// render inline (unblurred at the inner level) rather than allocating unbounded VRAM.
const MAX_FILTER_DEPTH: usize = 2;

/// Number of [`BlurParams`] slots in the shared blur-params buffer (one per blur pass per frame).
/// Each frame uses 4 passes per backdrop/group plus one blit; 256 covers dozens of filters.
const BLUR_PARAMS_SLOTS: u64 = 256;
/// Initial number of surface draw uniforms reserved in one command encoder. The buffer grows
/// before encoding a larger scene so every surface in one host submission gets a distinct slot.
const INITIAL_SURFACE_UNIFORM_SLOTS: u64 = 256;

pub struct WgpuSceneRenderer {
    resources: Option<WgpuResources>,
    config: WgpuSceneRendererConfig,
    atlas: Arc<WgpuAtlas>,
    path_globals_offset: u64,
    gamma_offset: u64,
    instance_buffer_capacity: u64,
    max_buffer_size: u64,
    storage_buffer_alignment: u64,
    /// Stride between [`BlurParams`] slots in `blur_params_buffer`, and a per-frame bump cursor
    /// (in slots) handed out to blur passes. Cell so the `&self` blur helpers can advance it.
    blur_params_stride: u64,
    blur_params_slot: std::cell::Cell<u64>,
    rendering_params: RenderingParameters,
    is_bgr: bool,
    dual_source_blending: bool,
    adapter_info: wgpu::AdapterInfo,
    max_texture_size: u32,
    surface_uniform_stride: u64,
    surface_uniform_capacity: u64,
    surface_uniform_slot: std::cell::Cell<u64>,
    pending_external_surface_replacements: HashMap<ExternalSurfaceId, u64>,
}

impl WgpuSceneRenderer {
    fn resources(&self) -> &WgpuResources {
        self.resources
            .as_ref()
            .expect("GPU resources not available")
    }

    fn resources_mut(&mut self) -> &mut WgpuResources {
        self.resources
            .as_mut()
            .expect("GPU resources not available")
    }

    /// Returns optional device features used by GPUI's subpixel text pipeline.
    pub fn requirements(adapter: &wgpu::Adapter) -> WgpuDeviceRequirements {
        let mut requirements = WgpuDeviceRequirements::default();
        if adapter
            .features()
            .contains(wgpu::Features::DUAL_SOURCE_BLENDING)
        {
            requirements.features |= wgpu::Features::DUAL_SOURCE_BLENDING;
        }
        requirements
    }

    /// Creates a surface-free scene renderer from host-owned GPU handles.
    pub fn from_host(
        adapter: &wgpu::Adapter,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        config: WgpuSceneRendererConfig,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            config.size.width.0 >= 0 && config.size.height.0 >= 0,
            "host render target dimensions must be non-negative"
        );
        anyhow::ensure!(
            config.size.width.0 as u32 <= device.limits().max_texture_dimension_2d
                && config.size.height.0 as u32 <= device.limits().max_texture_dimension_2d,
            "host render target size {:?} exceeds device limit {}",
            config.size,
            device.limits().max_texture_dimension_2d
        );
        Self::validate_host_target_format(adapter, config.format)?;
        let atlas = Arc::new(WgpuAtlas::new(device.clone(), queue.clone(), config.format));
        Self::from_parts(adapter, device, queue, config, atlas)
    }

    fn validate_host_target_format(
        adapter: &wgpu::Adapter,
        format: wgpu::TextureFormat,
    ) -> anyhow::Result<()> {
        let format_features = adapter.get_texture_format_features(format);
        anyhow::ensure!(
            format_features.allowed_usages.contains(
                wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
            ),
            "host target format {:?} does not support GPUI render and atlas usages",
            format
        );
        Ok(())
    }

    fn from_parts(
        adapter: &wgpu::Adapter,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        config: WgpuSceneRendererConfig,
        atlas: Arc<WgpuAtlas>,
    ) -> anyhow::Result<Self> {
        let max_texture_size = device.limits().max_texture_dimension_2d;
        let alpha_mode = if config.transparent {
            wgpu::CompositeAlphaMode::PreMultiplied
        } else {
            wgpu::CompositeAlphaMode::Opaque
        };
        let dual_source_blending = device
            .features()
            .contains(wgpu::Features::DUAL_SOURCE_BLENDING);
        let rendering_params = RenderingParameters::new(adapter, config.format);
        let bind_group_layouts = Self::create_bind_group_layouts(&device);
        let pipelines = Self::create_pipelines(
            &device,
            &bind_group_layouts,
            config.format,
            alpha_mode,
            rendering_params.path_sample_count,
            dual_source_blending,
        );

        let atlas_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("atlas_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let surface_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("surface_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let uniform_alignment = device.limits().min_uniform_buffer_offset_alignment as u64;
        let surface_uniform_stride =
            (std::mem::size_of::<SurfaceParams>() as u64).next_multiple_of(uniform_alignment);
        let surface_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("surface_uniform_buffer"),
            size: surface_uniform_stride * INITIAL_SURFACE_UNIFORM_SLOTS,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let blur_params_stride =
            (std::mem::size_of::<BlurParams>() as u64).next_multiple_of(uniform_alignment);
        let blur_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("blur_params_buffer"),
            size: blur_params_stride * BLUR_PARAMS_SLOTS,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let globals_size = std::mem::size_of::<GlobalParams>() as u64;
        let gamma_size = std::mem::size_of::<GammaParams>() as u64;
        let path_globals_offset = globals_size.next_multiple_of(uniform_alignment);
        let gamma_offset = (path_globals_offset + globals_size).next_multiple_of(uniform_alignment);
        let globals_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals_buffer"),
            size: gamma_offset + gamma_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let max_buffer_size = device.limits().max_buffer_size;
        let storage_buffer_alignment = device.limits().min_storage_buffer_offset_alignment as u64;
        let initial_instance_buffer_capacity = 2 * 1024 * 1024;
        anyhow::ensure!(
            initial_instance_buffer_capacity <= max_buffer_size,
            "device max buffer size is too small for GPUI's instance buffer"
        );
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instance_buffer"),
            size: initial_instance_buffer_capacity,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let globals_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals_bind_group"),
            layout: &bind_group_layouts.globals,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &globals_buffer,
                        offset: 0,
                        size: Some(NonZeroU64::new(globals_size).unwrap()),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &globals_buffer,
                        offset: gamma_offset,
                        size: Some(NonZeroU64::new(gamma_size).unwrap()),
                    }),
                },
            ],
        });
        let path_globals_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("path_globals_bind_group"),
            layout: &bind_group_layouts.globals,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &globals_buffer,
                        offset: path_globals_offset,
                        size: Some(NonZeroU64::new(globals_size).unwrap()),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &globals_buffer,
                        offset: gamma_offset,
                        size: Some(NonZeroU64::new(gamma_size).unwrap()),
                    }),
                },
            ],
        });

        let adapter_info = adapter.get_info();

        let resources = WgpuResources {
            device,
            queue,
            pipelines,
            bind_group_layouts,
            atlas_sampler,
            surface_sampler,
            surface_uniform_buffer,
            blur_params_buffer,
            globals_buffer,
            globals_bind_group,
            path_globals_bind_group,
            instance_buffer,
            path_intermediate_texture: None,
            path_intermediate_view: None,
            path_msaa_texture: None,
            path_msaa_view: None,
            scene_color_texture: None,
            scene_color_view: None,
            blur_ping_texture: None,
            blur_ping_view: None,
            blur_pong_texture: None,
            blur_pong_view: None,
            group_textures: Vec::new(),
            group_views: Vec::new(),
        };

        Ok(Self {
            resources: Some(resources),
            config,
            atlas,
            path_globals_offset,
            gamma_offset,
            instance_buffer_capacity: initial_instance_buffer_capacity,
            max_buffer_size,
            storage_buffer_alignment,
            blur_params_stride,
            blur_params_slot: std::cell::Cell::new(0),
            rendering_params,
            is_bgr: false,
            dual_source_blending,
            adapter_info,
            max_texture_size,
            surface_uniform_stride,
            surface_uniform_capacity: INITIAL_SURFACE_UNIFORM_SLOTS,
            surface_uniform_slot: std::cell::Cell::new(0),
            pending_external_surface_replacements: HashMap::new(),
        })
    }

    fn install_error_handler(&self, last_error: Arc<Mutex<Option<String>>>) {
        let device = self.resources().device.clone();
        device.on_uncaptured_error(Arc::new(move |error| {
            *last_error.lock().unwrap() = Some(error.to_string());
        }));
    }

    fn create_bind_group_layouts(device: &wgpu::Device) -> WgpuBindGroupLayouts {
        let globals =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("globals_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(
                                std::mem::size_of::<GlobalParams>() as u64
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(
                                std::mem::size_of::<GammaParams>() as u64
                            ),
                        },
                        count: None,
                    },
                ],
            });

        let storage_buffer_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let instances = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("instances_layout"),
            entries: &[storage_buffer_entry(0)],
        });

        let instances_with_texture =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("instances_with_texture_layout"),
                entries: &[
                    storage_buffer_entry(0),
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let surfaces = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("surfaces_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<SurfaceParams>() as u64
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let blur = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blur_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(std::mem::size_of::<BlurParams>() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        WgpuBindGroupLayouts {
            globals,
            instances,
            instances_with_texture,
            surfaces,
            blur,
        }
    }

    fn create_pipelines(
        device: &wgpu::Device,
        layouts: &WgpuBindGroupLayouts,
        surface_format: wgpu::TextureFormat,
        alpha_mode: wgpu::CompositeAlphaMode,
        path_sample_count: u32,
        dual_source_blending: bool,
    ) -> WgpuPipelines {
        // Diagnostic guard: verify the device actually has
        // DUAL_SOURCE_BLENDING. We have a crash report (ZED-5G1) where a
        // feature mismatch caused a wgpu-hal abort, but we haven't
        // identified the code path that produces the mismatch. This
        // guard prevents the crash and logs more evidence.
        // Remove this check once:
        // a) We find and fix the root cause, or
        // b) There are no reports of this warning appearing for some time.
        let device_has_feature = device
            .features()
            .contains(wgpu::Features::DUAL_SOURCE_BLENDING);
        if dual_source_blending && !device_has_feature {
            log::error!(
                "BUG: dual_source_blending flag is true but device does not \
                 have DUAL_SOURCE_BLENDING enabled (device features: {:?}). \
                 Falling back to mono text rendering. Please report this at \
                 https://github.com/zed-industries/zed/issues",
                device.features(),
            );
        }
        let dual_source_blending = dual_source_blending && device_has_feature;

        let base_shader_source = include_str!("shaders.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpui_shaders"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(base_shader_source)),
        });

        let subpixel_shader_source = include_str!("shaders_subpixel.wgsl");
        let subpixel_shader_module = if dual_source_blending {
            let combined = format!(
                "enable dual_source_blending;\n{base_shader_source}\n{subpixel_shader_source}"
            );
            Some(device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gpui_subpixel_shaders"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(combined)),
            }))
        } else {
            None
        };

        let blend_mode = match alpha_mode {
            wgpu::CompositeAlphaMode::PreMultiplied => {
                wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING
            }
            _ => wgpu::BlendState::ALPHA_BLENDING,
        };

        let color_target = wgpu::ColorTargetState {
            format: surface_format,
            blend: Some(blend_mode),
            write_mask: wgpu::ColorWrites::ALL,
        };

        let create_pipeline = |name: &str,
                               vs_entry: &str,
                               fs_entry: &str,
                               globals_layout: &wgpu::BindGroupLayout,
                               data_layout: &wgpu::BindGroupLayout,
                               topology: wgpu::PrimitiveTopology,
                               color_targets: &[Option<wgpu::ColorTargetState>],
                               sample_count: u32,
                               module: &wgpu::ShaderModule| {
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{name}_layout")),
                bind_group_layouts: &[Some(globals_layout), Some(data_layout)],
                immediate_size: 0,
            });

            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(name),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some(vs_entry),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some(fs_entry),
                    targets: color_targets,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            })
        };

        let quads = create_pipeline(
            "quads",
            "vs_quad",
            "fs_quad",
            &layouts.globals,
            &layouts.instances,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target.clone())],
            1,
            &shader_module,
        );

        let shadows = create_pipeline(
            "shadows",
            "vs_shadow",
            "fs_shadow",
            &layouts.globals,
            &layouts.instances,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target.clone())],
            1,
            &shader_module,
        );

        let path_rasterization = create_pipeline(
            "path_rasterization",
            "vs_path_rasterization",
            "fs_path_rasterization",
            &layouts.globals,
            &layouts.instances,
            wgpu::PrimitiveTopology::TriangleList,
            &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            path_sample_count,
            &shader_module,
        );

        let paths_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };

        let paths = create_pipeline(
            "paths",
            "vs_path",
            "fs_path",
            &layouts.globals,
            &layouts.instances_with_texture,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(paths_blend),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            1,
            &shader_module,
        );

        let underlines = create_pipeline(
            "underlines",
            "vs_underline",
            "fs_underline",
            &layouts.globals,
            &layouts.instances,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target.clone())],
            1,
            &shader_module,
        );

        let mono_sprites = create_pipeline(
            "mono_sprites",
            "vs_mono_sprite",
            "fs_mono_sprite",
            &layouts.globals,
            &layouts.instances_with_texture,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target.clone())],
            1,
            &shader_module,
        );

        let subpixel_sprites = if let Some(subpixel_module) = &subpixel_shader_module {
            let subpixel_blend = wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Src1,
                    dst_factor: wgpu::BlendFactor::OneMinusSrc1,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
            };

            Some(create_pipeline(
                "subpixel_sprites",
                "vs_subpixel_sprite",
                "fs_subpixel_sprite",
                &layouts.globals,
                &layouts.instances_with_texture,
                wgpu::PrimitiveTopology::TriangleStrip,
                &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(subpixel_blend),
                    write_mask: wgpu::ColorWrites::COLOR,
                })],
                1,
                subpixel_module,
            ))
        } else {
            None
        };

        let poly_sprites = create_pipeline(
            "poly_sprites",
            "vs_poly_sprite",
            "fs_poly_sprite",
            &layouts.globals,
            &layouts.instances_with_texture,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target.clone())],
            1,
            &shader_module,
        );

        let surfaces = create_pipeline(
            "surfaces",
            "vs_surface",
            "fs_surface",
            &layouts.globals,
            &layouts.surfaces,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target)],
            1,
            &shader_module,
        );

        // Blur pipelines all sample one texture into another; the downsample and gaussian passes
        // overwrite their (intermediate) target, while the composite blends over the scene.
        let no_blend_target = wgpu::ColorTargetState {
            format: surface_format,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        };

        let blur_downsample = create_pipeline(
            "blur_downsample",
            "vs_blur_fullscreen",
            "fs_blur_downsample",
            &layouts.globals,
            &layouts.blur,
            wgpu::PrimitiveTopology::TriangleList,
            &[Some(no_blend_target.clone())],
            1,
            &shader_module,
        );

        let blur = create_pipeline(
            "blur",
            "vs_blur_fullscreen",
            "fs_blur",
            &layouts.globals,
            &layouts.blur,
            wgpu::PrimitiveTopology::TriangleList,
            &[Some(no_blend_target)],
            1,
            &shader_module,
        );

        // The blurred sample is premultiplied (blurring against the transparent, rgb=0 region
        // around the source scales rgb with the fading alpha), so the composite outputs
        // premultiplied and blends premultiplied — straight alpha blending would multiply rgb by
        // alpha a second time and darken the faded edges. Independent of the window's alpha mode.
        let premultiplied_target = wgpu::ColorTargetState {
            format: surface_format,
            blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::ALL,
        };
        let blur_composite = create_pipeline(
            "blur_composite",
            "vs_blur_composite",
            "fs_blur_composite",
            &layouts.globals,
            &layouts.blur,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(premultiplied_target)],
            1,
            &shader_module,
        );

        WgpuPipelines {
            quads,
            shadows,
            path_rasterization,
            paths,
            underlines,
            mono_sprites,
            subpixel_sprites,
            poly_sprites,
            surfaces,
            blur_downsample,
            blur,
            blur_composite,
        }
    }

    fn create_path_intermediate(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("path_intermediate"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn create_msaa_if_needed(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        sample_count: u32,
    ) -> Option<(wgpu::Texture, wgpu::TextureView)> {
        if sample_count <= 1 {
            return None;
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("path_msaa"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Some((texture, view))
    }

    pub fn update_drawable_size(&mut self, size: Size<DevicePixels>) {
        let width = size.width.0.max(0) as u32;
        let height = size.height.0.max(0) as u32;
        let current_width = self.config.size.width.0.max(0) as u32;
        let current_height = self.config.size.height.0.max(0) as u32;

        if width != current_width || height != current_height {
            let clamped_width = width.min(self.max_texture_size);
            let clamped_height = height.min(self.max_texture_size);

            if clamped_width != width || clamped_height != height {
                warn!(
                    "Requested surface size ({}, {}) exceeds maximum texture dimension {}. \
                     Clamping to ({}, {}). Window content may not fill the entire window.",
                    width, height, self.max_texture_size, clamped_width, clamped_height
                );
            }

            self.config.size = Size {
                width: DevicePixels(clamped_width as i32),
                height: DevicePixels(clamped_height as i32),
            };

            // Dropping renderer-owned intermediate handles is the lifetime boundary. The host
            // owns device polling and decides when old GPU work is retired.
            if let Some(resources) = self.resources.as_mut() {
                resources.invalidate_intermediate_textures();
            }
        }
    }

    fn ensure_intermediate_textures(&mut self) {
        if self.resources().path_intermediate_texture.is_some() {
            return;
        }

        let format = self.config.format;
        let width = self.config.size.width.0.max(1) as u32;
        let height = self.config.size.height.0.max(1) as u32;
        let path_sample_count = self.rendering_params.path_sample_count;
        let resources = self.resources_mut();

        let (t, v) = Self::create_path_intermediate(&resources.device, format, width, height);
        resources.path_intermediate_texture = Some(t);
        resources.path_intermediate_view = Some(v);

        let (path_msaa_texture, path_msaa_view) = Self::create_msaa_if_needed(
            &resources.device,
            format,
            width,
            height,
            path_sample_count,
        )
        .map(|(t, v)| (Some(t), Some(v)))
        .unwrap_or((None, None));
        resources.path_msaa_texture = path_msaa_texture;
        resources.path_msaa_view = path_msaa_view;
    }

    /// Lazily allocate the blur offscreen targets — the full-res scene texture, half-res
    /// ping/pong, and one full-res group texture per nesting level. Called only on frames that
    /// actually use a blur filter, so non-blurring apps never pay this VRAM. A no-op once
    /// allocated (invalidated alongside the path intermediates on resize / device loss).
    fn ensure_blur_textures(&mut self) {
        if self.resources().scene_color_texture.is_some() {
            return;
        }
        let format = self.config.format;
        let width = self.config.size.width.0.max(1) as u32;
        let height = self.config.size.height.0.max(1) as u32;
        let blur_width = (width / 2).max(1);
        let blur_height = (height / 2).max(1);
        let resources = self.resources_mut();

        let (t, v) = Self::create_path_intermediate(&resources.device, format, width, height);
        resources.scene_color_texture = Some(t);
        resources.scene_color_view = Some(v);
        let (t, v) =
            Self::create_path_intermediate(&resources.device, format, blur_width, blur_height);
        resources.blur_ping_texture = Some(t);
        resources.blur_ping_view = Some(v);
        let (t, v) =
            Self::create_path_intermediate(&resources.device, format, blur_width, blur_height);
        resources.blur_pong_texture = Some(t);
        resources.blur_pong_view = Some(v);

        for _ in 0..MAX_FILTER_DEPTH {
            let (t, v) = Self::create_path_intermediate(&resources.device, format, width, height);
            resources.group_textures.push(t);
            resources.group_views.push(v);
        }
    }

    pub fn set_subpixel_layout(&mut self, is_bgr: bool) {
        self.is_bgr = is_bgr;
    }

    pub fn update_transparency(&mut self, transparent: bool) {
        if transparent != self.config.transparent {
            self.config.transparent = transparent;
            let alpha_mode = self.alpha_mode();
            let path_sample_count = self.rendering_params.path_sample_count;
            let dual_source_blending = self.dual_source_blending;
            let Some(resources) = self.resources.as_mut() else {
                return;
            };
            resources.pipelines = Self::create_pipelines(
                &resources.device,
                &resources.bind_group_layouts,
                self.config.format,
                alpha_mode,
                path_sample_count,
                dual_source_blending,
            );
        }
    }

    #[allow(dead_code)]
    pub fn viewport_size(&self) -> Size<DevicePixels> {
        self.config.size
    }

    /// Returns the format required for encoded targets.
    pub fn target_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    /// Rebuilds all GPUI GPU resources for a replacement host device and queue.
    ///
    /// Construction is transactional: the live renderer and its shared atlas remain untouched
    /// until replacement resources have been built successfully. The atlas [`Arc`] itself is
    /// preserved so the GPUI window keeps the same platform-atlas identity. The renderer format
    /// is fixed for its lifetime. When an external-surface registry is supplied, its current
    /// generations are snapshotted and each registered view must be replaced before that ID is
    /// encoded again. Hosts without external surfaces may pass `None`.
    pub fn replace_gpu_context(
        &mut self,
        adapter: &wgpu::Adapter,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        external_surfaces: Option<&ExternalSurfaceRegistry<WgpuExternalSurface>>,
    ) -> anyhow::Result<()> {
        let config = self.config;
        Self::validate_host_target_format(adapter, config.format)?;
        let atlas = self.atlas.clone();
        let pending_external_surface_replacements = external_surfaces
            .map(|registry| registry.generations().collect())
            .unwrap_or_default();
        let replacement_device = device.clone();
        let replacement_queue = queue.clone();
        let mut replacement = Self::from_parts(adapter, device, queue, config, atlas.clone())?;
        replacement.pending_external_surface_replacements = pending_external_surface_replacements;
        atlas.replace_gpu_context(replacement_device, replacement_queue, config.format);
        *self = replacement;
        Ok(())
    }

    fn alpha_mode(&self) -> wgpu::CompositeAlphaMode {
        if self.config.transparent {
            wgpu::CompositeAlphaMode::PreMultiplied
        } else {
            wgpu::CompositeAlphaMode::Opaque
        }
    }

    pub fn sprite_atlas(&self) -> &Arc<WgpuAtlas> {
        &self.atlas
    }

    pub fn supports_dual_source_blending(&self) -> bool {
        self.dual_source_blending
    }

    pub fn gpu_context(&self) -> (Arc<wgpu::Device>, Arc<wgpu::Queue>) {
        let resources = self.resources();
        (resources.device.clone(), resources.queue.clone())
    }

    pub fn gpu_specs(&self) -> GpuSpecs {
        GpuSpecs {
            is_software_emulated: self.adapter_info.device_type == wgpu::DeviceType::Cpu,
            device_name: self.adapter_info.name.clone(),
            driver_name: self.adapter_info.driver.clone(),
            driver_info: self.adapter_info.driver_info.clone(),
        }
    }

    pub fn max_texture_size(&self) -> u32 {
        self.max_texture_size
    }

    fn ensure_host_instance_capacity(&mut self, required: u64) -> anyhow::Result<()> {
        while required > self.instance_buffer_capacity {
            anyhow::ensure!(
                self.instance_buffer_capacity < self.max_buffer_size,
                "GPUI scene requires {required} bytes, exceeding the device buffer limit"
            );
            self.grow_instance_buffer();
        }
        Ok(())
    }

    fn ensure_surface_uniform_capacity(&mut self, required: u64) -> anyhow::Result<()> {
        if required <= self.surface_uniform_capacity {
            return Ok(());
        }

        let mut capacity = self.surface_uniform_capacity;
        while capacity < required {
            capacity = capacity
                .checked_mul(2)
                .ok_or_else(|| anyhow::anyhow!("surface uniform capacity overflow"))?;
        }
        let buffer_size = self
            .surface_uniform_stride
            .checked_mul(capacity)
            .ok_or_else(|| anyhow::anyhow!("surface uniform buffer size overflow"))?;
        anyhow::ensure!(
            buffer_size <= self.resources().device.limits().max_buffer_size,
            "GPUI scene requires {buffer_size} bytes for external surface uniforms, exceeding the device buffer limit"
        );
        let resources = self.resources_mut();
        resources.surface_uniform_buffer =
            resources.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("surface_uniform_buffer"),
                size: buffer_size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        self.surface_uniform_capacity = capacity;
        Ok(())
    }

    fn required_host_instance_bytes(&self, scene: &Scene) -> u64 {
        let mut total: u64 = 0;
        let alignment = self.storage_buffer_alignment;
        let mut add = |bytes: usize| {
            let bytes = (bytes as u64).max(16);
            total = total.next_multiple_of(alignment).saturating_add(bytes);
        };

        for batch in scene.batches() {
            match batch {
                PrimitiveBatch::Quads(range) => add(std::mem::size_of_val(&scene.quads[range])),
                PrimitiveBatch::Shadows(range) => add(std::mem::size_of_val(&scene.shadows[range])),
                PrimitiveBatch::Paths(range) => {
                    let paths = &scene.paths[range];
                    let vertex_count = paths.iter().map(|path| path.vertices.len()).sum::<usize>();
                    add(std::mem::size_of::<PathRasterizationVertex>() * vertex_count);
                    add(std::mem::size_of::<PathSprite>() * paths.len());
                }
                PrimitiveBatch::Underlines(range) => {
                    add(std::mem::size_of_val(&scene.underlines[range]))
                }
                PrimitiveBatch::MonochromeSprites { range, .. } => {
                    add(std::mem::size_of_val(&scene.monochrome_sprites[range]))
                }
                PrimitiveBatch::SubpixelSprites { range, .. } => {
                    add(std::mem::size_of_val(&scene.subpixel_sprites[range]))
                }
                PrimitiveBatch::PolychromeSprites { range, .. } => {
                    add(std::mem::size_of_val(&scene.polychrome_sprites[range]))
                }
                PrimitiveBatch::Surfaces(_)
                | PrimitiveBatch::ExternalSurfaces(_)
                | PrimitiveBatch::BackdropFilters(_)
                | PrimitiveBatch::FilterBoundary(_) => {}
            }
        }
        total
    }

    /// Records a retained GPUI scene into the supplied host-owned target.
    ///
    /// Submit the host encoder before calling this again on the same renderer. GPUI reuses its
    /// instance and uniform buffers, and the queue uploads for a later frame must not overwrite
    /// data referenced by an earlier unsubmitted command buffer.
    pub fn encode(
        &mut self,
        scene: &Scene,
        encoder: &mut wgpu::CommandEncoder,
        target: WgpuRenderTarget<'_>,
    ) -> anyhow::Result<RenderStats> {
        self.encode_scene(scene, encoder, target, false, None)
    }

    /// Records a scene and samples registered host-owned external surfaces in scene order.
    pub fn encode_with_external_surfaces(
        &mut self,
        scene: &Scene,
        encoder: &mut wgpu::CommandEncoder,
        target: WgpuRenderTarget<'_>,
        registry: &ExternalSurfaceRegistry<WgpuExternalSurface>,
    ) -> anyhow::Result<RenderStats> {
        self.encode_scene(scene, encoder, target, false, Some(registry))
    }

    fn encode_native(
        &mut self,
        scene: &Scene,
        encoder: &mut wgpu::CommandEncoder,
        target: WgpuRenderTarget<'_>,
    ) -> anyhow::Result<RenderStats> {
        self.encode_scene(scene, encoder, target, true, None)
    }

    fn invalidate_intermediate_textures(&mut self) {
        if let Some(resources) = self.resources.as_mut() {
            resources.invalidate_intermediate_textures();
        }
    }

    fn encode_scene(
        &mut self,
        scene: &Scene,
        encoder: &mut wgpu::CommandEncoder,
        target: WgpuRenderTarget<'_>,
        allow_filters: bool,
        external_surfaces: Option<&ExternalSurfaceRegistry<WgpuExternalSurface>>,
    ) -> anyhow::Result<RenderStats> {
        anyhow::ensure!(
            target.size.width.0 > 0 && target.size.height.0 > 0,
            "host render target dimensions must be positive; skip encoding while minimized"
        );
        anyhow::ensure!(
            target.size.width.0 as u32 <= self.max_texture_size
                && target.size.height.0 as u32 <= self.max_texture_size,
            "host render target size {:?} exceeds device limit {}",
            target.size,
            self.max_texture_size
        );
        anyhow::ensure!(
            target.format == self.config.format,
            "host target format {:?} does not match renderer format {:?}",
            target.format,
            self.config.format
        );
        if !allow_filters {
            anyhow::ensure!(
                scene.backdrop_filters.is_empty() && scene.filter_boundaries.is_empty(),
                "backdrop and content filters require a host-provided sampleable background target"
            );
        }
        self.validate_external_surfaces(scene, external_surfaces)?;

        let width = target.size.width.0 as u32;
        let height = target.size.height.0 as u32;
        self.update_drawable_size(target.size);

        self.atlas.before_frame();
        self.ensure_intermediate_textures();
        let use_offscreen =
            !scene.backdrop_filters.is_empty() || !scene.filter_boundaries.is_empty();
        if use_offscreen {
            self.ensure_blur_textures();
        }
        self.ensure_host_instance_capacity(self.required_host_instance_bytes(scene))?;
        self.ensure_surface_uniform_capacity(
            (scene.surfaces.len() + scene.external_surfaces.len()) as u64,
        )?;

        let gamma_params = GammaParams {
            gamma_ratios: self.rendering_params.gamma_ratios,
            grayscale_enhanced_contrast: self.rendering_params.grayscale_enhanced_contrast,
            subpixel_enhanced_contrast: self.rendering_params.subpixel_enhanced_contrast,
            is_bgr: self.is_bgr as u32,
            _pad: 0,
        };
        let globals = GlobalParams {
            viewport_size: [width as f32, height as f32],
            premultiplied_alpha: if self.alpha_mode() == wgpu::CompositeAlphaMode::PreMultiplied {
                1
            } else {
                0
            },
            pad: 0,
        };
        let path_globals = GlobalParams {
            premultiplied_alpha: 0,
            ..globals
        };
        {
            let resources = self.resources();
            resources.queue.write_buffer(
                &resources.globals_buffer,
                0,
                bytemuck::bytes_of(&globals),
            );
            resources.queue.write_buffer(
                &resources.globals_buffer,
                self.path_globals_offset,
                bytemuck::bytes_of(&path_globals),
            );
            resources.queue.write_buffer(
                &resources.globals_buffer,
                self.gamma_offset,
                bytemuck::bytes_of(&gamma_params),
            );
        }

        let mut instance_offset = 0;
        self.blur_params_slot.set(0);
        self.surface_uniform_slot.set(0);
        let scene_color_view = if use_offscreen {
            Some(
                self.resources()
                    .scene_color_view
                    .as_ref()
                    .expect("scene color allocated for filtered scene")
                    .clone(),
            )
        } else {
            None
        };
        let mut current_target = scene_color_view
            .clone()
            .unwrap_or_else(|| target.view.clone());
        let group_views = if use_offscreen {
            self.resources().group_views.clone()
        } else {
            Vec::new()
        };
        let mut filter_stack: Vec<(FilterBoundary, wgpu::TextureView, bool)> = Vec::new();
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(if use_offscreen {
                "gpui_native_scene"
            } else {
                "gpui_host_scene"
            }),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &current_target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: if use_offscreen {
                        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
                    } else {
                        target.load
                    },
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });

        for batch in scene.batches() {
            let ok = match batch {
                PrimitiveBatch::Quads(range) => {
                    self.draw_quads(&scene.quads[range], &mut instance_offset, &mut pass)
                }
                PrimitiveBatch::Shadows(range) => {
                    self.draw_shadows(&scene.shadows[range], &mut instance_offset, &mut pass)
                }
                PrimitiveBatch::Paths(range) => {
                    let paths = &scene.paths[range];
                    if paths.is_empty() {
                        continue;
                    }
                    drop(pass);
                    let did_draw =
                        self.draw_paths_to_intermediate(encoder, paths, &mut instance_offset);
                    pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("gpui_scene_continued"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &current_target,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: None,
                        ..Default::default()
                    });
                    if did_draw {
                        self.draw_paths_from_intermediate(paths, &mut instance_offset, &mut pass)
                    } else {
                        false
                    }
                }
                PrimitiveBatch::Underlines(range) => {
                    self.draw_underlines(&scene.underlines[range], &mut instance_offset, &mut pass)
                }
                PrimitiveBatch::MonochromeSprites { texture_id, range } => self
                    .draw_monochrome_sprites(
                        &scene.monochrome_sprites[range],
                        texture_id,
                        &mut instance_offset,
                        &mut pass,
                    ),
                PrimitiveBatch::SubpixelSprites { texture_id, range } => self
                    .draw_subpixel_sprites(
                        &scene.subpixel_sprites[range],
                        texture_id,
                        &mut instance_offset,
                        &mut pass,
                    ),
                PrimitiveBatch::PolychromeSprites { texture_id, range } => self
                    .draw_polychrome_sprites(
                        &scene.polychrome_sprites[range],
                        texture_id,
                        &mut instance_offset,
                        &mut pass,
                    ),
                PrimitiveBatch::Surfaces(range) => {
                    self.draw_surfaces(&scene.surfaces[range], &mut pass)
                }
                PrimitiveBatch::ExternalSurfaces(range) => {
                    let registry = external_surfaces
                        .expect("external surfaces were validated before encoding");
                    self.draw_external_surfaces(
                        &scene.external_surfaces[range],
                        registry,
                        &mut pass,
                    )?;
                    true
                }
                PrimitiveBatch::BackdropFilters(range) => {
                    drop(pass);
                    for filter in &scene.backdrop_filters[range] {
                        self.draw_backdrop_filter(encoder, filter, &current_target);
                    }
                    pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("gpui_scene_continued"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &current_target,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: None,
                        ..Default::default()
                    });
                    true
                }
                PrimitiveBatch::FilterBoundary(ix) => {
                    let boundary = scene.filter_boundaries[ix].clone();
                    if boundary.is_start {
                        let depth = filter_stack.iter().filter(|entry| entry.2).count();
                        if depth < group_views.len() {
                            drop(pass);
                            let parent = current_target.clone();
                            current_target = group_views[depth].clone();
                            filter_stack.push((boundary, parent, true));
                            pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("gpui_filter_group"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &current_target,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                        store: wgpu::StoreOp::Store,
                                    },
                                    depth_slice: None,
                                })],
                                depth_stencil_attachment: None,
                                ..Default::default()
                            });
                        } else {
                            filter_stack.push((boundary, current_target.clone(), false));
                        }
                    } else if let Some((boundary, parent, isolated)) = filter_stack.pop() {
                        if isolated {
                            drop(pass);
                            self.blur_and_composite(
                                encoder,
                                &current_target,
                                &parent,
                                boundary.bounds,
                                boundary.content_mask.bounds,
                                [
                                    boundary.corner_radii.top_left.0,
                                    boundary.corner_radii.top_right.0,
                                    boundary.corner_radii.bottom_right.0,
                                    boundary.corner_radii.bottom_left.0,
                                ],
                                max_blur_radius(&boundary.filters),
                                boundary.opacity,
                                false,
                            );
                            current_target = parent;
                            pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("gpui_scene_continued"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &current_target,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Load,
                                        store: wgpu::StoreOp::Store,
                                    },
                                    depth_slice: None,
                                })],
                                depth_stencil_attachment: None,
                                ..Default::default()
                            });
                        }
                    }
                    true
                }
            };
            if !ok {
                drop(pass);
                return Err(anyhow::anyhow!(
                    "GPUI host scene exceeded its instance-buffer capacity"
                ));
            }
        }
        drop(pass);

        if let Some(scene_color_view) = &scene_color_view {
            self.blit_to_frame(encoder, scene_color_view, target.view);
        }

        Ok(RenderStats {
            instance_bytes: instance_offset,
        })
    }

    fn draw_quads(
        &self,
        quads: &[Quad],
        instance_offset: &mut u64,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> bool {
        let data = unsafe { Self::instance_bytes(quads) };
        self.draw_instances(
            data,
            quads.len() as u32,
            &self.resources().pipelines.quads,
            instance_offset,
            pass,
        )
    }

    fn draw_shadows(
        &self,
        shadows: &[Shadow],
        instance_offset: &mut u64,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> bool {
        let data = unsafe { Self::instance_bytes(shadows) };
        self.draw_instances(
            data,
            shadows.len() as u32,
            &self.resources().pipelines.shadows,
            instance_offset,
            pass,
        )
    }

    fn draw_underlines(
        &self,
        underlines: &[Underline],
        instance_offset: &mut u64,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> bool {
        let data = unsafe { Self::instance_bytes(underlines) };
        self.draw_instances(
            data,
            underlines.len() as u32,
            &self.resources().pipelines.underlines,
            instance_offset,
            pass,
        )
    }

    fn draw_monochrome_sprites(
        &self,
        sprites: &[MonochromeSprite],
        texture_id: AtlasTextureId,
        instance_offset: &mut u64,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> bool {
        let tex_info = self.atlas.get_texture_info(texture_id);
        let data = unsafe { Self::instance_bytes(sprites) };
        self.draw_instances_with_texture(
            data,
            sprites.len() as u32,
            &tex_info.view,
            &self.resources().pipelines.mono_sprites,
            instance_offset,
            pass,
        )
    }

    fn draw_subpixel_sprites(
        &self,
        sprites: &[SubpixelSprite],
        texture_id: AtlasTextureId,
        instance_offset: &mut u64,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> bool {
        let tex_info = self.atlas.get_texture_info(texture_id);
        let data = unsafe { Self::instance_bytes(sprites) };
        let resources = self.resources();
        let pipeline = resources
            .pipelines
            .subpixel_sprites
            .as_ref()
            .unwrap_or(&resources.pipelines.mono_sprites);
        self.draw_instances_with_texture(
            data,
            sprites.len() as u32,
            &tex_info.view,
            pipeline,
            instance_offset,
            pass,
        )
    }

    #[cfg(any(target_os = "linux", target_os = "freebsd"))]
    fn draw_surfaces(&self, surfaces: &[PaintSurface], pass: &mut wgpu::RenderPass<'_>) -> bool {
        for surface in surfaces {
            let Some(wgpu_texture) = surface.texture.downcast_ref::<wgpu::Texture>() else {
                continue;
            };

            let texture_view = wgpu_texture.create_view(&wgpu::TextureViewDescriptor::default());
            let params = SurfaceParams {
                bounds: surface.bounds.into(),
                content_mask: surface.content_mask.bounds.into(),
            };
            let bind_group = self.make_surface_bind_group(&params, &texture_view);
            pass.set_pipeline(&self.resources().pipelines.surfaces);
            pass.set_bind_group(0, &self.resources().globals_bind_group, &[]);
            pass.set_bind_group(1, &bind_group, &[]);
            pass.draw(0..4, 0..1);
        }
        true
    }

    #[cfg(not(any(target_os = "linux", target_os = "freebsd")))]
    fn draw_surfaces(&self, _surfaces: &[PaintSurface], _pass: &mut wgpu::RenderPass<'_>) -> bool {
        true
    }

    fn make_surface_bind_group(
        &self,
        params: &SurfaceParams,
        texture_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        let resources = self.resources();
        let slot = self.surface_uniform_slot.get();
        debug_assert!(slot < self.surface_uniform_capacity);
        self.surface_uniform_slot.set(slot + 1);
        let offset = slot * self.surface_uniform_stride;
        resources.queue.write_buffer(
            &resources.surface_uniform_buffer,
            offset,
            bytemuck::bytes_of(params),
        );
        resources
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("surface_bind_group"),
                layout: &resources.bind_group_layouts.surfaces,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &resources.surface_uniform_buffer,
                            offset,
                            size: Some(
                                NonZeroU64::new(std::mem::size_of::<SurfaceParams>() as u64)
                                    .unwrap(),
                            ),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&resources.surface_sampler),
                    },
                ],
            })
    }

    fn validate_external_surfaces(
        &mut self,
        scene: &Scene,
        registry: Option<&ExternalSurfaceRegistry<WgpuExternalSurface>>,
    ) -> anyhow::Result<()> {
        if scene.external_surfaces.is_empty() {
            return Ok(());
        }
        let registry = registry.ok_or_else(|| {
            anyhow::anyhow!(
                "GPUI scene contains {} external surfaces but no external-surface registry was supplied",
                scene.external_surfaces.len()
            )
        })?;
        let mut validated_ids = Vec::with_capacity(scene.external_surfaces.len());
        for surface in &scene.external_surfaces {
            anyhow::ensure!(
                registry.get(surface.id).is_some(),
                "missing external surface ID {}",
                surface.id.value()
            );
            let current_generation = registry
                .generation(surface.id)
                .expect("external surface was validated before reading its generation");
            if let Some(recovery_generation) =
                self.pending_external_surface_replacements.get(&surface.id)
            {
                anyhow::ensure!(
                    current_generation > *recovery_generation,
                    "external surface ID {} must be replaced after GPU recovery",
                    surface.id.value()
                );
            }
            validated_ids.push(surface.id);
        }
        for id in validated_ids {
            self.pending_external_surface_replacements.remove(&id);
        }
        Ok(())
    }

    fn draw_external_surfaces(
        &self,
        surfaces: &[ExternalSurface],
        registry: &ExternalSurfaceRegistry<WgpuExternalSurface>,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> anyhow::Result<()> {
        for surface in surfaces {
            let resource = registry
                .get(surface.id)
                .expect("external surfaces were validated before encoding");
            let params = SurfaceParams {
                bounds: surface.bounds.into(),
                content_mask: surface.content_mask.bounds.into(),
            };
            let bind_group = self.make_surface_bind_group(&params, &resource.view);
            let resources = self.resources();
            pass.set_pipeline(&resources.pipelines.surfaces);
            pass.set_bind_group(0, &resources.globals_bind_group, &[]);
            pass.set_bind_group(1, &bind_group, &[]);
            pass.draw(0..4, 0..1);
        }
        Ok(())
    }

    /// Build a bind group for a blur pass. Writes `params` into the next slot of the shared
    /// `blur_params_buffer` (no per-pass allocation) and references that slot, the source texture,
    /// and the filtering sampler. Distinct per-pass offsets keep `write_buffer`'s
    /// last-write-at-submit semantics from clobbering earlier passes within a frame.
    fn make_blur_bind_group(
        &self,
        params: BlurParams,
        source: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        let resources = self.resources();
        let slot = self.blur_params_slot.get() % BLUR_PARAMS_SLOTS;
        self.blur_params_slot.set(slot + 1);
        let offset = slot * self.blur_params_stride;
        resources.queue.write_buffer(
            &resources.blur_params_buffer,
            offset,
            bytemuck::bytes_of(&params),
        );
        resources
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("blur_bind_group"),
                layout: &resources.bind_group_layouts.blur,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &resources.blur_params_buffer,
                            offset,
                            size: NonZeroU64::new(std::mem::size_of::<BlurParams>() as u64),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(source),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&resources.surface_sampler),
                    },
                ],
            })
    }

    /// Run a full-screen (3-vertex) blur pass that overwrites `target` by sampling `source`.
    /// `scissor` (x, y, w, h, in `target` pixels) limits fragment work to the region that
    /// actually feeds the composite — the element bounds dilated by the kernel radius.
    fn run_blur_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        label: &str,
        pipeline: &wgpu::RenderPipeline,
        target: &wgpu::TextureView,
        source: &wgpu::TextureView,
        params: BlurParams,
        scissor: [u32; 4],
    ) {
        let bind_group = self.make_blur_bind_group(params, source);
        let resources = self.resources();
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &resources.globals_bind_group, &[]);
        pass.set_bind_group(1, &bind_group, &[]);
        pass.set_scissor_rect(scissor[0], scissor[1], scissor[2], scissor[3]);
        pass.draw(0..3, 0..1);
    }

    /// Blur `source` (full-resolution) and composite the result into `target`, clipped to
    /// `bounds`/`corner_radii`/`content_mask` and modulated by `opacity`. Shared by the backdrop
    /// and content-filter paths. Uses the half-resolution ping/pong textures as scratch.
    #[allow(clippy::too_many_arguments)]
    fn blur_and_composite(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::TextureView,
        target: &wgpu::TextureView,
        bounds: Bounds<ScaledPixels>,
        content_mask: Bounds<ScaledPixels>,
        corner_radii: [f32; 4],
        blur_radius: f32,
        opacity: f32,
        // Backdrop clips to the rounded rect; content (`filter`) bleeds past its bounds.
        clip_rounded: bool,
    ) {
        // Sigma is halved because the blur runs at half resolution.
        let sigma = (blur_radius * 0.5).max(0.0);
        if sigma <= 0.0 {
            return;
        }
        // Span ±3σ. If that needs more than 32 taps, spread the taps apart (tap_step > 1) rather
        // than truncating the kernel — keeps very large radii from clipping (review #6).
        let ideal_taps = (3.0 * sigma).ceil();
        let tap_count = ideal_taps.clamp(1.0, 32.0);
        let tap_step = (ideal_taps / tap_count).max(1.0);
        let full_w = self.config.size.width.0.max(1) as u32;
        let full_h = self.config.size.height.0.max(1) as u32;
        let blur_width = (full_w / 2).max(1) as f32;
        let blur_height = (full_h / 2).max(1) as f32;

        // Limit the half-res passes to the element bounds dilated by the kernel radius (3·sigma,
        // full-res) — outside that the composite never samples, so there's no reason to blur it.
        let dilation = 3.0 * blur_radius;
        let hw = (full_w / 2).max(1);
        let hh = (full_h / 2).max(1);
        let x0 = (((bounds.origin.x.0 - dilation) * 0.5).floor().max(0.0) as u32).min(hw);
        let y0 = (((bounds.origin.y.0 - dilation) * 0.5).floor().max(0.0) as u32).min(hh);
        let x1 = ((((bounds.origin.x.0 + bounds.size.width.0 + dilation) * 0.5)
            .ceil()
            .max(0.0) as u32)
            .min(hw))
        .max(x0);
        let y1 = ((((bounds.origin.y.0 + bounds.size.height.0 + dilation) * 0.5)
            .ceil()
            .max(0.0) as u32)
            .min(hh))
        .max(y0);
        let scissor = [x0, y0, x1 - x0, y1 - y0];
        if scissor[2] == 0 || scissor[3] == 0 {
            return;
        }

        // Owned handles so the passes below don't borrow `self`.
        let (ping, pong) = {
            let resources = self.resources();
            match (
                resources.blur_ping_view.as_ref(),
                resources.blur_pong_view.as_ref(),
            ) {
                (Some(ping), Some(pong)) => (ping.clone(), pong.clone()),
                _ => return,
            }
        };

        // Downsample source -> ping, then separable gaussian ping -> pong -> ping.
        self.run_blur_pass(
            encoder,
            "blur_downsample",
            &self.resources().pipelines.blur_downsample,
            &ping,
            source,
            BlurParams {
                downsample: 1.0,
                ..Default::default()
            },
            scissor,
        );
        self.run_blur_pass(
            encoder,
            "blur_horizontal",
            &self.resources().pipelines.blur,
            &pong,
            &ping,
            BlurParams {
                direction: [1.0 / blur_width, 0.0],
                sigma,
                tap_count,
                tap_step,
                ..Default::default()
            },
            scissor,
        );
        self.run_blur_pass(
            encoder,
            "blur_vertical",
            &self.resources().pipelines.blur,
            &ping,
            &pong,
            BlurParams {
                direction: [0.0, 1.0 / blur_height],
                sigma,
                tap_count,
                tap_step,
                ..Default::default()
            },
            scissor,
        );

        // Composite the blurred result into the target (loads existing content). For content blur
        // the quad covers the dilated region so the blur can fade out past the element box (no
        // sharp clip); for backdrop the quad is the element bounds and the shader clips to the
        // rounded rect.
        let composite_bounds = if clip_rounded {
            bounds
        } else {
            bounds.dilate(ScaledPixels(dilation))
        };
        let params = BlurParams {
            bounds: composite_bounds.into(),
            content_mask: content_mask.into(),
            corner_radii,
            opacity,
            clip_rounded: if clip_rounded { 1.0 } else { 0.0 },
            ..Default::default()
        };
        let bind_group = self.make_blur_bind_group(params, &ping);
        let resources = self.resources();
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("blur_composite"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        pass.set_pipeline(&resources.pipelines.blur_composite);
        pass.set_bind_group(0, &resources.globals_bind_group, &[]);
        pass.set_bind_group(1, &bind_group, &[]);
        pass.draw(0..4, 0..1);
    }

    /// Blur the scene painted so far behind `filter.bounds` and composite it back as frosted glass.
    fn draw_backdrop_filter(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        filter: &BackdropFilter,
        scene_color_view: &wgpu::TextureView,
    ) {
        self.blur_and_composite(
            encoder,
            scene_color_view,
            scene_color_view,
            filter.bounds,
            filter.content_mask.bounds,
            [
                filter.corner_radii.top_left.0,
                filter.corner_radii.top_right.0,
                filter.corner_radii.bottom_right.0,
                filter.corner_radii.bottom_left.0,
            ],
            max_blur_radius(&filter.filters),
            filter.opacity,
            true,
        );
    }

    /// Copy the offscreen scene texture into the swapchain texture.
    fn blit_to_frame(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::TextureView,
        frame_view: &wgpu::TextureView,
    ) {
        let bind_group = self.make_blur_bind_group(BlurParams::default(), source);
        let resources = self.resources();
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("scene_blit"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: frame_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        pass.set_pipeline(&resources.pipelines.blur_downsample);
        pass.set_bind_group(0, &resources.globals_bind_group, &[]);
        pass.set_bind_group(1, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    fn draw_polychrome_sprites(
        &self,
        sprites: &[PolychromeSprite],
        texture_id: AtlasTextureId,
        instance_offset: &mut u64,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> bool {
        let tex_info = self.atlas.get_texture_info(texture_id);
        let data = unsafe { Self::instance_bytes(sprites) };
        self.draw_instances_with_texture(
            data,
            sprites.len() as u32,
            &tex_info.view,
            &self.resources().pipelines.poly_sprites,
            instance_offset,
            pass,
        )
    }

    fn draw_instances(
        &self,
        data: &[u8],
        instance_count: u32,
        pipeline: &wgpu::RenderPipeline,
        instance_offset: &mut u64,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> bool {
        if instance_count == 0 {
            return true;
        }
        let Some((offset, size)) = self.write_to_instance_buffer(instance_offset, data) else {
            return false;
        };
        let resources = self.resources();
        let bind_group = resources
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &resources.bind_group_layouts.instances,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.instance_binding(offset, size),
                }],
            });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &resources.globals_bind_group, &[]);
        pass.set_bind_group(1, &bind_group, &[]);
        pass.draw(0..4, 0..instance_count);
        true
    }

    fn draw_instances_with_texture(
        &self,
        data: &[u8],
        instance_count: u32,
        texture_view: &wgpu::TextureView,
        pipeline: &wgpu::RenderPipeline,
        instance_offset: &mut u64,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> bool {
        if instance_count == 0 {
            return true;
        }
        let Some((offset, size)) = self.write_to_instance_buffer(instance_offset, data) else {
            return false;
        };
        let resources = self.resources();
        let bind_group = resources
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &resources.bind_group_layouts.instances_with_texture,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.instance_binding(offset, size),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&resources.atlas_sampler),
                    },
                ],
            });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &resources.globals_bind_group, &[]);
        pass.set_bind_group(1, &bind_group, &[]);
        pass.draw(0..4, 0..instance_count);
        true
    }

    unsafe fn instance_bytes<T>(instances: &[T]) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                instances.as_ptr() as *const u8,
                std::mem::size_of_val(instances),
            )
        }
    }

    fn draw_paths_from_intermediate(
        &self,
        paths: &[Path<ScaledPixels>],
        instance_offset: &mut u64,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> bool {
        let first_path = &paths[0];
        let sprites: Vec<PathSprite> = if paths.last().map(|p| &p.order) == Some(&first_path.order)
        {
            paths
                .iter()
                .map(|p| PathSprite {
                    bounds: p.clipped_bounds(),
                })
                .collect()
        } else {
            let mut bounds = first_path.clipped_bounds();
            for path in paths.iter().skip(1) {
                bounds = bounds.union(&path.clipped_bounds());
            }
            vec![PathSprite { bounds }]
        };

        let resources = self.resources();
        let Some(path_intermediate_view) = resources.path_intermediate_view.as_ref() else {
            return true;
        };

        let sprite_data = unsafe { Self::instance_bytes(&sprites) };
        self.draw_instances_with_texture(
            sprite_data,
            sprites.len() as u32,
            path_intermediate_view,
            &resources.pipelines.paths,
            instance_offset,
            pass,
        )
    }

    fn draw_paths_to_intermediate(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        paths: &[Path<ScaledPixels>],
        instance_offset: &mut u64,
    ) -> bool {
        let mut vertices = Vec::new();
        for path in paths {
            let bounds = path.clipped_bounds();
            vertices.extend(path.vertices.iter().map(|v| PathRasterizationVertex {
                xy_position: v.xy_position,
                st_position: v.st_position,
                color: path.color,
                bounds,
            }));
        }

        if vertices.is_empty() {
            return true;
        }

        let vertex_data = unsafe { Self::instance_bytes(&vertices) };
        let Some((vertex_offset, vertex_size)) =
            self.write_to_instance_buffer(instance_offset, vertex_data)
        else {
            return false;
        };

        let resources = self.resources();
        let data_bind_group = resources
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("path_rasterization_bind_group"),
                layout: &resources.bind_group_layouts.instances,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.instance_binding(vertex_offset, vertex_size),
                }],
            });

        let Some(path_intermediate_view) = resources.path_intermediate_view.as_ref() else {
            return true;
        };

        let (target_view, resolve_target) = if let Some(ref msaa_view) = resources.path_msaa_view {
            (msaa_view, Some(path_intermediate_view))
        } else {
            (path_intermediate_view, None)
        };

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("path_rasterization_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            pass.set_pipeline(&resources.pipelines.path_rasterization);
            pass.set_bind_group(0, &resources.path_globals_bind_group, &[]);
            pass.set_bind_group(1, &data_bind_group, &[]);
            pass.draw(0..vertices.len() as u32, 0..1);
        }

        true
    }

    fn grow_instance_buffer(&mut self) {
        let new_capacity = (self.instance_buffer_capacity * 2).min(self.max_buffer_size);
        log::info!("increased instance buffer size to {}", new_capacity);
        let resources = self.resources_mut();
        resources.instance_buffer = resources.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instance_buffer"),
            size: new_capacity,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.instance_buffer_capacity = new_capacity;
    }

    fn write_to_instance_buffer(
        &self,
        instance_offset: &mut u64,
        data: &[u8],
    ) -> Option<(u64, NonZeroU64)> {
        let offset = (*instance_offset).next_multiple_of(self.storage_buffer_alignment);
        let size = (data.len() as u64).max(16);
        if offset + size > self.instance_buffer_capacity {
            return None;
        }
        let resources = self.resources();
        resources
            .queue
            .write_buffer(&resources.instance_buffer, offset, data);
        *instance_offset = offset + size;
        Some((offset, NonZeroU64::new(size).expect("size is at least 16")))
    }

    fn instance_binding(&self, offset: u64, size: NonZeroU64) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.resources().instance_buffer,
            offset,
            size: Some(size),
        })
    }
}

#[cfg(not(target_family = "wasm"))]
fn create_surface(
    instance: &wgpu::Instance,
    raw_window_handle: raw_window_handle::RawWindowHandle,
) -> anyhow::Result<wgpu::Surface<'static>> {
    unsafe {
        instance
            .create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                // Fall back to the display handle already provided via InstanceDescriptor::display.
                raw_display_handle: None,
                raw_window_handle,
            })
            .map_err(|e| anyhow::anyhow!("{e}"))
    }
}

/// Native surface wrapper around [`WgpuSceneRenderer`].
///
/// This type owns surface acquisition, configuration, submission, presentation, and native GPU
/// recovery. The reusable scene renderer below it only owns GPUI GPU resources and command
/// encoding.
pub struct WgpuRenderer {
    context: Option<GpuContext>,
    compositor_gpu: Option<CompositorGpuHint>,
    extra_requirements: Option<WgpuDeviceRequirements>,
    scene: WgpuSceneRenderer,
    surface: Option<wgpu::Surface<'static>>,
    surface_config: wgpu::SurfaceConfiguration,
    transparent_alpha_mode: wgpu::CompositeAlphaMode,
    opaque_alpha_mode: wgpu::CompositeAlphaMode,
    last_error: Arc<Mutex<Option<String>>>,
    failed_frame_count: u32,
    device_lost: Arc<std::sync::atomic::AtomicBool>,
    surface_configured: bool,
    needs_redraw: bool,
}

impl WgpuRenderer {
    /// Creates a native surface-backed renderer from raw window handles.
    #[cfg(not(target_family = "wasm"))]
    pub fn new<W>(
        gpu_context: GpuContext,
        window: &W,
        config: WgpuSurfaceConfig,
        compositor_gpu: Option<CompositorGpuHint>,
        extra_requirements: Option<WgpuDeviceRequirements>,
    ) -> anyhow::Result<Self>
    where
        W: HasWindowHandle + HasDisplayHandle + std::fmt::Debug + Send + Sync + Clone + 'static,
    {
        let window_handle = window
            .window_handle()
            .map_err(|e| anyhow::anyhow!("Failed to get window handle: {e}"))?;
        let target = wgpu::SurfaceTargetUnsafe::RawHandle {
            raw_display_handle: None,
            raw_window_handle: window_handle.as_raw(),
        };
        let instance = gpu_context
            .borrow()
            .as_ref()
            .map(|ctx| ctx.instance.clone())
            .unwrap_or_else(|| WgpuContext::instance(Box::new(window.clone())));
        let surface = unsafe {
            instance
                .create_surface_unsafe(target)
                .map_err(|e| anyhow::anyhow!("Failed to create surface: {e}"))?
        };
        let mut ctx_ref = gpu_context.borrow_mut();
        let context = match ctx_ref.as_mut() {
            Some(context) => {
                context.check_compatible_with_surface(&surface)?;
                context
            }
            None => ctx_ref.insert(WgpuContext::new(
                instance,
                &surface,
                compositor_gpu,
                extra_requirements.as_ref(),
            )?),
        };
        let atlas = Arc::new(WgpuAtlas::from_context(context));
        Self::new_internal(
            Some(Rc::clone(&gpu_context)),
            context,
            surface,
            config,
            compositor_gpu,
            extra_requirements,
            atlas,
        )
    }

    /// Creates a surface-backed renderer for a web canvas.
    #[cfg(target_family = "wasm")]
    pub fn new_from_canvas(
        context: &WgpuContext,
        canvas: &web_sys::HtmlCanvasElement,
        config: WgpuSurfaceConfig,
    ) -> anyhow::Result<Self> {
        let surface = context
            .instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .map_err(|e| anyhow::anyhow!("Failed to create surface: {e}"))?;
        let atlas = Arc::new(WgpuAtlas::from_context(context));
        Self::new_internal(None, context, surface, config, None, None, atlas)
    }

    fn new_internal(
        context_handle: Option<GpuContext>,
        context: &WgpuContext,
        surface: wgpu::Surface<'static>,
        config: WgpuSurfaceConfig,
        compositor_gpu: Option<CompositorGpuHint>,
        extra_requirements: Option<WgpuDeviceRequirements>,
        atlas: Arc<WgpuAtlas>,
    ) -> anyhow::Result<Self> {
        let surface_caps = surface.get_capabilities(&context.adapter);
        let preferred_formats = [
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Rgba8Unorm,
        ];
        let surface_format = preferred_formats
            .iter()
            .find(|format| surface_caps.formats.contains(format))
            .copied()
            .or_else(|| {
                surface_caps
                    .formats
                    .iter()
                    .find(|format| !format.is_srgb())
                    .copied()
            })
            .or_else(|| surface_caps.formats.first().copied())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Surface reports no supported texture formats for adapter {:?}",
                    context.adapter.get_info().name
                )
            })?;
        let pick_alpha_mode =
            |preferences: &[wgpu::CompositeAlphaMode]| -> anyhow::Result<wgpu::CompositeAlphaMode> {
                preferences
                    .iter()
                    .find(|mode| surface_caps.alpha_modes.contains(mode))
                    .copied()
                    .or_else(|| surface_caps.alpha_modes.first().copied())
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "Surface reports no supported alpha modes for adapter {:?}",
                            context.adapter.get_info().name
                        )
                    })
            };
        let transparent_alpha_mode = pick_alpha_mode(&[
            wgpu::CompositeAlphaMode::PreMultiplied,
            wgpu::CompositeAlphaMode::Inherit,
        ])?;
        let opaque_alpha_mode = pick_alpha_mode(&[
            wgpu::CompositeAlphaMode::Opaque,
            wgpu::CompositeAlphaMode::Inherit,
        ])?;
        let alpha_mode = if config.transparent {
            transparent_alpha_mode
        } else {
            opaque_alpha_mode
        };
        let max_texture_size = context.device.limits().max_texture_dimension_2d;
        let requested_width = config.size.width.0.max(0) as u32;
        let requested_height = config.size.height.0.max(0) as u32;
        let width = requested_width.min(max_texture_size).max(1);
        let height = requested_height.min(max_texture_size).max(1);
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: config
                .preferred_present_mode
                .filter(|mode| surface_caps.present_modes.contains(mode))
                .unwrap_or(wgpu::PresentMode::Fifo),
            desired_maximum_frame_latency: 2,
            alpha_mode,
            color_space: wgpu::SurfaceColorSpace::Auto,
            view_formats: vec![],
        };
        surface.configure(&context.device, &surface_config);
        let scene = WgpuSceneRenderer::from_parts(
            &context.adapter,
            context.device.clone(),
            context.queue.clone(),
            WgpuSceneRendererConfig {
                size: Size {
                    width: DevicePixels(width as i32),
                    height: DevicePixels(height as i32),
                },
                format: surface_format,
                transparent: alpha_mode == wgpu::CompositeAlphaMode::PreMultiplied,
            },
            atlas,
        )?;
        let last_error = Arc::new(Mutex::new(None));
        scene.install_error_handler(last_error.clone());
        Ok(Self {
            context: context_handle,
            compositor_gpu,
            extra_requirements,
            scene,
            surface: Some(surface),
            surface_config,
            transparent_alpha_mode,
            opaque_alpha_mode,
            last_error,
            failed_frame_count: 0,
            device_lost: context.device_lost_flag(),
            surface_configured: true,
            needs_redraw: false,
        })
    }

    /// Encodes the scene into the current surface and submits/presents it.
    pub fn draw(&mut self, scene: &Scene) -> bool {
        if !self.surface_configured {
            return false;
        }
        if let Some(error) = self.last_error.lock().unwrap().take() {
            self.failed_frame_count += 1;
            log::error!(
                "GPU error during frame (failure {} of 10): {error}",
                self.failed_frame_count
            );
            if self.failed_frame_count > 10 {
                panic!("Too many consecutive GPU errors. Last error: {error}");
            }
            if self.failed_frame_count > 5 {
                self.scene.invalidate_intermediate_textures();
                self.scene.atlas.clear();
                self.needs_redraw = true;
                self.failed_frame_count = 0;
                return false;
            }
        } else {
            self.failed_frame_count = 0;
        }

        let frame = match self
            .surface
            .as_ref()
            .expect("GPU surface not available")
            .get_current_texture()
        {
            wgpu::CurrentSurfaceTexture::Success(frame) => frame,
            wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                drop(frame);
                self.configure_surface();
                return false;
            }
            wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                self.configure_surface();
                return false;
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return false;
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                *self.last_error.lock().unwrap() = Some("Surface texture validation error".into());
                return false;
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let (device, queue) = self.scene.gpu_context();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("main_encoder"),
        });
        let target = WgpuRenderTarget {
            view: &view,
            size: Size {
                width: DevicePixels(self.surface_config.width as i32),
                height: DevicePixels(self.surface_config.height as i32),
            },
            format: self.surface_config.format,
            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        };
        let result = self.scene.encode_native(scene, &mut encoder, target);
        match result {
            Ok(_) => {
                queue.submit(Some(encoder.finish()));
                queue.present(frame);
                true
            }
            Err(error) => {
                log::error!("failed to encode GPUI scene: {error:#}");
                queue.present(frame);
                false
            }
        }
    }

    fn configure_surface(&self) {
        if let (Some(surface), Some(resources)) = (&self.surface, &self.scene.resources) {
            surface.configure(&resources.device, &self.surface_config);
        }
    }

    /// Updates the surface and scene target dimensions.
    pub fn update_drawable_size(&mut self, size: Size<DevicePixels>) {
        let width = size.width.0.max(0) as u32;
        let height = size.height.0.max(0) as u32;
        let width = width.min(self.scene.max_texture_size()).max(1);
        let height = height.min(self.scene.max_texture_size()).max(1);
        if self.surface_config.width == width && self.surface_config.height == height {
            return;
        }
        self.surface_config.width = width;
        self.surface_config.height = height;
        self.scene.update_drawable_size(Size {
            width: DevicePixels(width as i32),
            height: DevicePixels(height as i32),
        });
        self.configure_surface();
    }

    /// Updates surface alpha mode and rebuilds the scene pipelines.
    pub fn update_transparency(&mut self, transparent: bool) {
        let alpha_mode = if transparent {
            self.transparent_alpha_mode
        } else {
            self.opaque_alpha_mode
        };
        if self.surface_config.alpha_mode == alpha_mode {
            return;
        }
        self.surface_config.alpha_mode = alpha_mode;
        self.scene
            .update_transparency(alpha_mode == wgpu::CompositeAlphaMode::PreMultiplied);
        self.configure_surface();
    }

    /// Returns the GPUI atlas shared with the platform window.
    pub fn sprite_atlas(&self) -> &Arc<WgpuAtlas> {
        self.scene.sprite_atlas()
    }

    /// Returns GPU information for the platform layer.
    pub fn gpu_specs(&self) -> GpuSpecs {
        self.scene.gpu_specs()
    }

    /// Returns the device and queue used by the native renderer.
    pub fn gpu_context(&self) -> (Arc<wgpu::Device>, Arc<wgpu::Queue>) {
        self.scene.gpu_context()
    }

    /// Returns the largest supported 2D texture dimension.
    pub fn max_texture_size(&self) -> u32 {
        self.scene.max_texture_size()
    }

    /// Selects the subpixel text channel layout used by the native target.
    pub fn set_subpixel_layout(&mut self, is_bgr: bool) {
        self.scene.set_subpixel_layout(is_bgr);
    }

    /// Returns whether dual-source text blending is enabled.
    pub fn supports_dual_source_blending(&self) -> bool {
        self.scene.supports_dual_source_blending()
    }

    /// Marks the native surface unavailable until it is replaced.
    pub fn unconfigure_surface(&mut self) {
        self.surface_configured = false;
        self.scene.invalidate_intermediate_textures();
    }

    /// Replaces the native surface while preserving the scene renderer and atlas.
    #[cfg(not(target_family = "wasm"))]
    pub fn replace_surface<W: HasWindowHandle>(
        &mut self,
        window: &W,
        config: WgpuSurfaceConfig,
        instance: &wgpu::Instance,
    ) -> anyhow::Result<()> {
        let handle = window
            .window_handle()
            .map_err(|e| anyhow::anyhow!("Failed to get window handle: {e}"))?;
        let surface = create_surface(instance, handle.as_raw())?;
        let alpha_mode = if config.transparent {
            self.transparent_alpha_mode
        } else {
            self.opaque_alpha_mode
        };
        let width = config.size.width.0.max(0) as u32;
        let height = config.size.height.0.max(0) as u32;
        self.surface_config.width = width.min(self.scene.max_texture_size()).max(1);
        self.surface_config.height = height.min(self.scene.max_texture_size()).max(1);
        self.surface_config.alpha_mode = alpha_mode;
        if let Some(mode) = config.preferred_present_mode {
            self.surface_config.present_mode = mode;
        }
        self.surface = Some(surface);
        self.scene
            .update_transparency(alpha_mode == wgpu::CompositeAlphaMode::PreMultiplied);
        self.scene.update_drawable_size(Size {
            width: DevicePixels(self.surface_config.width as i32),
            height: DevicePixels(self.surface_config.height as i32),
        });
        self.configure_surface();
        self.surface_configured = true;
        Ok(())
    }

    /// Releases GPU resources before destroying the native window.
    pub fn destroy(&mut self) {
        self.scene.resources = None;
        self.surface = None;
    }

    /// Returns whether the native renderer requested another frame after an error.
    pub fn needs_redraw(&mut self) -> bool {
        std::mem::take(&mut self.needs_redraw)
    }

    /// Returns whether the shared native GPU context reported device loss.
    pub fn device_lost(&self) -> bool {
        self.device_lost.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Recovers the native device and surface after device loss.
    #[cfg(not(target_family = "wasm"))]
    pub fn recover<W>(&mut self, window: &W) -> anyhow::Result<()>
    where
        W: HasWindowHandle + HasDisplayHandle + std::fmt::Debug + Send + Sync + Clone + 'static,
    {
        let gpu_context = self.context.as_ref().expect("recover requires gpu_context");
        let needs_new_context = gpu_context
            .borrow()
            .as_ref()
            .is_none_or(|context| context.device_lost());
        let window_handle = window
            .window_handle()
            .map_err(|e| anyhow::anyhow!("Failed to get window handle: {e}"))?;
        let surface = if needs_new_context {
            log::warn!("GPU device lost, recreating context...");
            self.scene.resources = None;
            *gpu_context.borrow_mut() = None;
            std::thread::sleep(std::time::Duration::from_millis(350));
            let instance = WgpuContext::instance(Box::new(window.clone()));
            let surface = create_surface(&instance, window_handle.as_raw())?;
            let new_context = WgpuContext::new_rejecting_software(
                instance,
                &surface,
                self.compositor_gpu,
                self.extra_requirements.as_ref(),
            )?;
            *gpu_context.borrow_mut() = Some(new_context);
            surface
        } else {
            let context_ref = gpu_context.borrow();
            let instance = &context_ref.as_ref().unwrap().instance;
            create_surface(instance, window_handle.as_raw())?
        };
        let surface_config = WgpuSurfaceConfig {
            size: Size {
                width: DevicePixels(self.surface_config.width as i32),
                height: DevicePixels(self.surface_config.height as i32),
            },
            transparent: self.surface_config.alpha_mode != wgpu::CompositeAlphaMode::Opaque,
            preferred_present_mode: Some(self.surface_config.present_mode),
        };
        let context_ref = gpu_context.borrow();
        let context = context_ref.as_ref().expect("context should exist");
        let atlas = self.scene.atlas.clone();
        let scene = WgpuSceneRenderer::from_parts(
            &context.adapter,
            context.device.clone(),
            context.queue.clone(),
            WgpuSceneRendererConfig {
                size: surface_config.size,
                format: self.surface_config.format,
                transparent: surface_config.transparent,
            },
            atlas.clone(),
        )?;
        atlas.replace_gpu_context(
            context.device.clone(),
            context.queue.clone(),
            self.surface_config.format,
        );
        scene.install_error_handler(self.last_error.clone());
        self.scene = scene;
        self.surface = Some(surface);
        self.configure_surface();
        self.device_lost = context.device_lost_flag();
        Ok(())
    }
}

struct RenderingParameters {
    path_sample_count: u32,
    gamma_ratios: [f32; 4],
    grayscale_enhanced_contrast: f32,
    subpixel_enhanced_contrast: f32,
}

impl RenderingParameters {
    fn new(adapter: &wgpu::Adapter, surface_format: wgpu::TextureFormat) -> Self {
        use std::env;

        let format_features = adapter.get_texture_format_features(surface_format);
        let path_sample_count = [4, 2, 1]
            .into_iter()
            .find(|&n| format_features.flags.sample_count_supported(n))
            .unwrap_or(1);

        let gamma = env::var("ZED_FONTS_GAMMA")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.8_f32)
            .clamp(1.0, 2.2);
        let gamma_ratios = get_gamma_correction_ratios(gamma);

        let grayscale_enhanced_contrast = env::var("ZED_FONTS_GRAYSCALE_ENHANCED_CONTRAST")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.0_f32)
            .max(0.0);

        let subpixel_enhanced_contrast = env::var("ZED_FONTS_SUBPIXEL_ENHANCED_CONTRAST")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.5_f32)
            .max(0.0);

        Self {
            path_sample_count,
            gamma_ratios,
            grayscale_enhanced_contrast,
            subpixel_enhanced_contrast,
        }
    }
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::*;
    use gpui::block_on;

    fn test_external_scene(ids: &[ExternalSurfaceId]) -> Scene {
        let bounds = Bounds {
            origin: Point {
                x: ScaledPixels(0.0),
                y: ScaledPixels(0.0),
            },
            size: Size {
                width: ScaledPixels(16.0),
                height: ScaledPixels(16.0),
            },
        };
        let mut scene = Scene::default();
        for &id in ids {
            scene.insert_primitive(ExternalSurface {
                order: 0,
                bounds,
                content_mask: gpui::ContentMask { bounds },
                id,
            });
        }
        scene.finish();
        scene
    }

    fn test_external_view(device: &wgpu::Device) -> Arc<wgpu::TextureView> {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("external_surface_recovery_test_view"),
            size: wgpu::Extent3d {
                width: 16,
                height: 16,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        Arc::new(texture.create_view(&wgpu::TextureViewDescriptor::default()))
    }

    fn encode_external_scene(
        renderer: &mut WgpuSceneRenderer,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &Scene,
        registry: &ExternalSurfaceRegistry<WgpuExternalSurface>,
    ) -> anyhow::Result<()> {
        let target_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("external_surface_recovery_test_target"),
            size: wgpu::Extent3d {
                width: 16,
                height: 16,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let target_view = target_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("external_surface_recovery_test_encoder"),
        });
        renderer.encode_with_external_surfaces(
            scene,
            &mut encoder,
            WgpuRenderTarget {
                view: &target_view,
                size: Size {
                    width: DevicePixels(16),
                    height: DevicePixels(16),
                },
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            },
            registry,
        )?;
        queue.submit(Some(encoder.finish()));
        Ok(())
    }

    #[test]
    fn replacement_preserves_atlas_identity_and_renderer_format() -> anyhow::Result<()> {
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
        .map_err(|error| anyhow::anyhow!("failed to request adapter: {error}"))?;
        let requirements = WgpuSceneRenderer::requirements(&adapter);
        let (device, queue) = block_on(
            adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("wgpu_renderer_recovery_test"),
                required_features: requirements.features,
                required_limits: wgpu::Limits::downlevel_defaults()
                    .using_resolution(adapter.limits())
                    .using_alignment(adapter.limits()),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            }),
        )
        .map_err(|error| anyhow::anyhow!("failed to request device: {error}"))?;
        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let mut renderer = WgpuSceneRenderer::from_host(
            &adapter,
            device.clone(),
            queue.clone(),
            WgpuSceneRendererConfig {
                size: Size {
                    width: DevicePixels(64),
                    height: DevicePixels(64),
                },
                format,
                transparent: false,
            },
        )?;
        let atlas = renderer.sprite_atlas().clone();
        renderer.replace_gpu_context(&adapter, device, queue, None)?;

        assert!(Arc::ptr_eq(&atlas, renderer.sprite_atlas()));
        assert_eq!(renderer.target_format(), format);
        Ok(())
    }

    #[test]
    fn external_surface_recovery_tracks_each_id_until_replaced() -> anyhow::Result<()> {
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
        .map_err(|error| anyhow::anyhow!("failed to request adapter: {error}"))?;
        let requirements = WgpuSceneRenderer::requirements(&adapter);
        let (device, queue) = block_on(
            adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("wgpu_external_surface_recovery_test"),
                required_features: requirements.features,
                required_limits: wgpu::Limits::downlevel_defaults()
                    .using_resolution(adapter.limits())
                    .using_alignment(adapter.limits()),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            }),
        )
        .map_err(|error| anyhow::anyhow!("failed to request device: {error}"))?;
        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let mut renderer = WgpuSceneRenderer::from_host(
            &adapter,
            device.clone(),
            queue.clone(),
            WgpuSceneRendererConfig {
                size: Size {
                    width: DevicePixels(16),
                    height: DevicePixels(16),
                },
                format,
                transparent: false,
            },
        )?;

        let mut registry = ExternalSurfaceRegistry::default();
        let id_a = registry.register(WgpuExternalSurface {
            view: test_external_view(&device),
        });
        let id_b = registry.register(WgpuExternalSurface {
            view: test_external_view(&device),
        });
        let scene_ab = test_external_scene(&[id_a, id_b]);
        let scene_a = test_external_scene(&[id_a]);
        let scene_b = test_external_scene(&[id_b]);

        // Both registered surfaces are valid before recovery.
        encode_external_scene(&mut renderer, &device, &queue, &scene_ab, &registry)?;

        // Recovery snapshots both generations. Replacing A must not release B's requirement.
        renderer.replace_gpu_context(&adapter, device.clone(), queue.clone(), Some(&registry))?;
        assert!(registry.replace(
            id_a,
            WgpuExternalSurface {
                view: test_external_view(&device),
            },
        ));
        encode_external_scene(&mut renderer, &device, &queue, &scene_a, &registry)?;

        let empty_scene = Scene::default();
        encode_external_scene(&mut renderer, &device, &queue, &empty_scene, &registry)?;

        // The stale B view is rejected during validation, before any command submission or GPU
        // work. The failed scene must not clear A or B state prematurely.
        let target_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("external_surface_recovery_test_stale_target"),
            size: wgpu::Extent3d {
                width: 16,
                height: 16,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let target_view = target_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut stale_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("external_surface_recovery_test_stale_encoder"),
        });
        let error = renderer
            .encode_with_external_surfaces(
                &scene_b,
                &mut stale_encoder,
                WgpuRenderTarget {
                    view: &target_view,
                    size: Size {
                        width: DevicePixels(16),
                        height: DevicePixels(16),
                    },
                    format,
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                },
                &registry,
            )
            .expect_err("stale external surface B should be rejected after recovery");
        assert!(
            error
                .to_string()
                .contains("external surface ID 2 must be replaced after GPU recovery")
        );

        assert!(registry.replace(
            id_b,
            WgpuExternalSurface {
                view: test_external_view(&device),
            },
        ));
        encode_external_scene(&mut renderer, &device, &queue, &scene_b, &registry)?;
        Ok(())
    }
}
