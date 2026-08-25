use gpui::{DevicePixels, Scene, Size};
use gpui_wgpu::wgpu;
use std::sync::Arc;

/// A borrowed single-sample target supplied by the host.
pub type HostRenderTarget<'a> = gpui_wgpu::WgpuRenderTarget<'a>;

/// GPU objects supplied and owned by the embedding host.
#[derive(Clone)]
pub struct HostGpu {
    /// The adapter that created the host device.
    pub adapter: Arc<wgpu::Adapter>,
    /// The host's device.
    pub device: Arc<wgpu::Device>,
    /// The host's queue. The adapter never submits to it.
    pub queue: Arc<wgpu::Queue>,
    /// The format used by GPUI atlas textures and render targets.
    pub color_format: wgpu::TextureFormat,
}

impl HostGpu {
    /// Creates a host GPU description from an existing device and queue.
    pub fn new(
        adapter: Arc<wgpu::Adapter>,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        color_format: wgpu::TextureFormat,
    ) -> Self {
        Self {
            adapter,
            device,
            queue,
            color_format,
        }
    }

    /// Creates the shared surface-free GPUI scene renderer.
    pub fn scene_renderer(
        &self,
        size: Size<DevicePixels>,
        transparent: bool,
    ) -> gpui::Result<gpui_wgpu::WgpuSceneRenderer> {
        gpui_wgpu::WgpuSceneRenderer::from_host(
            &self.adapter,
            self.device.clone(),
            self.queue.clone(),
            gpui_wgpu::WgpuSceneRendererConfig {
                size,
                format: self.color_format,
                transparent,
            },
        )
    }
}

/// A host-owned offscreen WGPU color target for headless proofs and simple integrations.
pub struct OffscreenTarget {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    size: Size<DevicePixels>,
    format: wgpu::TextureFormat,
}

impl OffscreenTarget {
    /// Allocates an offscreen target. Allocation does not submit or present.
    pub fn new(gpu: &HostGpu, size: Size<DevicePixels>) -> Self {
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gpui_embed_render_target"),
            size: wgpu::Extent3d {
                width: size.width.0.max(1) as u32,
                height: size.height.0.max(1) as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: gpu.color_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            texture,
            view,
            size,
            format: gpu.color_format,
        }
    }

    /// Returns the target texture view for host-owned encoding.
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// Returns the target texture.
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    /// Returns the target size in device pixels.
    pub fn size(&self) -> Size<DevicePixels> {
        self.size
    }

    /// Returns the target format.
    pub fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    /// Encodes a rendered GPUI scene into a host command encoder.
    pub fn target(&self, load: wgpu::LoadOp<wgpu::Color>) -> HostRenderTarget<'_> {
        HostRenderTarget {
            view: &self.view,
            size: self.size,
            format: self.format,
            load,
        }
    }

    /// Encodes a rendered GPUI scene into this convenience target.
    pub fn encode_scene(
        &self,
        renderer: &mut gpui_wgpu::WgpuSceneRenderer,
        scene: &Scene,
        encoder: &mut wgpu::CommandEncoder,
    ) -> gpui::Result<gpui_wgpu::RenderStats> {
        renderer.encode(
            scene,
            encoder,
            self.target(wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)),
        )
    }
}
