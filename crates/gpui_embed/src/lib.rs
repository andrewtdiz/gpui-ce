//! A small, surface-free host adapter for driving GPUI from an embedding run loop.
//!
//! The adapter owns GPUI's virtual window and dispatcher, while the host owns
//! the GPU command encoder and decides when to submit it.  No native window
//! handles, surfaces, submissions, or presentation are used here.
//!
//! A host frame follows this order:
//!
//! ```rust,ignore
//! let input = ui.dispatch_input(platform_input)?;
//! if input.propagate && !input.pointer_capture {
//!     route_to_engine_viewport(platform_input, viewport_bounds);
//! }
//!
//! let poll = ui.poll();
//! let frame = ui.prepare_frame()?;
//! update_viewport_texture_from_layout();
//! render_engine_viewport(&mut encoder);
//! ui.encode_with_external_surfaces(
//!     HostRenderTarget {
//!         view: &frame_view,
//!         size: drawable_size,
//!         format: surface_format,
//!         load: wgpu::LoadOp::Load,
//!     },
//!     &mut encoder,
//!     &external_registry,
//! )?;
//! queue.submit([encoder.finish()]);
//! surface_frame.present();
//! ui.mark_presented(frame.scene_generation)?;
//! schedule_next_wake(poll.next_deadline);
//! apply_commands(ui.take_host_commands());
//! ```
//!
//! `HostRenderTarget` is a borrowed, single-sample, positive-size target in the configured
//! format. GPUI always stores its output. Skip encoding while the host is minimized. After GPU
//! recovery, recreate host targets and replace registered external views under their stable IDs
//! before encoding again.

#![warn(missing_docs)]

mod dispatcher;
mod platform;
mod renderer;

pub use dispatcher::PollOutcome;
pub use gpui::FrameStatus;
pub use platform::{
    EmbeddedConfig, EmbeddedGpui, HostCommand, HostServices, InputOutcome, RenderedScene,
    WindowMetrics,
};
pub use renderer::{HostGpu, HostRenderTarget, OffscreenTarget};
