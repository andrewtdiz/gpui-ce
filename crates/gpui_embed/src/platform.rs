use crate::{HostGpu, HostRenderTarget, OffscreenTarget, dispatcher::EmbeddedDispatcher};
use gpui::{
    AnyWindowHandle, AppContext, AssetSource, BackgroundExecutor, Bounds, Capslock, ClipboardItem,
    CursorStyle, DevicePixels, DispatchEventResult, DisplayId, DummyKeyboardMapper,
    ForegroundExecutor, Keymap, MacActivationPolicy, Modifiers, PathPromptOptions, Pixels,
    Platform, PlatformAtlas, PlatformDisplay, PlatformInput, PlatformInputHandler,
    PlatformKeyboardLayout, PlatformKeyboardMapper, PlatformTextSystem, PlatformWindow, Point,
    PromptButton, PromptLevel, Render, RequestFrameOptions, Scene, SharedString, Size, Task,
    ThermalState, Window, WindowAppearance, WindowBackgroundAppearance, WindowBounds,
    WindowControlArea, WindowDecorations, WindowInsets, WindowKind, WindowOptions, WindowParams,
    div,
};
use gpui_wgpu::wgpu::rwh::{HandleError, HasDisplayHandle, HasWindowHandle};
use gpui_wgpu::{CosmicTextSystem, WgpuAtlas};
use std::{
    borrow::Cow,
    cell::{Cell, RefCell},
    collections::VecDeque,
    path::{Path, PathBuf},
    rc::Rc,
    sync::{Arc, Mutex},
};
use uuid::Uuid;

/// The logical and device geometry supplied by the host.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindowMetrics {
    /// Window bounds in logical GPUI pixels.
    pub bounds: Bounds<Pixels>,
    /// Logical-to-device scale factor.
    pub scale_factor: f32,
    /// Whether the host considers the virtual window active.
    pub active: bool,
    /// Whether the host considers the virtual window hovered.
    pub hovered: bool,
    /// The host's current appearance preference.
    pub appearance: WindowAppearance,
}

/// Host callbacks for services that GPUI cannot provide without an OS window.
#[derive(Clone, Default)]
pub struct HostServices {
    /// Reads the current clipboard contents synchronously.
    pub read_clipboard: Option<Arc<dyn Fn() -> Option<ClipboardItem> + Send + Sync>>,
    /// Replaces the current clipboard contents.
    pub write_clipboard: Option<Arc<dyn Fn(ClipboardItem) + Send + Sync>>,
}

impl WindowMetrics {
    /// Creates metrics for a window whose origin is `(0, 0)`.
    pub fn new(size: Size<Pixels>, scale_factor: f32) -> Self {
        Self {
            bounds: Bounds::new(gpui::point(gpui::px(0.), gpui::px(0.)), size),
            scale_factor: scale_factor.max(f32::EPSILON),
            active: true,
            hovered: false,
            appearance: WindowAppearance::Light,
        }
    }

    /// Returns the backing size in device pixels.
    pub fn device_size(self) -> Size<DevicePixels> {
        self.bounds.size.to_device_pixels(self.scale_factor)
    }
}

impl Default for WindowMetrics {
    fn default() -> Self {
        Self::new(gpui::size(gpui::px(800.), gpui::px(600.)), 1.0)
    }
}

/// Configuration for an embedded GPUI instance.
#[derive(Clone)]
pub struct EmbeddedConfig {
    /// Host-owned GPU objects used for the atlas and render targets.
    pub gpu: HostGpu,
    /// Initial virtual-window metrics.
    pub metrics: WindowMetrics,
    /// Host callbacks for clipboard and other optional services.
    pub services: HostServices,
    /// Optional callback used to wake the host run loop when GPUI queues work.
    pub wake: Option<Arc<dyn Fn() + Send + Sync>>,
    /// Optional host-owned source for GPUI and component SVG assets.
    pub assets: Option<Arc<dyn AssetSource>>,
}

impl EmbeddedConfig {
    /// Creates a configuration with default window metrics.
    pub fn new(gpu: HostGpu) -> Self {
        Self {
            gpu,
            metrics: WindowMetrics::default(),
            services: HostServices::default(),
            wake: None,
            assets: None,
        }
    }

    /// Replaces the initial window metrics.
    pub fn with_metrics(mut self, metrics: WindowMetrics) -> Self {
        self.metrics = metrics;
        self
    }

    /// Installs a callback that wakes the host run loop when work is queued.
    pub fn with_wake(mut self, wake: impl Fn() + Send + Sync + 'static) -> Self {
        self.wake = Some(Arc::new(wake));
        self
    }

    /// Installs a host-owned source for GPUI and component assets.
    pub fn with_assets(mut self, assets: impl AssetSource) -> Self {
        self.assets = Some(Arc::new(assets));
        self
    }

    /// Installs synchronous clipboard callbacks supplied by the host.
    pub fn with_clipboard(
        mut self,
        read: impl Fn() -> Option<ClipboardItem> + Send + Sync + 'static,
        write: impl Fn(ClipboardItem) + Send + Sync + 'static,
    ) -> Self {
        self.services.read_clipboard = Some(Arc::new(read));
        self.services.write_clipboard = Some(Arc::new(write));
        self
    }
}

struct SharedAssetSource(Arc<dyn AssetSource>);

impl AssetSource for SharedAssetSource {
    fn load(&self, path: &str) -> gpui::Result<Option<Cow<'static, [u8]>>> {
        self.0.load(path)
    }

    fn list(&self, path: &str) -> gpui::Result<Vec<SharedString>> {
        self.0.list(path)
    }
}

/// Commands emitted by GPUI for the embedding host to apply.
#[derive(Clone, Debug, PartialEq)]
pub enum HostCommand {
    /// Set the host cursor style for the embedded window.
    SetCursor(CursorStyle),
    /// Move the host IME candidate rectangle.
    SetImePosition(Bounds<Pixels>),
}

/// The result of dispatching one input event to the virtual window.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct InputOutcome {
    /// Whether the host should continue routing the original event.
    pub propagate: bool,
    /// Whether a GPUI handler prevented the event's default action.
    pub default_prevented: bool,
    /// Whether GPUI invalidated the virtual window and needs another encoded frame.
    pub redraw_requested: bool,
    /// Whether GPUI retained pointer capture for this event stream.
    pub pointer_capture: bool,
}

/// A description of the last scene prepared for the host.
///
/// GPUI keeps the scene's storage inside its window. The scene itself must be
/// encoded during the host's frame callback; this value is stable metadata for checking whether a
/// frame is available after that callback.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderedScene {
    /// Retained scene generation used for presentation acknowledgements.
    pub frame_id: u64,
    /// Number of paint operations in the scene.
    pub primitive_count: usize,
    /// Metrics used for the frame.
    pub metrics: WindowMetrics,
}

/// The virtual display exposed to GPUI.
#[derive(Debug)]
struct EmbeddedDisplay {
    bounds: Mutex<Bounds<Pixels>>,
}

impl EmbeddedDisplay {
    fn new(bounds: Bounds<Pixels>) -> Self {
        Self {
            bounds: Mutex::new(bounds),
        }
    }

    fn set_bounds(&self, bounds: Bounds<Pixels>) {
        *self.bounds.lock().expect("embedded display poisoned") = bounds;
    }
}

impl PlatformDisplay for EmbeddedDisplay {
    fn id(&self) -> DisplayId {
        DisplayId::new(1)
    }

    fn uuid(&self) -> gpui::Result<Uuid> {
        Ok(Uuid::from_u128(1))
    }

    fn bounds(&self) -> Bounds<Pixels> {
        *self.bounds.lock().expect("embedded display poisoned")
    }
}

struct EmbeddedWindowState {
    handle: AnyWindowHandle,
    metrics: WindowMetrics,
    display: Rc<EmbeddedDisplay>,
    atlas: Arc<dyn PlatformAtlas>,
    host_commands: Arc<Mutex<VecDeque<HostCommand>>>,
    input_handler: Option<PlatformInputHandler>,
    input_callback: Option<Box<dyn FnMut(PlatformInput) -> DispatchEventResult>>,
    request_frame_callback: Option<Box<dyn FnMut(RequestFrameOptions)>>,
    resize_callback: Option<ResizeCallback>,
    active_callback: Option<Box<dyn FnMut(bool)>>,
    hover_callback: Option<Box<dyn FnMut(bool)>>,
    moved_callback: Option<Box<dyn FnMut()>>,
    should_close_callback: Option<Box<dyn FnMut() -> bool>>,
    hit_test_callback: Option<Box<dyn FnMut() -> Option<WindowControlArea>>>,
    close_callback: Option<Box<dyn FnOnce()>>,
    appearance_callback: Option<Box<dyn FnMut()>>,
    title: String,
    background: WindowBackgroundAppearance,
    appearance: WindowAppearance,
    active: bool,
    hovered: bool,
    mouse_position: Point<Pixels>,
    modifiers: Modifiers,
    capslock: Capslock,
    fullscreen: bool,
    rendered_scene: Option<RenderedScene>,
}

type ResizeCallback = Box<dyn FnMut(Size<Pixels>, f32)>;

/// The single virtual GPUI window owned by [`EmbeddedGpui`].
#[derive(Clone)]
struct EmbeddedWindow {
    state: Rc<Mutex<EmbeddedWindowState>>,
}

impl EmbeddedWindow {
    fn new(
        handle: AnyWindowHandle,
        metrics: WindowMetrics,
        display: Rc<EmbeddedDisplay>,
        atlas: Arc<WgpuAtlas>,
        host_commands: Arc<Mutex<VecDeque<HostCommand>>>,
    ) -> Self {
        Self {
            state: Rc::new(Mutex::new(EmbeddedWindowState {
                handle,
                metrics,
                display,
                atlas,
                host_commands,
                input_handler: None,
                input_callback: None,
                request_frame_callback: None,
                resize_callback: None,
                active_callback: None,
                hover_callback: None,
                moved_callback: None,
                should_close_callback: None,
                hit_test_callback: None,
                close_callback: None,
                appearance_callback: None,
                title: String::new(),
                background: WindowBackgroundAppearance::Opaque,
                appearance: metrics.appearance,
                active: metrics.active,
                hovered: metrics.hovered,
                mouse_position: Point::default(),
                modifiers: Modifiers::default(),
                capslock: Capslock::default(),
                fullscreen: false,
                rendered_scene: None,
            })),
        }
    }

    /// Returns the GPUI handle for this virtual window.
    fn handle(&self) -> AnyWindowHandle {
        self.state.lock().expect("embedded window poisoned").handle
    }

    /// Updates metrics and invokes GPUI's resize callback.
    fn set_metrics(&self, metrics: WindowMetrics) {
        let (resize_callback, active_callback, hover_callback, appearance_callback) = {
            let mut state = self.state.lock().expect("embedded window poisoned");
            state.metrics = metrics;
            state.display.set_bounds(metrics.bounds);
            state.active = metrics.active;
            state.hovered = metrics.hovered;
            state.appearance = metrics.appearance;
            (
                state.resize_callback.take(),
                state.active_callback.take(),
                state.hover_callback.take(),
                state.appearance_callback.take(),
            )
        };
        if let Some(mut callback) = resize_callback {
            callback(metrics.bounds.size, metrics.scale_factor);
            self.state
                .lock()
                .expect("embedded window poisoned")
                .resize_callback = Some(callback);
        }
        if let Some(mut callback) = active_callback {
            callback(metrics.active);
            self.state
                .lock()
                .expect("embedded window poisoned")
                .active_callback = Some(callback);
        }
        if let Some(mut callback) = hover_callback {
            callback(metrics.hovered);
            self.state
                .lock()
                .expect("embedded window poisoned")
                .hover_callback = Some(callback);
        }
        if let Some(mut callback) = appearance_callback {
            callback();
            self.state
                .lock()
                .expect("embedded window poisoned")
                .appearance_callback = Some(callback);
        }
    }

    /// Dispatches host input to GPUI.
    fn dispatch_input(&self, input: PlatformInput) -> gpui::Result<DispatchEventResult> {
        self.update_input_state(&input);
        let callback = {
            self.state
                .lock()
                .expect("embedded window poisoned")
                .input_callback
                .take()
        };
        let Some(mut callback) = callback else {
            return Err(std::io::Error::other("embedded window has no input callback").into());
        };
        let result = callback(input);
        self.state
            .lock()
            .expect("embedded window poisoned")
            .input_callback = Some(callback);
        Ok(result)
    }

    fn update_input_state(&self, input: &PlatformInput) {
        let mut state = self.state.lock().expect("embedded window poisoned");
        match input {
            PlatformInput::MouseDown(event) => {
                state.mouse_position = event.position;
                state.modifiers = event.modifiers;
            }
            PlatformInput::MouseUp(event) => {
                state.mouse_position = event.position;
                state.modifiers = event.modifiers;
            }
            PlatformInput::MouseMove(event) => {
                state.mouse_position = event.position;
                state.modifiers = event.modifiers;
            }
            PlatformInput::MousePressure(event) => {
                state.mouse_position = event.position;
                state.modifiers = event.modifiers;
            }
            PlatformInput::MouseExited(event) => {
                state.mouse_position = event.position;
                state.modifiers = event.modifiers;
                state.hovered = false;
            }
            PlatformInput::ScrollWheel(event) => {
                state.mouse_position = event.position;
                state.modifiers = event.modifiers;
            }
            PlatformInput::Pinch(event) => {
                state.mouse_position = event.position;
                state.modifiers = event.modifiers;
            }
            PlatformInput::ModifiersChanged(event) => {
                state.modifiers = event.modifiers;
                state.capslock = event.capslock;
            }
            PlatformInput::Touch(event) => {
                state.mouse_position = event.position;
            }
            PlatformInput::KeyDown(_) | PlatformInput::KeyUp(_) | PlatformInput::FileDrop(_) => {}
        }
    }

    fn with_input_handler<R>(&self, f: impl FnOnce(&mut PlatformInputHandler) -> R) -> Option<R> {
        let mut handler = self
            .state
            .lock()
            .expect("embedded window poisoned")
            .input_handler
            .take()?;
        let result = f(&mut handler);
        self.state
            .lock()
            .expect("embedded window poisoned")
            .input_handler = Some(handler);
        Some(result)
    }

    /// Inserts committed text into the active GPUI input handler.
    fn insert_text(&self, text: &str) -> bool {
        self.with_input_handler(|handler| handler.replace_text_in_range(None, text))
            .is_some()
    }

    /// Replaces or updates the marked IME composition.
    fn set_marked_text(
        &self,
        range_utf16: Option<std::ops::Range<usize>>,
        text: &str,
        selected_range: Option<std::ops::Range<usize>>,
    ) -> bool {
        self.with_input_handler(|handler| {
            handler.replace_and_mark_text_in_range(range_utf16, text, selected_range)
        })
        .is_some()
    }

    /// Ends the active IME composition.
    fn unmark_text(&self) -> bool {
        self.with_input_handler(|handler| handler.unmark_text())
            .is_some()
    }

    /// Returns the active UTF-16 selection, if GPUI has an input handler.
    fn selected_text_range(&self) -> Option<gpui::UTF16Selection> {
        self.with_input_handler(|handler| handler.selected_text_range(true))
            .flatten()
    }

    /// Returns the active marked-text range, if GPUI has an input handler.
    fn marked_text_range(&self) -> Option<std::ops::Range<usize>> {
        self.with_input_handler(|handler| handler.marked_text_range())
            .flatten()
    }

    /// Returns the host-space candidate bounds for the active IME composition.
    fn ime_candidate_bounds(&self) -> Option<Bounds<Pixels>> {
        self.with_input_handler(|handler| handler.ime_candidate_bounds())
            .flatten()
    }

    fn record_scene(&self, scene: &Scene, status: gpui::FrameStatus) {
        let mut state = self.state.lock().expect("embedded window poisoned");
        state.rendered_scene = Some(RenderedScene {
            frame_id: status.scene_generation,
            primitive_count: scene.len(),
            metrics: state.metrics,
        });
    }

    /// Returns metadata for the last rendered scene.
    fn rendered_scene(&self) -> Option<RenderedScene> {
        self.state
            .lock()
            .expect("embedded window poisoned")
            .rendered_scene
    }
}

impl HasWindowHandle for EmbeddedWindow {
    fn window_handle(&self) -> Result<gpui_wgpu::wgpu::rwh::WindowHandle<'_>, HandleError> {
        Err(HandleError::NotSupported)
    }
}

impl HasDisplayHandle for EmbeddedWindow {
    fn display_handle(&self) -> Result<gpui_wgpu::wgpu::rwh::DisplayHandle<'_>, HandleError> {
        Err(HandleError::NotSupported)
    }
}

impl PlatformWindow for EmbeddedWindow {
    fn bounds(&self) -> Bounds<Pixels> {
        self.state
            .lock()
            .expect("embedded window poisoned")
            .metrics
            .bounds
    }

    fn is_maximized(&self) -> bool {
        false
    }

    fn window_bounds(&self) -> WindowBounds {
        WindowBounds::Windowed(self.bounds())
    }

    fn content_size(&self) -> Size<Pixels> {
        self.bounds().size
    }

    fn resize(&mut self, size: Size<Pixels>) {
        let mut state = self.state.lock().expect("embedded window poisoned");
        state.metrics.bounds.size = size;
    }

    fn scale_factor(&self) -> f32 {
        self.state
            .lock()
            .expect("embedded window poisoned")
            .metrics
            .scale_factor
    }

    fn appearance(&self) -> WindowAppearance {
        self.state
            .lock()
            .expect("embedded window poisoned")
            .appearance
    }

    fn display(&self) -> Option<Rc<dyn PlatformDisplay>> {
        Some(
            self.state
                .lock()
                .expect("embedded window poisoned")
                .display
                .clone(),
        )
    }

    fn mouse_position(&self) -> Point<Pixels> {
        self.state
            .lock()
            .expect("embedded window poisoned")
            .mouse_position
    }
    fn modifiers(&self) -> Modifiers {
        self.state
            .lock()
            .expect("embedded window poisoned")
            .modifiers
    }
    fn capslock(&self) -> Capslock {
        self.state
            .lock()
            .expect("embedded window poisoned")
            .capslock
    }

    fn set_input_handler(&mut self, input_handler: PlatformInputHandler) {
        self.state
            .lock()
            .expect("embedded window poisoned")
            .input_handler = Some(input_handler);
    }

    fn take_input_handler(&mut self) -> Option<PlatformInputHandler> {
        self.state
            .lock()
            .expect("embedded window poisoned")
            .input_handler
            .take()
    }

    fn prompt(
        &self,
        _level: PromptLevel,
        _msg: &str,
        _detail: Option<&str>,
        _answers: &[PromptButton],
    ) -> Option<futures::channel::oneshot::Receiver<usize>> {
        None
    }

    fn activate(&self) {
        let callback = {
            let mut state = self.state.lock().expect("embedded window poisoned");
            state.active = true;
            state.active_callback.take()
        };
        if let Some(mut callback) = callback {
            callback(true);
            self.state
                .lock()
                .expect("embedded window poisoned")
                .active_callback = Some(callback);
        }
    }

    fn is_active(&self) -> bool {
        self.state.lock().expect("embedded window poisoned").active
    }
    fn is_hovered(&self) -> bool {
        self.state.lock().expect("embedded window poisoned").hovered
    }
    fn background_appearance(&self) -> WindowBackgroundAppearance {
        self.state
            .lock()
            .expect("embedded window poisoned")
            .background
    }
    fn set_title(&mut self, title: &str) {
        self.state.lock().expect("embedded window poisoned").title = title.to_owned();
    }
    fn set_background_appearance(&self, background: WindowBackgroundAppearance) {
        self.state
            .lock()
            .expect("embedded window poisoned")
            .background = background;
    }
    fn minimize(&self) {}
    fn zoom(&self) {}
    fn toggle_fullscreen(&self) {
        let mut state = self.state.lock().expect("embedded window poisoned");
        state.fullscreen = !state.fullscreen;
    }
    fn is_fullscreen(&self) -> bool {
        self.state
            .lock()
            .expect("embedded window poisoned")
            .fullscreen
    }

    fn on_request_frame(&self, callback: Box<dyn FnMut(RequestFrameOptions)>) {
        self.state
            .lock()
            .expect("embedded window poisoned")
            .request_frame_callback = Some(callback);
    }
    fn on_input(&self, callback: Box<dyn FnMut(PlatformInput) -> DispatchEventResult>) {
        self.state
            .lock()
            .expect("embedded window poisoned")
            .input_callback = Some(callback);
    }
    fn on_active_status_change(&self, callback: Box<dyn FnMut(bool)>) {
        self.state
            .lock()
            .expect("embedded window poisoned")
            .active_callback = Some(callback);
    }
    fn on_hover_status_change(&self, callback: Box<dyn FnMut(bool)>) {
        self.state
            .lock()
            .expect("embedded window poisoned")
            .hover_callback = Some(callback);
    }
    fn on_resize(&self, callback: Box<dyn FnMut(Size<Pixels>, f32)>) {
        self.state
            .lock()
            .expect("embedded window poisoned")
            .resize_callback = Some(callback);
    }
    fn on_moved(&self, callback: Box<dyn FnMut()>) {
        self.state
            .lock()
            .expect("embedded window poisoned")
            .moved_callback = Some(callback);
    }
    fn on_should_close(&self, callback: Box<dyn FnMut() -> bool>) {
        self.state
            .lock()
            .expect("embedded window poisoned")
            .should_close_callback = Some(callback);
    }
    fn on_hit_test_window_control(&self, callback: Box<dyn FnMut() -> Option<WindowControlArea>>) {
        self.state
            .lock()
            .expect("embedded window poisoned")
            .hit_test_callback = Some(callback);
    }
    fn on_close(&self, callback: Box<dyn FnOnce()>) {
        self.state
            .lock()
            .expect("embedded window poisoned")
            .close_callback = Some(callback);
    }
    fn on_appearance_changed(&self, callback: Box<dyn FnMut()>) {
        self.state
            .lock()
            .expect("embedded window poisoned")
            .appearance_callback = Some(callback);
    }

    fn draw(&self, scene: &Scene) {
        self.record_scene(scene, gpui::FrameStatus::default());
    }

    fn sprite_atlas(&self) -> Arc<dyn PlatformAtlas> {
        self.state
            .lock()
            .expect("embedded window poisoned")
            .atlas
            .clone()
    }
    fn is_subpixel_rendering_supported(&self) -> bool {
        false
    }
    fn show_character_palette(&self) {}
    fn gpu_specs(&self) -> Option<gpui::GpuSpecs> {
        None
    }
    fn update_ime_position(&self, bounds: Bounds<Pixels>) {
        self.state
            .lock()
            .expect("embedded window poisoned")
            .host_commands
            .lock()
            .expect("host command queue poisoned")
            .push_back(HostCommand::SetImePosition(bounds));
    }
    fn insets(&self) -> WindowInsets {
        WindowInsets::default()
    }
}

/// The GPUI platform implementation used by the embedded adapter.
struct EmbeddedPlatform {
    dispatcher: Arc<EmbeddedDispatcher>,
    background_executor: BackgroundExecutor,
    foreground_executor: ForegroundExecutor,
    text_system: Arc<dyn PlatformTextSystem>,
    display: Rc<EmbeddedDisplay>,
    atlas: Arc<WgpuAtlas>,
    host_commands: Arc<Mutex<VecDeque<HostCommand>>>,
    services: HostServices,
    metrics: WindowMetrics,
    window: RefCell<Option<EmbeddedWindow>>,
    active_window: RefCell<Option<AnyWindowHandle>>,
    quit: Cell<bool>,
}

impl EmbeddedPlatform {
    /// Creates an embedded platform. The caller normally uses [`EmbeddedGpui::new`].
    fn new(config: &EmbeddedConfig, atlas: Arc<WgpuAtlas>) -> Rc<Self> {
        let dispatcher = EmbeddedDispatcher::new(config.wake.clone());
        let dispatcher_for_background: Arc<dyn gpui::PlatformDispatcher> = dispatcher.clone();
        let dispatcher_for_foreground: Arc<dyn gpui::PlatformDispatcher> = dispatcher.clone();
        Rc::new(Self {
            background_executor: BackgroundExecutor::new(dispatcher_for_background),
            foreground_executor: ForegroundExecutor::new(dispatcher_for_foreground),
            text_system: Arc::new(CosmicTextSystem::new("sans-serif")),
            display: Rc::new(EmbeddedDisplay::new(config.metrics.bounds)),
            atlas,
            host_commands: Arc::new(Mutex::new(VecDeque::new())),
            services: config.services.clone(),
            metrics: config.metrics,
            window: RefCell::new(None),
            active_window: RefCell::new(None),
            quit: Cell::new(false),
            dispatcher,
        })
    }

    fn poll(&self) -> crate::PollOutcome {
        self.dispatcher.poll()
    }

    /// Returns the single virtual window, once the application has launched.
    fn window(&self) -> Option<EmbeddedWindow> {
        self.window.borrow().clone()
    }

    /// Drains host-owned cursor and IME commands.
    fn take_host_commands(&self) -> Vec<HostCommand> {
        self.host_commands
            .lock()
            .expect("host command queue poisoned")
            .drain(..)
            .collect()
    }
}

impl Platform for EmbeddedPlatform {
    fn background_executor(&self) -> BackgroundExecutor {
        self.background_executor.clone()
    }
    fn foreground_executor(&self) -> ForegroundExecutor {
        self.foreground_executor.clone()
    }
    fn text_system(&self) -> Arc<dyn PlatformTextSystem> {
        self.text_system.clone()
    }

    fn set_mac_activation_policy(&self, _policy: MacActivationPolicy) {}
    fn run(&self, on_finish_launching: Box<dyn FnOnce()>) {
        on_finish_launching();
    }
    fn quit(&self) {
        self.quit.set(true);
    }
    fn restart(&self, _binary_path: Option<PathBuf>) {}
    fn activate(&self, _ignoring_other_apps: bool) {
        if let Some(window) = self.window() {
            window.activate();
        }
    }
    fn hide(&self) {}
    fn hide_other_apps(&self) {}
    fn unhide_other_apps(&self) {}

    fn displays(&self) -> Vec<Rc<dyn PlatformDisplay>> {
        vec![self.display.clone()]
    }
    fn primary_display(&self) -> Option<Rc<dyn PlatformDisplay>> {
        Some(self.display.clone())
    }
    fn active_window(&self) -> Option<AnyWindowHandle> {
        *self.active_window.borrow()
    }

    fn open_window(
        &self,
        handle: AnyWindowHandle,
        options: WindowParams,
    ) -> gpui::Result<Box<dyn PlatformWindow>> {
        if self.window.borrow().is_some() {
            return Err(std::io::Error::other("EmbeddedGpui supports one virtual window").into());
        }
        let window = EmbeddedWindow::new(
            handle,
            WindowMetrics {
                bounds: options.bounds,
                scale_factor: self.metrics.scale_factor,
                active: self.metrics.active,
                hovered: self.metrics.hovered,
                appearance: self.metrics.appearance,
            },
            self.display.clone(),
            self.atlas.clone(),
            self.host_commands.clone(),
        );
        self.active_window.replace(Some(handle));
        self.window.replace(Some(window.clone()));
        Ok(Box::new(window))
    }

    fn window_appearance(&self) -> WindowAppearance {
        WindowAppearance::Light
    }
    fn open_url(&self, _url: &str) {}
    fn on_open_urls(&self, _callback: Box<dyn FnMut(Vec<String>)>) {}
    fn register_url_scheme(&self, _url: &str) -> Task<gpui::Result<()>> {
        Task::ready(Ok(()))
    }

    fn prompt_for_paths(
        &self,
        _options: PathPromptOptions,
    ) -> futures::channel::oneshot::Receiver<gpui::Result<Option<Vec<PathBuf>>>> {
        let (sender, receiver) = futures::channel::oneshot::channel();
        let _ = sender.send(Err(std::io::Error::other(
            "embedded platform does not provide file prompts",
        )
        .into()));
        receiver
    }
    fn prompt_for_new_path(
        &self,
        _directory: &Path,
        _suggested_name: Option<&str>,
    ) -> futures::channel::oneshot::Receiver<gpui::Result<Option<PathBuf>>> {
        let (sender, receiver) = futures::channel::oneshot::channel();
        let _ = sender.send(Err(std::io::Error::other(
            "embedded platform does not provide file prompts",
        )
        .into()));
        receiver
    }
    fn can_select_mixed_files_and_dirs(&self) -> bool {
        false
    }
    fn reveal_path(&self, _path: &Path) {}
    fn open_with_system(&self, _path: &Path) {}
    fn on_quit(&self, _callback: Box<dyn FnMut()>) {}
    fn on_reopen(&self, _callback: Box<dyn FnMut()>) {}
    fn on_system_wake(&self, _callback: Box<dyn FnMut()>) {}

    fn set_menus(&self, _menus: Vec<gpui::Menu>, _keymap: &Keymap) {}
    fn set_dock_menu(&self, _menu: Vec<gpui::MenuItem>, _keymap: &Keymap) {}
    fn on_app_menu_action(&self, _callback: Box<dyn FnMut(&dyn gpui::Action)>) {}
    fn on_will_open_app_menu(&self, _callback: Box<dyn FnMut()>) {}
    fn on_validate_app_menu_command(&self, _callback: Box<dyn FnMut(&dyn gpui::Action) -> bool>) {}
    fn thermal_state(&self) -> ThermalState {
        ThermalState::Nominal
    }
    fn on_thermal_state_change(&self, _callback: Box<dyn FnMut()>) {}

    fn app_path(&self) -> gpui::Result<PathBuf> {
        Ok(std::env::current_exe()?)
    }
    fn path_for_auxiliary_executable(&self, name: &str) -> gpui::Result<PathBuf> {
        Ok(std::env::current_exe()?.with_file_name(name))
    }
    fn set_cursor_style(&self, style: CursorStyle) {
        self.host_commands
            .lock()
            .expect("host command queue poisoned")
            .push_back(HostCommand::SetCursor(style));
    }
    fn hide_cursor_until_mouse_moves(&self) {}
    fn is_cursor_visible(&self) -> bool {
        true
    }
    fn should_auto_hide_scrollbars(&self) -> bool {
        false
    }
    fn read_from_clipboard(&self) -> Option<ClipboardItem> {
        self.services
            .read_clipboard
            .as_ref()
            .and_then(|read| read())
    }
    fn write_to_clipboard(&self, item: ClipboardItem) {
        if let Some(write) = &self.services.write_clipboard {
            write(item);
        }
    }
    #[cfg(any(target_os = "linux", target_os = "freebsd"))]
    fn read_from_primary(&self) -> Option<ClipboardItem> {
        None
    }
    #[cfg(any(target_os = "linux", target_os = "freebsd"))]
    fn write_to_primary(&self, _item: ClipboardItem) {}
    #[cfg(target_os = "macos")]
    fn read_from_find_pasteboard(&self) -> Option<ClipboardItem> {
        None
    }
    #[cfg(target_os = "macos")]
    fn write_to_find_pasteboard(&self, _item: ClipboardItem) {}
    fn write_credentials(
        &self,
        _url: &str,
        _username: &str,
        _password: &[u8],
    ) -> Task<gpui::Result<()>> {
        Task::ready(Ok(()))
    }
    fn read_credentials(&self, _url: &str) -> Task<gpui::Result<Option<(String, Vec<u8>)>>> {
        Task::ready(Ok(None))
    }
    fn delete_credentials(&self, _url: &str) -> Task<gpui::Result<()>> {
        Task::ready(Ok(()))
    }
    fn keyboard_layout(&self) -> Box<dyn PlatformKeyboardLayout> {
        Box::new(EmbeddedKeyboardLayout)
    }
    fn keyboard_mapper(&self) -> Rc<dyn PlatformKeyboardMapper> {
        Rc::new(DummyKeyboardMapper)
    }
    fn on_keyboard_layout_change(&self, _callback: Box<dyn FnMut()>) {}
}

struct EmbeddedKeyboardLayout;
impl PlatformKeyboardLayout for EmbeddedKeyboardLayout {
    fn id(&self) -> &str {
        "gpui-embedded"
    }
    fn name(&self) -> &str {
        "Embedded"
    }
}

struct EmptyRoot;
impl Render for EmptyRoot {
    fn render(
        &mut self,
        _window: &mut Window,
        _cx: &mut gpui::Context<Self>,
    ) -> impl gpui::IntoElement {
        div()
    }
}

/// The host-facing GPUI application handle.
pub struct EmbeddedGpui {
    platform: Rc<EmbeddedPlatform>,
    application: gpui::ApplicationHandle,
    window: EmbeddedWindow,
    renderer: RefCell<gpui_wgpu::WgpuSceneRenderer>,
}

impl EmbeddedGpui {
    /// Creates an inaccessible GPUI app with one virtual window.
    pub fn new(config: EmbeddedConfig) -> gpui::Result<Self> {
        Self::new_with_root(config, |_, cx| cx.new(|_| EmptyRoot))
    }

    /// Creates an inaccessible GPUI app and installs the supplied root view in its virtual window.
    pub fn new_with_root<V>(
        config: EmbeddedConfig,
        build_root_view: impl FnOnce(&mut Window, &mut gpui::App) -> gpui::Entity<V> + 'static,
    ) -> gpui::Result<Self>
    where
        V: 'static + Render,
    {
        let metrics = config.metrics;
        let scene_renderer = config.gpu.scene_renderer(metrics.device_size(), false)?;
        let platform = EmbeddedPlatform::new(&config, scene_renderer.sprite_atlas().clone());
        let launch_error = Rc::new(RefCell::new(None));
        let launch_error_for_callback = launch_error.clone();
        let platform_for_callback = platform.clone();
        let application = gpui::Application::new_inaccessible(platform.clone());
        let application = if let Some(assets) = config.assets.clone() {
            application.with_assets(SharedAssetSource(assets))
        } else {
            application
        };
        let application = application.run_embedded(move |cx| {
            let options = WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(metrics.bounds)),
                titlebar: None,
                kind: WindowKind::Normal,
                window_decorations: Some(WindowDecorations::Client),
                ..Default::default()
            };
            if let Err(error) = cx.open_window(options, build_root_view) {
                *launch_error_for_callback.borrow_mut() = Some(error.to_string());
            }
            let _ = platform_for_callback.window();
        });
        if let Some(error) = launch_error.take() {
            return Err(std::io::Error::other(error).into());
        }
        let window = platform
            .window()
            .ok_or_else(|| std::io::Error::other("embedded window was not created"))?;
        Ok(Self {
            platform,
            application,
            window,
            renderer: RefCell::new(scene_renderer),
        })
    }

    /// Runs the explicit foreground task phase and returns the next host wake deadline.
    ///
    /// Background workers remain GPUI-owned. The host should call this method from its event-loop
    /// update phase, then schedule one wake for [`crate::PollOutcome::next_deadline`].
    pub fn poll(&self) -> crate::PollOutcome {
        self.platform.poll()
    }

    /// Updates the virtual window metrics.
    pub fn set_window_metrics(&self, metrics: WindowMetrics) {
        self.window.set_metrics(metrics);
    }
    /// Dispatches one host input event.
    pub fn dispatch_input(&self, input: PlatformInput) -> gpui::Result<InputOutcome> {
        let result = self.window.dispatch_input(input)?;
        let (redraw_requested, pointer_capture) = self.application.update(|cx| {
            self.window.handle().update(cx, |_, window, _| {
                (
                    window.is_dirty() || window.needs_redraw(),
                    window.captured_hitbox().is_some(),
                )
            })
        })?;
        Ok(InputOutcome {
            propagate: result.propagate,
            default_prevented: result.default_prevented,
            redraw_requested,
            pointer_capture,
        })
    }
    /// Inserts committed text into the active input handler.
    pub fn insert_text(&self, text: &str) -> bool {
        self.window.insert_text(text)
    }

    /// Updates marked text using UTF-16 indices for both ranges.
    ///
    /// A host adapter that receives byte-indexed preedit ranges must convert them to UTF-16
    /// offsets before calling this method.
    pub fn set_marked_text(
        &self,
        range_utf16: Option<std::ops::Range<usize>>,
        text: &str,
        selected_range_utf16: Option<std::ops::Range<usize>>,
    ) -> bool {
        self.window
            .set_marked_text(range_utf16, text, selected_range_utf16)
    }

    /// Ends the active marked-text composition.
    pub fn unmark_text(&self) -> bool {
        self.window.unmark_text()
    }

    /// Returns the active UTF-16 selection range.
    pub fn selected_text_range(&self) -> Option<gpui::UTF16Selection> {
        self.window.selected_text_range()
    }

    /// Returns the active marked-text range in UTF-16 indices.
    pub fn marked_text_range(&self) -> Option<std::ops::Range<usize>> {
        self.window.marked_text_range()
    }

    /// Returns the host-space candidate bounds for the active IME composition.
    pub fn ime_candidate_bounds(&self) -> Option<Bounds<Pixels>> {
        self.window.ime_candidate_bounds()
    }

    /// Requests GPUI to build its next frame.
    ///
    /// Task polling is an explicit preceding host phase; this method never calls [`Self::poll`]
    /// implicitly.
    pub fn prepare_frame(&self) -> gpui::Result<gpui::FrameStatus> {
        self.application.update(|cx| {
            self.window.handle().update(cx, |_, window, cx| {
                let clear = window.prepare_frame(cx);
                let status = window.frame_status();
                self.window.record_scene(window.rendered_scene(), status);
                clear.clear(cx);
                status
            })
        })
    }
    /// Returns metadata for the last frame.
    pub fn rendered_scene(&self) -> Option<RenderedScene> {
        self.window.rendered_scene()
    }
    /// Encodes the rendered frame through the host-owned target.
    pub fn encode(
        &self,
        target: HostRenderTarget<'_>,
        encoder: &mut gpui_wgpu::wgpu::CommandEncoder,
    ) -> gpui::Result<gpui_wgpu::RenderStats> {
        let mut renderer = self.renderer.borrow_mut();
        self.application.update(|cx| {
            self.window.handle().update(cx, |_, window, _| {
                renderer.encode(window.rendered_scene(), encoder, target)
            })
        })?
    }

    /// Encodes the rendered frame into the convenience target allocated by this crate.
    pub fn encode_offscreen(
        &self,
        target: &OffscreenTarget,
        encoder: &mut gpui_wgpu::wgpu::CommandEncoder,
    ) -> gpui::Result<gpui_wgpu::RenderStats> {
        self.encode(
            target.target(gpui_wgpu::wgpu::LoadOp::Clear(
                gpui_wgpu::wgpu::Color::TRANSPARENT,
            )),
            encoder,
        )
    }

    /// Encodes the rendered frame and registered engine viewports into a host target.
    pub fn encode_with_external_surfaces(
        &self,
        target: HostRenderTarget<'_>,
        encoder: &mut gpui_wgpu::wgpu::CommandEncoder,
        registry: &gpui::ExternalSurfaceRegistry<gpui_wgpu::WgpuExternalSurface>,
    ) -> gpui::Result<gpui_wgpu::RenderStats> {
        let mut renderer = self.renderer.borrow_mut();
        self.application.update(|cx| {
            self.window.handle().update(cx, |_, window, _| {
                renderer.encode_with_external_surfaces(
                    window.rendered_scene(),
                    encoder,
                    target,
                    registry,
                )
            })
        })?
    }

    /// Marks the retained frame as presented after the host submits its command buffer.
    ///
    /// Do not call this after an acquire, encode, submit, or present failure. Returns `false` when
    /// the acknowledgement refers to a stale scene generation.
    pub fn mark_presented(&self, scene_generation: u64) -> gpui::Result<bool> {
        self.application.update(|cx| {
            self.window
                .handle()
                .update(cx, |_, window, _| window.mark_presented(scene_generation))
        })
    }

    /// Rebuilds the surface-free renderer after host device recovery.
    ///
    /// The platform atlas keeps its identity and is rebound in place. The host must recreate
    /// borrowed render targets. Pass the external-surface registry when the host has registered
    /// external surfaces; their generations are snapshotted and each view must be replaced
    /// before that ID is encoded again. Hosts without external surfaces may pass `None`.
    pub fn replace_gpu_context(
        &self,
        adapter: &gpui_wgpu::wgpu::Adapter,
        device: Arc<gpui_wgpu::wgpu::Device>,
        queue: Arc<gpui_wgpu::wgpu::Queue>,
        external_surfaces: Option<&gpui::ExternalSurfaceRegistry<gpui_wgpu::WgpuExternalSurface>>,
    ) -> gpui::Result<()> {
        self.renderer.borrow_mut().replace_gpu_context(
            adapter,
            device,
            queue,
            external_surfaces,
        )?;
        self.application.update(|cx| {
            self.window.handle().update(cx, |_, window, _| {
                window.refresh();
            })
        })
    }
    /// Drains cursor and IME commands for the host to apply.
    pub fn take_host_commands(&self) -> Vec<HostCommand> {
        self.platform.take_host_commands()
    }
    /// Updates GPUI application state.
    pub fn update<R>(&self, f: impl FnOnce(&mut gpui::App) -> R) -> R {
        self.application.update(f)
    }
    /// Returns the virtual window's GPUI handle.
    pub fn window_handle(&self) -> AnyWindowHandle {
        self.window.handle()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_convert_logical_size_to_device_size() {
        let metrics = WindowMetrics::new(gpui::size(gpui::px(320.), gpui::px(180.)), 2.0);
        assert_eq!(
            metrics.device_size(),
            gpui::size(DevicePixels(640), DevicePixels(360))
        );
    }

    #[test]
    fn input_outcome_distinguishes_consumed_and_propagated_events() {
        let consumed = InputOutcome {
            propagate: false,
            ..Default::default()
        };
        let propagated = InputOutcome {
            propagate: true,
            ..Default::default()
        };
        assert!(!consumed.propagate);
        assert!(propagated.propagate);
        assert_ne!(consumed, propagated);
    }
}
