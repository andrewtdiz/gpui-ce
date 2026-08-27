//! The gpui-component compatibility slice for host-embedded GPUI.
//!
//! This crate vendors the upstream gpui-component source snapshot and exposes
//! the application-shell pieces used by its `hello_world` example: theme
//! initialization, [`Root`], and [`Button`]. The behavior layer is provided by
//! the vendored `gpui-base` crate and runs without `gpui_platform`.

mod compat_button;
mod compat_root;
mod compat_theme;

/// Buttons and button variants.
pub mod button {
    pub use crate::compat_button::{Button, ButtonVariant};
}

pub use compat_button::{Button, ButtonVariant};
pub use compat_root::Root;
pub use compat_theme::{ActiveTheme, Theme};
pub use gpui_base::{StyledExt, h_flex, v_flex};

use gpui::App;

/// Initializes the component behavior layer and default theme.
///
/// Call this once before constructing component views.
pub fn init(cx: &mut App) {
    gpui_base::init(cx);
    compat_theme::init(cx);
}
