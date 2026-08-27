use gpui::{App, Global, Hsla, IntoColor as _, Pixels, px};

/// The compact semantic theme used by the embedded component facade.
#[derive(Clone, Debug)]
pub struct Theme {
    pub background: Hsla,
    pub foreground: Hsla,
    pub surface: Hsla,
    pub surface_foreground: Hsla,
    pub primary: Hsla,
    pub primary_foreground: Hsla,
    pub secondary: Hsla,
    pub secondary_foreground: Hsla,
    pub muted: Hsla,
    pub muted_foreground: Hsla,
    pub border: Hsla,
    pub ring: Hsla,
    pub radius: Pixels,
}

impl Default for Theme {
    fn default() -> Self {
        let color = |hex| gpui::rgb(hex).into_color();
        Self {
            background: color(0xf8fafc),
            foreground: color(0x0f172a),
            surface: color(0xffffff),
            surface_foreground: color(0x0f172a),
            primary: color(0x2563eb),
            primary_foreground: color(0xffffff),
            secondary: color(0xe2e8f0),
            secondary_foreground: color(0x1e293b),
            muted: color(0xf1f5f9),
            muted_foreground: color(0x64748b),
            border: color(0xcbd5e1),
            ring: color(0x60a5fa),
            radius: px(8.),
        }
    }
}

impl Global for Theme {}

impl Theme {
    /// Returns the active theme.
    pub fn global(cx: &App) -> &Self {
        cx.global::<Self>()
    }
}

/// Access to the active component theme through an application context.
pub trait ActiveTheme {
    fn theme(&self) -> &Theme;
}

impl ActiveTheme for App {
    fn theme(&self) -> &Theme {
        Theme::global(self)
    }
}

pub(crate) fn init(cx: &mut App) {
    if !cx.has_global::<Theme>() {
        cx.set_global(Theme::default());
    }

    let theme = cx.global::<Theme>().clone();
    let base = gpui_base::Theme::global_mut(cx);
    base.tokens.colors = gpui_base::ColorTokens {
        background: theme.background,
        foreground: theme.foreground,
        surface: theme.surface,
        surface_foreground: theme.surface_foreground,
        primary: theme.primary,
        primary_foreground: theme.primary_foreground,
        secondary: theme.secondary,
        secondary_foreground: theme.secondary_foreground,
        muted: theme.muted,
        muted_foreground: theme.muted_foreground,
        accent: theme.secondary,
        accent_foreground: theme.secondary_foreground,
        destructive: gpui::rgb(0xdc2626).into_color(),
        destructive_foreground: theme.primary_foreground,
        border: theme.border,
        input: theme.border,
        ring: theme.ring,
    };
    base.tokens.radius.md = theme.radius;
    base.tokens.radius.lg = theme.radius;
}
