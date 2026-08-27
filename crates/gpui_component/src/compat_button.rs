use gpui::{
    AnyElement, App, ClickEvent, ColorExt as _, ElementId, InteractiveElement, Interactivity,
    IntoElement, ParentElement, RenderOnce, StatefulInteractiveElement, StyleRefinement, Styled,
    Window, prelude::FluentBuilder as _,
};

use crate::ActiveTheme as _;

/// The visual treatment applied to a [`Button`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ButtonVariant {
    /// A neutral surface button.
    #[default]
    Secondary,
    /// The primary application action.
    Primary,
}

/// A styled button backed by gpui-base interaction and accessibility behavior.
#[derive(IntoElement)]
pub struct Button {
    base: gpui_base::Button,
    variant: ButtonVariant,
    disabled: bool,
}

impl Button {
    pub fn new(id: impl Into<ElementId>) -> Self {
        Self {
            base: gpui_base::Button::new(id),
            variant: ButtonVariant::Secondary,
            disabled: false,
        }
    }

    /// Applies the primary action treatment.
    pub fn primary(mut self) -> Self {
        self.variant = ButtonVariant::Primary;
        self
    }

    /// Applies an explicit variant.
    pub fn with_variant(mut self, variant: ButtonVariant) -> Self {
        self.variant = variant;
        self
    }

    /// Sets the visible and accessible button label.
    pub fn label(mut self, label: impl Into<gpui::SharedString>) -> Self {
        let label = label.into();
        self.base = self.base.accessibility_label(label.clone()).child(label);
        self
    }

    /// Sets whether the button accepts pointer and keyboard activation.
    pub fn disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self.base = self.base.disabled(disabled);
        self
    }

    /// Handles pointer, Enter, and Space activation.
    pub fn on_click(
        mut self,
        handler: impl Fn(&ClickEvent, &mut Window, &mut App) + 'static,
    ) -> Self {
        self.base = self.base.on_click(handler);
        self
    }
}

impl Styled for Button {
    fn style(&mut self) -> &mut StyleRefinement {
        self.base.style()
    }
}

impl ParentElement for Button {
    fn extend(&mut self, elements: impl IntoIterator<Item = AnyElement>) {
        self.base.extend(elements);
    }
}

impl InteractiveElement for Button {
    fn interactivity(&mut self) -> &mut Interactivity {
        self.base.interactivity()
    }
}

impl StatefulInteractiveElement for Button {}

impl RenderOnce for Button {
    fn render(self, _window: &mut Window, cx: &mut App) -> impl IntoElement {
        let theme = cx.theme();
        let (background, foreground) = match self.variant {
            ButtonVariant::Primary => (theme.primary, theme.primary_foreground),
            ButtonVariant::Secondary => (theme.secondary, theme.secondary_foreground),
        };
        let hover = background.opacity(0.88);
        let active = background.opacity(0.76);
        let border = theme.border;
        let radius = theme.radius;

        self.base
            .px_4()
            .py_2()
            .rounded(radius)
            .border_1()
            .border_color(border)
            .bg(background)
            .text_color(foreground)
            .cursor_pointer()
            .hover(move |style| style.bg(hover))
            .active(move |style| style.bg(active))
            .when(self.disabled, |button| button.opacity(0.5))
    }
}
