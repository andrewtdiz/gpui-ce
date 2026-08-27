use gpui::{
    AnyView, Context, IntoElement, ParentElement as _, Render, StyleRefinement, Styled, Window, div,
};
use gpui_base::StyledExt as _;

use crate::ActiveTheme as _;

/// The first view hosted by a component window.
///
/// The embedded facade intentionally owns only the viewport and inherited
/// theme. Overlay managers can be added independently as more upstream example
/// cases are brought across.
pub struct Root {
    style: StyleRefinement,
    view: AnyView,
}

impl Root {
    pub fn new(view: impl Into<AnyView>, _window: &mut Window, _cx: &mut Context<Self>) -> Self {
        Self {
            style: StyleRefinement::default(),
            view: view.into(),
        }
    }
}

impl Styled for Root {
    fn style(&mut self) -> &mut StyleRefinement {
        &mut self.style
    }
}

impl Render for Root {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme();
        div()
            .size_full()
            .bg(theme.background)
            .text_color(theme.foreground)
            .child(self.view.clone())
            .refine_style(&self.style)
    }
}
