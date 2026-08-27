# Hosting gpui-component in gpui-ce

`gpui-ce` includes a source snapshot of `longbridge/gpui-component` at commit
`9890a77c2d427f05b2512f05ddc6cce84686f9a6`. The complete upstream UI module tree is compiled
against this workspace's `gpui` crate; `gpui_platform` is not introduced into the embedded host.

## Source layout

- `crates/gpui_base` contains the upstream behavior, accessibility, input, animation, and
  primitive-control foundation.
- `crates/gpui_component` contains the complete upstream presentation library, locales, themes,
  controls, overlays, inputs, data views, charts, text support, docking, and native menus.
- `crates/gpui_component_assets` and `crates/gpui_component_macros` contain the pinned upstream
  icons and procedural macros.
- `crates/gpui_embed/examples/gpui_component.rs` runs a windowed component over an application-owned
  WGPU scene.
- `crates/gpui_embed/examples/headless_component_suite.rs` renders a representative component suite
  into an application-owned offscreen texture.

The snapshot crates remain `publish = false`. The upstream commit is the update boundary; local
compatibility belongs in GPUI unless a dependency-version difference makes a narrow source change
unavoidable.

## Compatibility boundary

GPUI once exposed `Rgba { r, g, b, a }` and `Hsla { h, s, l, a }`. A direct migration to palette
aliases broke source compatibility, schema generation, hashing, and inherent color methods used by
the component ecosystem. GPUI now exposes the established public structs again and converts to
palette internally for color-space operations. Renderer-facing scene colors retain their explicit
four-float representation.

The remaining pinned-source adaptations are deliberately narrow:

1. `InteractiveElement::accessibility_id` projects stable author IDs into AccessKit nodes.
2. One direct `TextRun` initializer supplies the local `letter_spacing` field.
3. Windows native-menu calls use the workspace's `windows` 0.62 signatures.

## Verification

From the workspace root:

```sh
cargo check -p gpui-component --all-targets
cargo test -p gpui-component
cargo run -p gpui_embed --example headless_component_suite
```

The headless suite renders upstream Button, Checkbox, Switch, Progress, Badge, Label, and Separator
components in both light and dark themes. It checks that GPUI encoded GPU instances, reads the RGBA
texture back, validates opacity, contrast, accent colors, color diversity, and the light/dark pixel
delta, then writes `target/gpui-embed/headless-component-suite.png` for visual review.

## Runtime ownership

The examples do not hand application hosting back to GPUI:

- the application owns window creation and the event loop;
- the application owns the WGPU instance, adapter, device, queue, targets, and presentation;
- `gpui_embed` retains and encodes the GPUI scene into a host-provided target;
- external surfaces let the host compose its own WGPU scene inside GPUI layout;
- normalized host events are dispatched through the virtual GPUI window.

## Host capability boundary

Full component API coverage is separate from native-service coverage. In-window components render
through the embedded host, while native menus, file prompts, URL opening, credentials, and other OS
services require explicit host callbacks. The current embedded application also starts in
inaccessible mode; production accessibility requires forwarding AccessKit tree updates and action
requests through the host boundary. Unsupported native services should remain explicit rather than
being treated as verified component behavior.
