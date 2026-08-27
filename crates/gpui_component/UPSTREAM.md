# gpui-component source snapshot

This directory was imported from
[`longbridge/gpui-component`](https://github.com/longbridge/gpui-component) at commit
`9890a77c2d427f05b2512f05ddc6cce84686f9a6` (Apache-2.0).

The complete upstream `crates/ui/src` module tree is active, including its normal `init`, `Root`,
theme, control, input, overlay, data, chart, text, dock, and native-menu APIs. Compatibility is
maintained at the local GPUI public API boundary so this directory can stay close to its pinned
upstream source instead of accumulating widget-by-widget rewrites.

This deliberately avoids pulling `gpui_platform` back into the dependency graph. The native
host owns the event loop, window, WGPU device, command encoder, submission, and presentation.
See [`../../docs/gpui-component.md`](../../docs/gpui-component.md) for the porting notes and
support boundary.

Run the embedded windowed example and headless component suite with:

```sh
cargo run -p gpui_embed --example gpui_component
cargo run -p gpui_embed --example headless_component_suite
```
