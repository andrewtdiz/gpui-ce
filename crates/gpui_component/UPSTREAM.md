# gpui-component compatibility snapshot

This directory was imported from
[`longbridge/gpui-component`](https://github.com/longbridge/gpui-component) at commit
`9890a77c2d427f05b2512f05ddc6cce84686f9a6` (Apache-2.0).

The upstream source remains in `src/` as the porting reference. The crate root currently
compiles the first host-independent compatibility slice needed by upstream's `hello_world`
example:

- `gpui_component::init`
- `Root`
- `Theme` and `ActiveTheme`
- `button::Button` and `ButtonVariant`
- the vendored `gpui-base` behavior layer

This deliberately avoids pulling `gpui_platform` back into the dependency graph. The native
host owns the event loop, window, WGPU device, command encoder, submission, and presentation.
See [`../../docs/gpui-component.md`](../../docs/gpui-component.md) for the porting notes and
support boundary.

Run the embedded version of the upstream example with:

```sh
cargo run -p gpui_embed --example gpui_component
```
