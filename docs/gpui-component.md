# Hosting gpui-component in gpui-ce

`gpui-ce` includes a source snapshot of `longbridge/gpui-component` at commit
`9890a77c2d427f05b2512f05ddc6cce84686f9a6`. Copying the source into this workspace is the
smallest useful integration boundary: it lets compatibility changes land atomically with GPUI
API changes, keeps the host free of `gpui_platform`, and retains upstream file history and
license provenance in one clearly marked directory. A permanent repository fork only becomes
useful once enough of the component catalog is active to benefit from independent releases.

## Source layout

- `crates/gpui_base` is the imported behavior, interaction, input, animation, and theme-token
  foundation. It is compiled against the local `gpui` crate.
- `crates/gpui_component` preserves the upstream component source as a porting reference and
  currently exposes the `hello_world` compatibility slice (`init`, `Root`, semantic theme, and
  `Button`).
- `crates/gpui_component_assets` and `crates/gpui_component_macros` preserve upstream assets
  and procedural macros for subsequent component slices.
- `crates/gpui_embed/examples/gpui_component.rs` ports upstream's `examples/hello_world` API
  shape into the arbitrary native WGPU host.

The snapshot crates are `publish = false`; they are an in-tree compatibility layer, not a
replacement crates.io release.

Upstream auto-discovered bins, examples, benches, and integration tests are kept as migration
references but disabled in the snapshot manifests until their corresponding component modules
are activated. Workspace `--all-targets` checks still compile both libraries and their unit tests.

## Compatibility work

The imported base layer needed four narrow GPUI compatibility additions/adaptations:

1. GPUI's spring state/configuration API was restored as a host-independent module.
2. Palette conversion and alpha helpers were exposed so component colors use GPUI's current
   `palette`-backed `Hsla`/`Rgba` types.
3. Imported direct `TextRun` construction now initializes `letter_spacing`.
4. Theme-token schemas use the local `Hsla` and shadow schema helpers.

The complete upstream presentation module is not compiled yet. Current upstream main also
assumes newer Zed-only scene-color, accessibility, and platform APIs across that catalog.
Keeping those inactive sources beside the facade makes each later component a bounded port
instead of hiding hundreds of unrelated compatibility shims in the first working example.

## Runtime ownership

The example proves that the component path does not take over hosting:

- winit owns window creation and the event loop;
- the application owns the WGPU instance, adapter, device, queue, and surface;
- `gpui_embed` renders GPUI into an offscreen target;
- the host encodes a rotating triangle into an external-surface texture;
- one host command encoder composites GPUI and the external surface, then submits and presents;
- translated pointer events activate the component button and update its label.

Run it from the workspace root:

```sh
cargo run -p gpui_embed --example gpui_component
```
