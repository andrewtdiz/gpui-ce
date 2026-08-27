use anyhow::{Context as _, bail};
pub use palette::IntoColor;
use palette::{OklabHue, Oklcha};
use schemars::{JsonSchema, json_schema};
use serde::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{self, Visitor},
};
use std::{
    borrow::Cow,
    fmt::{self, Display, Formatter},
    hash::{Hash, Hasher},
};

/// GPUI's internal palette RGBA representation.
pub(crate) type PaletteRgba = palette::rgb::Rgba;

/// GPUI's internal palette HSLA representation.
pub(crate) type PaletteHsla = palette::Hsla;

/// Convert an RGB hex color code number to a color type
pub fn rgb(hex: u32) -> Rgba {
    let [_, r, g, b] = hex.to_be_bytes().map(|b| (b as f32) / 255.0);
    Rgba { r, g, b, a: 1.0 }
}

/// Convert an RGBA hex color code number to [`Rgba`]
pub fn rgba(hex: u32) -> Rgba {
    let [r, g, b, a] = hex.to_be_bytes().map(|b| (b as f32) / 255.0);
    Rgba { r, g, b, a }
}

/// Swap from RGBA with premultiplied alpha to BGRA
pub fn swap_rgba_pa_to_bgra(color: &mut [u8]) {
    color.swap(0, 2);
    if color[3] > 0 {
        let a = color[3] as f32 / 255.;
        color[0] = (color[0] as f32 / a) as u8;
        color[1] = (color[1] as f32 / a) as u8;
        color[2] = (color[2] as f32 / a) as u8;
    }
}

/// An RGBA color.
#[derive(PartialEq, Clone, Copy, Default)]
#[repr(C)]
pub struct Rgba {
    /// The red component of the color, in the range 0.0 to 1.0.
    pub r: f32,
    /// The green component of the color, in the range 0.0 to 1.0.
    pub g: f32,
    /// The blue component of the color, in the range 0.0 to 1.0.
    pub b: f32,
    /// The alpha component of the color, in the range 0.0 to 1.0.
    pub a: f32,
}

impl fmt::Debug for Rgba {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "rgba({:#010x})", u32::from(*self))
    }
}

impl Rgba {
    /// Creates an RGBA color from normalized channel values.
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    /// Blends `other` on top of this color using `other`'s alpha channel.
    pub fn blend(&self, other: Rgba) -> Self {
        if other.a >= 1.0 {
            other
        } else if other.a <= 0.0 {
            *self
        } else {
            Rgba {
                r: (self.r * (1.0 - other.a)) + (other.r * other.a),
                g: (self.g * (1.0 - other.a)) + (other.g * other.a),
                b: (self.b * (1.0 - other.a)) + (other.b * other.a),
                a: self.a,
            }
        }
    }

    /// Returns this color with its alpha channel replaced by `a`.
    pub fn alpha(&self, a: f32) -> Self {
        Self {
            a: a.clamp(0., 1.),
            ..*self
        }
    }

    /// Returns this color with its alpha channel multiplied by `factor`.
    pub fn opacity(&self, factor: f32) -> Self {
        Self {
            a: self.a * factor.clamp(0., 1.),
            ..*self
        }
    }
}

impl From<Rgba> for u32 {
    fn from(rgba: Rgba) -> Self {
        let r = (rgba.r * 255.0) as u32;
        let g = (rgba.g * 255.0) as u32;
        let b = (rgba.b * 255.0) as u32;
        let a = (rgba.a * 255.0) as u32;
        (r << 24) | (g << 16) | (b << 8) | a
    }
}

struct RgbaVisitor;

impl Visitor<'_> for RgbaVisitor {
    type Value = Rgba;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a string in the format #rgb, #rgba, #rrggbb, or #rrggbbaa")
    }

    fn visit_str<E: de::Error>(self, value: &str) -> Result<Rgba, E> {
        Rgba::try_from(value).map_err(E::custom)
    }
}

impl JsonSchema for Rgba {
    fn schema_name() -> Cow<'static, str> {
        "Rgba".into()
    }

    fn json_schema(_generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        json_schema!({
            "type": "string",
            "pattern": "^#([0-9a-fA-F]{3}|[0-9a-fA-F]{4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$"
        })
    }
}

impl<'de> Deserialize<'de> for Rgba {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_str(RgbaVisitor)
    }
}

impl Serialize for Rgba {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let r = (self.r * 255.0).round() as u8;
        let g = (self.g * 255.0).round() as u8;
        let b = (self.b * 255.0).round() as u8;
        let a = (self.a * 255.0).round() as u8;
        serializer.serialize_str(&format!("#{r:02x}{g:02x}{b:02x}{a:02x}"))
    }
}

impl TryFrom<&'_ str> for Rgba {
    type Error = anyhow::Error;

    fn try_from(value: &'_ str) -> Result<Self, Self::Error> {
        const EXPECTED_FORMATS: &str = "Expected #rgb, #rgba, #rrggbb, or #rrggbbaa";
        const INVALID_UNICODE: &str = "invalid unicode characters in color";

        let Some(("", hex)) = value.trim().split_once('#') else {
            bail!("invalid RGBA hex color: '{value}'. {EXPECTED_FORMATS}");
        };

        let (r, g, b, a) = match hex.len() {
            3 | 4 => {
                let component = |range, name| {
                    u8::from_str_radix(
                        hex.get(range).with_context(|| {
                            format!("{INVALID_UNICODE}: {name} component for value: '{value}'")
                        })?,
                        16,
                    )
                    .map_err(anyhow::Error::from)
                };
                let duplicate = |component: u8| (component << 4) | component;
                (
                    duplicate(component(0..1, "r")?),
                    duplicate(component(1..2, "g")?),
                    duplicate(component(2..3, "b")?),
                    if hex.len() == 4 {
                        duplicate(component(3..4, "a")?)
                    } else {
                        0xff
                    },
                )
            }
            6 | 8 => {
                let component = |range, name| {
                    u8::from_str_radix(
                        hex.get(range).with_context(|| {
                            format!("{INVALID_UNICODE}: {name} component for value: '{value}'")
                        })?,
                        16,
                    )
                    .map_err(anyhow::Error::from)
                };
                (
                    component(0..2, "r")?,
                    component(2..4, "g")?,
                    component(4..6, "b")?,
                    if hex.len() == 8 {
                        component(6..8, "a")?
                    } else {
                        0xff
                    },
                )
            }
            _ => bail!("invalid RGBA hex color: '{value}'. {EXPECTED_FORMATS}"),
        };

        Ok(Rgba {
            r: r as f32 / 255.,
            g: g as f32 / 255.,
            b: b as f32 / 255.,
            a: a as f32 / 255.,
        })
    }
}

/// An HSLA color.
#[derive(Default, Copy, Clone, Debug)]
#[repr(C)]
pub struct Hsla {
    /// Hue, in a range from 0 to 1.
    pub h: f32,
    /// Saturation, in a range from 0 to 1.
    pub s: f32,
    /// Lightness, in a range from 0 to 1.
    pub l: f32,
    /// Alpha, in a range from 0 to 1.
    pub a: f32,
}

#[cfg(feature = "proptest")]
mod property {
    use super::Hsla;
    use proptest::prelude::*;

    impl Hsla {
        /// Produces opaque colors for property tests.
        pub fn opaque_strategy() -> impl Strategy<Value = Self> {
            (0.0f32..=1.0, 0.0f32..=1.0, 0.0f32..=1.0).prop_map(|(h, s, l)| Hsla { h, s, l, a: 1. })
        }
    }

    impl Arbitrary for Hsla {
        type Strategy = BoxedStrategy<Self>;
        type Parameters = ();

        fn arbitrary_with((): Self::Parameters) -> Self::Strategy {
            (0.0f32..=1.0, 0.0f32..=1.0, 0.0f32..=1.0, 0.0f32..=1.0)
                .prop_map(|(h, s, l, a)| Hsla { h, s, l, a })
                .boxed()
        }
    }
}

impl PartialEq for Hsla {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for Hsla {}

impl PartialOrd for Hsla {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Hsla {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.h
            .total_cmp(&other.h)
            .then(self.s.total_cmp(&other.s))
            .then(self.l.total_cmp(&other.l))
            .then(self.a.total_cmp(&other.a))
    }
}

impl Hash for Hsla {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u32(u32::from_be_bytes(self.h.to_be_bytes()));
        state.write_u32(u32::from_be_bytes(self.s.to_be_bytes()));
        state.write_u32(u32::from_be_bytes(self.l.to_be_bytes()));
        state.write_u32(u32::from_be_bytes(self.a.to_be_bytes()));
    }
}

impl Display for Hsla {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "hsla({:.2}, {:.2}%, {:.2}%, {:.2})",
            self.h * 360.,
            self.s * 100.,
            self.l * 100.,
            self.a
        )
    }
}

/// Construct an [`Hsla`] object from plain values.
pub const fn hsla(h: f32, s: f32, l: f32, a: f32) -> Hsla {
    Hsla {
        h: h.clamp(0., 1.),
        s: s.clamp(0., 1.),
        l: l.clamp(0., 1.),
        a: a.clamp(0., 1.),
    }
}

/// Constructs a ['Oklcha'](palette::Oklcha) object from plain values.
pub fn oklcha<T>(lightness: T, chroma: T, hue: impl Into<OklabHue<T>>, alpha: T) -> Oklcha<T> {
    Oklcha::new(lightness, chroma, hue, alpha)
}

/// Pure black in [`Hsla`]
pub const fn black() -> Hsla {
    Hsla {
        h: 0.,
        s: 0.,
        l: 0.,
        a: 1.,
    }
}

/// Transparent black in [`Hsla`]
pub const fn transparent_black() -> Hsla {
    Hsla {
        h: 0.,
        s: 0.,
        l: 0.,
        a: 0.,
    }
}

/// Transparent white in [`Hsla`]
pub const fn transparent_white() -> Hsla {
    Hsla {
        h: 0.,
        s: 0.,
        l: 1.,
        a: 0.,
    }
}

/// Opaque grey in [`Hsla`], values must be provided in the range [0, 1]
pub const fn opaque_grey(lightness: f32, opacity: f32) -> Hsla {
    Hsla {
        h: 0.,
        s: 0.,
        l: lightness.clamp(0., 1.),
        a: opacity.clamp(0., 1.),
    }
}

/// Pure white in [`Hsla`]
pub const fn white() -> Hsla {
    Hsla {
        h: 0.,
        s: 0.,
        l: 1.,
        a: 1.,
    }
}

/// The color red in [`Hsla`]
pub const fn red() -> Hsla {
    Hsla {
        h: 0.,
        s: 1.,
        l: 0.5,
        a: 1.,
    }
}

/// The color blue in [`Hsla`]
pub const fn blue() -> Hsla {
    Hsla {
        h: 0.6666666667,
        s: 1.,
        l: 0.5,
        a: 1.,
    }
}

/// The color green in [`Hsla`]
pub const fn green() -> Hsla {
    Hsla {
        h: 0.3333333333,
        s: 1.,
        l: 0.25,
        a: 1.,
    }
}

/// The color yellow in [`Hsla`]
pub const fn yellow() -> Hsla {
    Hsla {
        h: 0.1666666667,
        s: 1.,
        l: 0.5,
        a: 1.,
    }
}

/// Generates the JSON schema used by [`Hsla`].
pub fn hsla_schemar(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
    Hsla::json_schema(generator)
}

/// Generates the JSON schema used by [`Rgba`].
pub fn rgba_schemar(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
    Rgba::json_schema(generator)
}

impl Hsla {
    /// Creates an HSLA color from normalized channel values.
    pub const fn new(h: f32, s: f32, l: f32, a: f32) -> Self {
        hsla(h, s, l, a)
    }

    /// Converts this color to RGBA.
    pub fn to_rgb(self) -> Rgba {
        self.into()
    }

    /// Returns red.
    pub const fn red() -> Self {
        red()
    }
    /// Returns green.
    pub const fn green() -> Self {
        green()
    }
    /// Returns blue.
    pub const fn blue() -> Self {
        blue()
    }
    /// Returns black.
    pub const fn black() -> Self {
        black()
    }
    /// Returns white.
    pub const fn white() -> Self {
        white()
    }
    /// Returns transparent black.
    pub const fn transparent_black() -> Self {
        transparent_black()
    }

    /// Returns whether this color is fully transparent.
    pub fn is_transparent(&self) -> bool {
        self.a == 0.0
    }

    /// Returns whether this color is fully opaque.
    pub fn is_opaque(&self) -> bool {
        self.a == 1.0
    }

    /// Blends `other` on top of this color using `other`'s alpha channel.
    pub fn blend(self, other: Hsla) -> Hsla {
        if other.a >= 1.0 {
            other
        } else if other.a <= 0.0 {
            self
        } else {
            Hsla::from(Rgba::from(self).blend(Rgba::from(other)))
        }
    }

    /// Returns this color with saturation removed.
    pub fn grayscale(&self) -> Self {
        Self { s: 0., ..*self }
    }

    /// Reduces this color's alpha by `factor`.
    pub fn fade_out(&mut self, factor: f32) {
        self.a *= 1.0 - factor.clamp(0., 1.);
    }

    /// Returns this color with its alpha multiplied by `factor`.
    pub fn opacity(&self, factor: f32) -> Self {
        Self {
            a: self.a * factor.clamp(0., 1.),
            ..*self
        }
    }

    /// Returns this color with its alpha replaced by `a`.
    pub fn alpha(&self, a: f32) -> Self {
        Self {
            a: a.clamp(0., 1.),
            ..*self
        }
    }
}

impl From<Hsla> for Rgba {
    fn from(color: Hsla) -> Self {
        let c = (1.0 - (2.0 * color.l - 1.0).abs()) * color.s;
        let x = c * (1.0 - ((color.h * 6.0) % 2.0 - 1.0).abs());
        let m = color.l - c / 2.0;
        let (r, g, b) = match (color.h * 6.0).floor() as i32 {
            0 | 6 => (c + m, x + m, m),
            1 => (x + m, c + m, m),
            2 => (m, c + m, x + m),
            3 => (m, x + m, c + m),
            4 => (x + m, m, c + m),
            _ => (c + m, m, x + m),
        };
        Self {
            r: r.clamp(0., 1.),
            g: g.clamp(0., 1.),
            b: b.clamp(0., 1.),
            a: color.a,
        }
    }
}

impl From<Rgba> for Hsla {
    fn from(color: Rgba) -> Self {
        let max = color.r.max(color.g.max(color.b));
        let min = color.r.min(color.g.min(color.b));
        let delta = max - min;
        let l = (max + min) / 2.0;
        let s = if l == 0.0 || l == 1.0 {
            0.0
        } else if l < 0.5 {
            delta / (2.0 * l)
        } else {
            delta / (2.0 - 2.0 * l)
        };
        let h = if delta == 0.0 {
            0.0
        } else if max == color.r {
            ((color.g - color.b) / delta).rem_euclid(6.0) / 6.0
        } else if max == color.g {
            ((color.b - color.r) / delta + 2.0) / 6.0
        } else {
            ((color.r - color.g) / delta + 4.0) / 6.0
        };
        Self {
            h,
            s,
            l,
            a: color.a,
        }
    }
}

impl JsonSchema for Hsla {
    fn schema_name() -> Cow<'static, str> {
        Rgba::schema_name()
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        Rgba::json_schema(generator)
    }
}

impl Serialize for Hsla {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        Rgba::from(*self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Hsla {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Rgba::deserialize(deserializer)?.into())
    }
}

impl From<Rgba> for PaletteRgba {
    fn from(color: Rgba) -> Self {
        Self::new(color.r, color.g, color.b, color.a)
    }
}

impl From<PaletteRgba> for Rgba {
    fn from(color: PaletteRgba) -> Self {
        Self {
            r: color.red,
            g: color.green,
            b: color.blue,
            a: color.alpha,
        }
    }
}

impl From<Hsla> for PaletteHsla {
    fn from(color: Hsla) -> Self {
        Self::new(color.h * 360., color.s, color.l, color.a)
    }
}

impl From<PaletteHsla> for Hsla {
    fn from(color: PaletteHsla) -> Self {
        Self {
            h: color.hue.into_positive_degrees() / 360.,
            s: color.saturation,
            l: color.lightness,
            a: color.alpha,
        }
    }
}

impl palette::FromColor<Rgba> for Hsla {
    fn from_color(color: Rgba) -> Self {
        color.into()
    }
}

impl palette::FromColor<Hsla> for Rgba {
    fn from_color(color: Hsla) -> Self {
        color.into()
    }
}

impl palette::FromColor<Hsla> for Hsla {
    fn from_color(color: Hsla) -> Self {
        color
    }
}

impl palette::FromColor<Rgba> for Rgba {
    fn from_color(color: Rgba) -> Self {
        color
    }
}

impl palette::FromColor<PaletteRgba> for Hsla {
    fn from_color(color: PaletteRgba) -> Self {
        Rgba::from(color).into()
    }
}

impl palette::FromColor<PaletteHsla> for Hsla {
    fn from_color(color: PaletteHsla) -> Self {
        color.into()
    }
}

impl palette::WithAlpha<f32> for Rgba {
    type Color = Self;
    type WithAlpha = Self;

    fn with_alpha(self, alpha: f32) -> Self {
        Self { a: alpha, ..self }
    }

    fn without_alpha(self) -> Self::Color {
        Self { a: 1., ..self }
    }

    fn split(self) -> (Self::Color, f32) {
        let alpha = self.a;
        (self.without_alpha(), alpha)
    }
}

impl palette::WithAlpha<f32> for Hsla {
    type Color = Self;
    type WithAlpha = Self;

    fn with_alpha(self, alpha: f32) -> Self {
        Self { a: alpha, ..self }
    }

    fn without_alpha(self) -> Self::Color {
        Self { a: 1., ..self }
    }

    fn split(self) -> (Self::Color, f32) {
        let alpha = self.a;
        (self.without_alpha(), alpha)
    }
}

/// Wrapper methods to make alpha operations more convenient
pub trait ColorExt {
    /// Performs a SrcAlpha x (1 - SrcAlpha) blend
    fn blend(&self, other: &Self) -> Self
    where
        Self: Sized;

    /// Fade out the color by a given factor. This factor should be between 0.0 and 1.0.
    /// Where 0.0 will leave the color unchanged, and 1.0 will completely fade out the color.
    fn fade_out(&mut self, factor: f32);

    /// Returns this color with its alpha channel replaced by `alpha`.
    ///
    /// This preserves GPUI's pre-palette color API for component libraries.
    fn alpha(&self, alpha: f32) -> Self
    where
        Self: Sized;

    /// Multiplies the alpha value of the color by a given factor and returns a new color.
    /// This is useful for transforming colors with dynamic opacity, such as a color from an
    /// external source.
    ///
    /// Example:
    /// ```
    /// use gpui::ColorExt;
    /// let color = gpui::red();
    /// let faded_color = color.opacity(0.5);
    /// assert_eq!(faded_color.a, 0.5);
    /// ```
    ///
    /// This will return a red color with half the opacity.
    ///
    /// Example:
    /// ```
    /// use gpui::{hsla, ColorExt};
    /// let color = hsla(0.7, 1.0, 0.5, 0.7); // A saturated blue
    /// let faded_color = color.opacity(0.16);
    /// assert!((faded_color.a - 0.112).abs() < 1e-6);
    /// ```
    ///
    /// This will return a blue color with around ~10% opacity,
    /// suitable for an element's hover or selected state.
    ///
    fn opacity(&self, factor: f32) -> Self
    where
        Self: Sized;
}
impl ColorExt for Rgba {
    fn blend(&self, other: &Self) -> Self {
        Rgba::blend(self, *other)
    }

    fn fade_out(&mut self, factor: f32) {
        self.a *= 1.0 - factor.clamp(0., 1.);
    }

    fn alpha(&self, alpha: f32) -> Self {
        Rgba::alpha(self, alpha)
    }

    fn opacity(&self, factor: f32) -> Self {
        Rgba::opacity(self, factor)
    }
}
impl ColorExt for Hsla {
    fn blend(&self, other: &Self) -> Self {
        Hsla::blend(*self, *other)
    }

    fn fade_out(&mut self, factor: f32) {
        Hsla::fade_out(self, factor)
    }

    fn alpha(&self, alpha: f32) -> Self {
        Hsla::alpha(self, alpha)
    }

    fn opacity(&self, factor: f32) -> Self {
        Hsla::opacity(self, factor)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[repr(C)]
pub(crate) enum BackgroundTag {
    Solid = 0,
    LinearGradient = 1,
    PatternSlash = 2,
    Checkerboard = 3,
}

/// A color space for color interpolation.
///
/// References:
/// - <https://developer.mozilla.org/en-US/docs/Web/CSS/color-interpolation-method>
/// - <https://www.w3.org/TR/css-color-4/#typedef-color-space>
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize, JsonSchema)]
#[repr(C)]
pub enum ColorSpace {
    #[default]
    /// The sRGB color space.
    Srgb = 0,
    /// The Oklab color space.
    Oklab = 1,
}

impl Display for ColorSpace {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            ColorSpace::Srgb => write!(f, "sRGB"),
            ColorSpace::Oklab => write!(f, "Oklab"),
        }
    }
}

/// A background color, which can be either a solid color or a linear gradient.
#[derive(Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[repr(C)]
pub struct Background {
    pub(crate) tag: BackgroundTag,
    pub(crate) color_space: ColorSpace,
    pub(crate) solid: Hsla,
    pub(crate) gradient_angle_or_pattern_height: f32,
    pub(crate) colors: [LinearColorStop; 2],
    /// Padding for alignment for repr(C) layout.
    pad: u32,
}

impl std::fmt::Debug for Background {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.tag {
            BackgroundTag::Solid => write!(f, "Solid({:?})", self.solid),
            BackgroundTag::LinearGradient => write!(
                f,
                "LinearGradient({}, {:?}, {:?})",
                self.gradient_angle_or_pattern_height, self.colors[0], self.colors[1]
            ),
            BackgroundTag::PatternSlash => write!(
                f,
                "PatternSlash({:?}, {})",
                self.solid, self.gradient_angle_or_pattern_height
            ),
            BackgroundTag::Checkerboard => write!(
                f,
                "Checkerboard({:?}, {})",
                self.solid, self.gradient_angle_or_pattern_height
            ),
        }
    }
}

impl Eq for Background {}
impl Default for Background {
    fn default() -> Self {
        Self {
            tag: BackgroundTag::Solid,
            solid: Hsla::default(),
            color_space: ColorSpace::default(),
            gradient_angle_or_pattern_height: 0.0,
            colors: [LinearColorStop::default(), LinearColorStop::default()],
            pad: 0,
        }
    }
}

/// Creates a hash pattern background
pub fn pattern_slash(color: impl Into<Hsla>, width: f32, interval: f32) -> Background {
    let width_scaled = (width * 255.0) as u32;
    let interval_scaled = (interval * 255.0) as u32;
    let height = ((width_scaled * 0xFFFF) + interval_scaled) as f32;

    Background {
        tag: BackgroundTag::PatternSlash,
        solid: color.into(),
        gradient_angle_or_pattern_height: height,
        ..Default::default()
    }
}

/// Creates a checkerboard pattern background
pub fn checkerboard(color: impl Into<Hsla>, size: f32) -> Background {
    Background {
        tag: BackgroundTag::Checkerboard,
        solid: color.into(),
        gradient_angle_or_pattern_height: size,
        ..Default::default()
    }
}

/// Creates a solid background color.
pub fn solid_background(color: impl Into<Hsla>) -> Background {
    Background {
        solid: color.into(),
        ..Default::default()
    }
}

/// Creates a LinearGradient background color.
///
/// The gradient line's angle of direction. A value of `0.` is equivalent to top; increasing values rotate clockwise from there.
///
/// The `angle` is in degrees value in the range 0.0 to 360.0.
///
/// <https://developer.mozilla.org/en-US/docs/Web/CSS/gradient/linear-gradient>
pub fn linear_gradient(
    angle: f32,
    from: impl Into<LinearColorStop>,
    to: impl Into<LinearColorStop>,
) -> Background {
    Background {
        tag: BackgroundTag::LinearGradient,
        gradient_angle_or_pattern_height: angle,
        colors: [from.into(), to.into()],
        ..Default::default()
    }
}

/// A color stop in a linear gradient.
///
/// <https://developer.mozilla.org/en-US/docs/Web/CSS/gradient/linear-gradient#linear-color-stop>
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[repr(C)]
pub struct LinearColorStop {
    /// The color of the color stop.
    pub color: Hsla,
    /// The percentage of the gradient, in the range 0.0 to 1.0.
    pub percentage: f32,
}

/// Creates a new linear color stop.
///
/// The percentage of the gradient, in the range 0.0 to 1.0.
pub fn linear_color_stop(color: impl Into<Hsla>, percentage: f32) -> LinearColorStop {
    LinearColorStop {
        color: color.into(),
        percentage,
    }
}

impl LinearColorStop {
    /// Returns a new color stop with the same color, but with a modified alpha value.
    pub fn opacity(&self, factor: f32) -> Self {
        Self {
            percentage: self.percentage,
            color: self.color.opacity(factor),
        }
    }
}

/// What a [`Background`] paints, decoded from its packed representation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BackgroundKind {
    /// A flat color.
    Solid(Hsla),
    /// A linear gradient between two color stops.
    LinearGradient {
        /// The gradient line's angle in degrees, `0.0` pointing up, increasing clockwise.
        angle: f32,
        /// The two ends of the gradient.
        stops: [LinearColorStop; 2],
    },
    /// A diagonal stripe pattern.
    PatternSlash {
        /// The stripe color.
        color: Hsla,
        /// The stripe width, in logical pixels.
        width: f32,
        /// The gap between stripes, in logical pixels.
        interval: f32,
    },
    /// Alternating squares of one color and full transparency.
    Checkerboard {
        /// The color of one set of squares. The other set is fully transparent.
        color: Hsla,
        /// The width and height of each square, in logical pixels.
        size: f32,
    },
}

impl Background {
    /// Returns the solid color if this is a solid background, None otherwise.
    pub fn as_solid(&self) -> Option<Hsla> {
        if self.tag == BackgroundTag::Solid {
            Some(self.solid)
        } else {
            None
        }
    }

    /// Returns the decoded form of this background.
    pub fn kind(&self) -> BackgroundKind {
        match self.tag {
            BackgroundTag::Solid => BackgroundKind::Solid(self.solid),
            BackgroundTag::LinearGradient => BackgroundKind::LinearGradient {
                angle: self.gradient_angle_or_pattern_height,
                stops: self.colors,
            },
            BackgroundTag::PatternSlash => {
                // `pattern_slash` packs both values into one f32 as `(width * 255) * 0xFFFF + (interval * 255)`.
                // floor + rem_euclid to invert it since that's the pairing that stays correct for negative inputs.
                // truncation and `%` give the wrong entry.
                let packed = self.gradient_angle_or_pattern_height;
                BackgroundKind::PatternSlash {
                    color: self.solid,
                    width: (packed / 0xFFFF as f32).floor() / 255.0,
                    interval: (packed.rem_euclid(0xFFFF as f32)) / 255.0,
                }
            }
            BackgroundTag::Checkerboard => BackgroundKind::Checkerboard {
                color: self.solid,
                size: self.gradient_angle_or_pattern_height,
            },
        }
    }

    /// Use specified color space for color interpolation.
    ///
    /// <https://developer.mozilla.org/en-US/docs/Web/CSS/color-interpolation-method>
    pub fn color_space(mut self, color_space: ColorSpace) -> Self {
        self.color_space = color_space;
        self
    }

    /// The color space used to interpolate this background, set by [`Background::color_space`].
    pub fn interpolation_space(&self) -> ColorSpace {
        self.color_space
    }

    /// Returns a new background color with the same hue, saturation, and lightness, but with a modified alpha value.
    pub fn opacity(&self, factor: f32) -> Self {
        let mut background = *self;
        background.solid = background.solid.opacity(factor);
        background.colors = [
            self.colors[0].opacity(factor),
            self.colors[1].opacity(factor),
        ];
        background
    }

    /// Returns whether the background color is transparent.
    pub fn is_transparent(&self) -> bool {
        match self.tag {
            BackgroundTag::Solid => self.solid.a == 0.,
            BackgroundTag::LinearGradient => self.colors.iter().all(|c| c.color.a == 0.),
            BackgroundTag::PatternSlash => self.solid.a == 0.,
            BackgroundTag::Checkerboard => self.solid.a == 0.,
        }
    }
}

impl From<Hsla> for Background {
    fn from(value: Hsla) -> Self {
        Self {
            tag: BackgroundTag::Solid,
            solid: value,
            ..Default::default()
        }
    }
}

impl From<Rgba> for Background {
    fn from(value: Rgba) -> Self {
        Hsla::from(value).into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_background_solid() {
        let color: Hsla = rgba(0xff0099ff).into_color();
        let mut background = Background::from(color);
        assert_eq!(background.tag, BackgroundTag::Solid);
        assert_eq!(background.solid, color);

        assert_eq!(background.opacity(0.5).solid, color.opacity(0.5));
        assert!(!background.is_transparent());
        background.solid = hsla(0.0, 0.0, 0.0, 0.0);
        assert!(background.is_transparent());
    }

    #[test]
    fn test_background_linear_gradient() {
        let from = linear_color_stop(rgba(0xff0099ff), 0.0);
        let to = linear_color_stop(rgba(0x00ff99ff), 1.0);
        let background = linear_gradient(90.0, from, to);
        assert_eq!(background.tag, BackgroundTag::LinearGradient);
        assert_eq!(background.colors[0], from);
        assert_eq!(background.colors[1], to);

        assert_eq!(background.opacity(0.5).colors[0], from.opacity(0.5));
        assert_eq!(background.opacity(0.5).colors[1], to.opacity(0.5));
        assert!(!background.is_transparent());
        assert!(background.opacity(0.0).is_transparent());
    }

    #[test]
    fn test_background_kind() {
        let color: Hsla = rgba(0xff0099ff).into_color();
        assert_eq!(Background::from(color).kind(), BackgroundKind::Solid(color));

        let from = linear_color_stop(rgba(0xff0099ff), 0.0);
        let to = linear_color_stop(rgba(0x00ff99ff), 1.0);
        assert_eq!(
            linear_gradient(90.0, from, to).kind(),
            BackgroundKind::LinearGradient {
                angle: 90.0,
                stops: [from, to],
            }
        );

        assert_eq!(
            checkerboard(color, 12.0).kind(),
            BackgroundKind::Checkerboard { color, size: 12.0 }
        );
    }

    #[test]
    fn test_background_kind_unpacks_pattern_slash() {
        let color: Hsla = rgba(0xff0099ff).into_color();
        // Both values survive to the 1/255 the constructor quantizes them to.
        for (width, interval) in [(1.0, 3.0), (0.5, 0.25), (2.0, 10.0)] {
            let BackgroundKind::PatternSlash {
                width: got_width,
                interval: got_interval,
                ..
            } = pattern_slash(color, width, interval).kind()
            else {
                panic!("pattern_slash did not produce a PatternSlash");
            };
            assert!((got_width - width).abs() <= 1.0 / 255.0);
            assert!((got_interval - interval).abs() <= 1.0 / 255.0);
        }
    }

    #[test]
    fn test_rgba_alpha() {
        let color = Rgba::new(0.2, 0.6, 1.0, 0.8);
        assert_eq!(color.alpha(0.25).a, 0.25);
        assert_eq!(color.alpha(1.5).a, 1.0);
    }

    #[test]
    fn test_rgba_opacity() {
        let color = Rgba::new(0.2, 0.6, 1.0, 0.8);
        assert!((color.opacity(0.5).a - 0.4).abs() < 1e-6);
        assert_eq!(color.opacity(2.0).a, 0.8);
    }
}
