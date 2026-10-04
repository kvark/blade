#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct Globals {
    mvp_transform: [[f32; 4]; 4],
}

#[derive(blade_macros::ShaderData)]
#[allow(dead_code)]
struct ShaderParams {
    globals: Globals,
    sprite_texture: blade_graphics::TextureView,
    sprite_sampler: blade_graphics::Sampler,
}

/// A vector from a math library that converts into `mint`, as glam's and
/// synaga's do. The `Vertex` derive finds its format through `mint`.
#[derive(Clone, Copy)]
struct Position([f32; 3]);
impl From<Position> for mint::Vector3<f32> {
    fn from(p: Position) -> Self {
        p.0.into()
    }
}
impl mint::IntoMint for Position {
    type MintType = mint::Vector3<f32>;
}

#[derive(blade_macros::Vertex)]
#[allow(dead_code)]
struct MixedVertex {
    pos: Position,
    tex_coords: mint::Vector2<f32>,
    color: u32,
    normal: [i32; 3],
}

#[test]
fn test_vertex_formats() {
    use blade_graphics::{Vertex as _, VertexFormat as Vf};

    let layout = MixedVertex::layout();
    let formats = layout
        .attributes
        .iter()
        .map(|&(name, attribute)| (name, attribute.format))
        .collect::<Vec<_>>();
    assert_eq!(
        formats,
        [
            ("pos", Vf::F32Vec3),
            ("tex_coords", Vf::F32Vec2),
            ("color", Vf::U32),
            ("normal", Vf::I32Vec3),
        ]
    );
}

#[derive(blade_macros::Flat, PartialEq, Debug)]
struct FlatData<'a> {
    array: [u32; 2],
    single: f32,
    slice: &'a [u16],
}

#[test]
fn test_flat_struct() {
    use blade_asset::Flat;

    let data = FlatData {
        array: [1, 2],
        single: 3.0,
        slice: &[4, 5, 6],
    };
    let mut vec = vec![0u8; data.size()];
    unsafe { data.write(vec.as_mut_ptr()) };
    let other = unsafe { Flat::read(vec.as_ptr()) };
    assert_eq!(data, other);
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(u32)]
#[non_exhaustive]
enum Foo {
    #[allow(dead_code)]
    A,
    B,
}

#[derive(blade_macros::Flat, Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
struct FooWrap(Foo);

#[test]
fn test_flat_wrap() {
    use blade_asset::Flat;

    let foo = FooWrap(Foo::B);
    let mut vec = vec![0u8; foo.size()];
    unsafe { foo.write(vec.as_mut_ptr()) };
    let other = unsafe { Flat::read(vec.as_ptr()) };
    assert_eq!(foo, other);
}
