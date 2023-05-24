#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct Globals {
    mvp_transform: [[f32; 4]; 4],
}

#[derive(blade_macros::ShaderData)]
struct ShaderParams {
    globals: Globals,
    sprite_texture: blade_graphics::TextureView,
    sprite_sampler: blade_graphics::Sampler,
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
