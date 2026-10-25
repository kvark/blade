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
#[test]
fn inspect_shader_uniform_bytes() {
    #[derive(blade_macros::ShaderData)]
    struct Data {
        dimensions: [u32; 4],
        scale: f32,
    }
    struct Capture(Vec<(u32, Vec<u8>)>);
    impl blade_graphics::ShaderDataVisitor for Capture {
        fn buffer(&mut self, _: u32, _: blade_graphics::BufferPiece) {
            panic!("unexpected buffer");
        }
        fn plain(&mut self, index: u32, bytes: &[u8]) {
            self.0.push((index, bytes.to_vec()));
        }
    }
    let data = Data {
        dimensions: [3, 7, 11, 0],
        scale: -0.25,
    };
    let mut capture = Capture(Vec::new());
    blade_graphics::ShaderData::visit(&data, &mut capture);
    assert_eq!(capture.0.len(), 2);
    assert_eq!(
        capture.0[0],
        (0, bytemuck::bytes_of(&data.dimensions).to_vec())
    );
    assert_eq!(capture.0[1], (1, bytemuck::bytes_of(&data.scale).to_vec()));
}
