struct StorageBuffer {
    foo: vec2<f32>,
}

#header

fn test_plain(i1: i32, i4: vec4<i32>, u2: vec2<u32>, f1: f32, f4: vec4<f32>) {}
fn test_buffer(buf: StorageBuffer) {}

@compute @workgroup_size(1, 2, 3)
fn main() {
    test_plain($i1, $i4, $u2, $f1, $f4);
    test_buffer($buffer);
}
