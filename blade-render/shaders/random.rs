#[derive(Clone, Copy, Default)]
pub struct RandomState {
    pub seed: u32,
    pub index: u32,
}

fn hash_jenkins(value: u32) -> u32 {
    let mut a = value;
    // http://burtleburtle.net/bob/hash/integer.html
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    a
}

fn rot32(x: u32, bits: u32) -> u32 {
    x.rotate_left(bits)
}

pub fn random_init(pixel_index: u32, frame_index: u32) -> RandomState {
    RandomState {
        seed: hash_jenkins(pixel_index) + frame_index,
        index: 0,
    }
}

fn murmur3(rng: &mut RandomState) -> u32 {
    let c1 = 0xcc9e2d51u32;
    let c2 = 0x1b873593u32;
    let r1 = 15u32;
    let r2 = 13u32;
    let m = 5u32;
    let n = 0xe6546b64u32;

    let mut hash = rng.seed;
    rng.index += 1;
    let mut k = rng.index;
    k *= c1;
    k = rot32(k, r1);
    k *= c2;

    hash ^= k;
    hash = rot32(hash, r2) * m + n;

    hash ^= 4;
    hash ^= hash >> 16;
    hash *= 0x85ebca6b;
    hash ^= hash >> 13;
    hash *= 0xc2b2ae35;
    hash ^= hash >> 16;

    hash
}

pub fn random_gen(rng: &mut RandomState) -> f32 {
    let v = murmur3(rng);
    let one = 1.0f32.to_bits();
    let mask = (1u32 << 23) - 1;
    f32::from_bits((mask & v) | one) - 1.0
}
