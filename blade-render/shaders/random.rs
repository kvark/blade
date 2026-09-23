use synaga_shader::*;

#[derive(Clone, Copy, Default)]
pub struct RandomState {
    pub seed: u32,
    pub index: u32,
}

#[shader]
pub fn hash_jenkins(value: u32) -> u32 {
    let mut a = value;
    // http://burtleburtle.net/bob/hash/integer.html
    a = (a + 0x7ed55d16u32) + (a << 12u32);
    a = (a ^ 0xc761c23cu32) ^ (a >> 19u32);
    a = (a + 0x165667b1u32) + (a << 5u32);
    a = (a + 0xd3a2646cu32) ^ (a << 9u32);
    a = (a + 0xfd7046c5u32) + (a << 3u32);
    a = (a ^ 0xb55a4f09u32) ^ (a >> 16u32);
    return a;
}

#[shader]
pub fn rot32(x: u32, bits: u32) -> u32 {
    return (x << bits) | (x >> (32u32 - bits));
}

#[shader]
pub fn random_init(pixel_index: u32, frame_index: u32) -> RandomState {
    let mut rs = RandomState::default();
    rs.seed = hash_jenkins(pixel_index) + frame_index;
    rs.index = 0u32;
    return rs;
}

#[shader]
pub fn murmur3(rng: &mut RandomState) -> u32 {
    let c1 = 0xcc9e2d51u32;
    let c2 = 0x1b873593u32;
    let r1 = 15u32;
    let r2 = 13u32;
    let m = 5u32;
    let n = 0xe6546b64u32;

    let mut hash = rng.seed;
    rng.index += 1u32;
    let mut k = rng.index;
    k *= c1;
    k = rot32(k, r1);
    k *= c2;

    hash ^= k;
    hash = rot32(hash, r2) * m + n;

    hash ^= 4u32;
    hash ^= (hash >> 16u32);
    hash *= 0x85ebca6bu32;
    hash ^= (hash >> 13u32);
    hash *= 0xc2b2ae35u32;
    hash ^= (hash >> 16u32);

    return hash;
}

#[shader]
pub fn random_gen(rng: &mut RandomState) -> f32 {
    let v = murmur3(rng);
    let one = bitcast::<u32>(1.0);
    let mask = (1u32 << 23u32) - 1u32;
    return bitcast::<f32>((mask & v) | one) - 1.0;
}
