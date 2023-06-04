const PI: f32 = 3.1415926;

var env_weights: texture_2d<f32>;

struct RandomState {
    seed: u32,
    index: u32,
}

// 32 bit Jenkins hash
fn hash_jenkins(value: u32) -> u32 {
    var a = value;
    // http://burtleburtle.net/bob/hash/integer.html
    a = (a + 0x7ed55d16u) + (a << 12u);
    a = (a ^ 0xc761c23cu) ^ (a >> 19u);
    a = (a + 0x165667b1u) + (a << 5u);
    a = (a + 0xd3a2646cu) ^ (a << 9u);
    a = (a + 0xfd7046c5u) + (a << 3u);
    a = (a ^ 0xb55a4f09u) ^ (a >> 16u);
    return a;
}

fn random_init(pixel_index: u32, frame_index: u32) -> RandomState {
    var rs: RandomState;
    rs.seed = hash_jenkins(pixel_index) + frame_index;
    rs.index = 0u;
    return rs;
}

fn rot32(x: u32, bits: u32) -> u32 {
    return (x << bits) | (x >> (32u - bits));
}

// https://en.wikipedia.org/wiki/MurmurHash
fn murmur3(rng: ptr<function, RandomState>) -> u32 {
    let c1 = 0xcc9e2d51u;
    let c2 = 0x1b873593u;
    let r1 = 15u;
    let r2 = 13u;
    let m = 5u;
    let n = 0xe6546b64u;

    var hash = (*rng).seed;
    (*rng).index += 1u;
    var k = (*rng).index;
    k *= c1;
    k = rot32(k, r1);
    k *= c2;

    hash ^= k;
    hash = rot32(hash, r2) * m + n;

    hash ^= 4u;
    hash ^= (hash >> 16u);
    hash *= 0x85ebca6bu;
    hash ^= (hash >> 13u);
    hash *= 0xc2b2ae35u;
    hash ^= (hash >> 16u);

    return hash;
}

fn random_gen(rng: ptr<function, RandomState>) -> f32 {
    let v = murmur3(rng);
    let one = bitcast<u32>(1.0);
    let mask = (1u << 23u) - 1u;
    return bitcast<f32>((mask & v) | one) - 1.0;
}

@vertex
fn vs_importance(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var rng = random_init(vi, 0u);
    var mip = i32(textureNumLevels(env_weights));
    var itc = vec2<i32>(0);
    // descend through the mip chain to find a concrete pixel
    while (mip != 0) {
        mip -= 1;
        let weights = textureLoad(env_weights, itc, mip);
        let sum = dot(vec4<f32>(1.0), weights);
        let r = random_gen(&rng) * sum;
        itc *= 2;
        if (r >= weights.x+weights.y) {
            itc.y += 1;
            itc.x += i32(r >= weights.x+weights.y+weights.z);
        } else {
            itc.x += i32(r >= weights.x);
        }
    }

    let extent = textureDimensions(env_weights, 0);
    let relative = (vec2<f32>(itc) + vec2<f32>(0.5)) / vec2<f32>(extent);
    return vec4<f32>(relative.x - 1.0, 1.0 - relative.y, 0.0, 1.0);
}

@fragment
fn fs_importance() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}


fn map_equirect_dir_to_uv(dir: vec3<f32>) -> vec2<f32> {
    //Note: Y axis is up
    let yaw = asin(dir.y);
    let pitch = atan2(dir.x, dir.z);
    return vec2<f32>(pitch + PI, -2.0 * yaw + PI) / (2.0 * PI);
}
fn map_equirect_uv_to_dir(uv: vec2<f32>) -> vec3<f32> {
    let yaw = PI * (0.5 - uv.y);
    let pitch = 2.0 * PI * (uv.x - 0.5);
    return vec3<f32>(cos(yaw) * sin(pitch), sin(yaw), cos(yaw) * cos(pitch));
}

struct UvOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_uv(@builtin(vertex_index) vi: u32) -> UvOutput {
    var vo: UvOutput;
    let uv = vec2<f32>(2.0 * f32(vi & 1u), f32(vi & 2u));
    vo.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    vo.uv = uv;
    return vo;
}

@fragment
fn fs_uv(input: UvOutput) -> @location(0) vec4<f32> {
    let dir = map_equirect_uv_to_dir(input.uv);
    let uv = map_equirect_dir_to_uv(dir);
    return vec4<f32>(uv, length(uv - input.uv), 0.0);
}
