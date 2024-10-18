var env_weights: texture_2d<f32>;

struct EnvImportantSample {
    pixel: vec2<i32>,
    pdf: f32,
}

// Returns the range of values proportional to the area, given the texel Y
fn compute_latitude_area_bounds(texel_y: i32, dim: u32) -> vec2<f32> {
    return cos(vec2<f32>(vec2<i32>(texel_y, texel_y + 1)) / f32(dim) * PI);
}

fn compute_texel_solid_angle(itc: vec2<i32>, dim: vec2<u32>) -> f32 {
    //Note: this has to agree with `map_equirect_uv_to_dir`
    let meridian_solid_angle = 4.0 * PI / f32(dim.x);
    let bounds = compute_latitude_area_bounds(itc.y, dim.y);
    let meridian_part = 0.5 * (bounds.x - bounds.y);
    return meridian_solid_angle * meridian_part;
}

fn generate_environment_sample(rng: ptr<function, RandomState>, dim: vec2<u32>) -> EnvImportantSample {
    var es = EnvImportantSample();
    es.pdf = 1.0;
    var mip = i32(textureNumLevels(env_weights));
    var itc = vec2<i32>(0);
    // descend through the mip chain to find a concrete pixel
    while (mip != 0) {
        mip -= 1;
        let weights = textureLoad(env_weights, itc, mip);
        let sum = dot(vec4<f32>(1.0), weights);
        let r = random_gen(rng) * sum;
        var weight: f32;
        itc *= 2;
        if (r >= weights.x+weights.y) {
            itc.y += 1;
            if (r >= weights.x+weights.y+weights.z) {
                weight = weights.w;
                itc.x += 1;
            } else {
                weight = weights.z;
            }
        } else {
            if (r >= weights.x) {
                weight = weights.y;
                itc.x += 1;
            } else {
                weight = weights.x;
            }
        }
        es.pdf *= weight / sum;
    }

    // adjust for the texel's solid angle
    es.pdf /= compute_texel_solid_angle(itc, dim);
    es.pixel = itc;
    return es;
}

fn compute_environment_sample_pdf(pixel: vec2<i32>, dim: vec2<u32>) -> f32 {
    var itc = pixel;
    var pdf = 1.0 / compute_texel_solid_angle(itc, dim);
    let mip_count = i32(textureNumLevels(env_weights));
    for (var mip = 0; mip < mip_count; mip += 1) {
        let rem = itc & vec2<i32>(1);
        itc >>= vec2<u32>(1u);
        let weights = textureLoad(env_weights, itc, mip);
        let sum = dot(vec4<f32>(1.0), weights);
        let w2 = select(weights.xy, weights.zw, rem.y != 0);
        let weight = select(w2.x, w2.y, rem.x != 0);
        pdf *= weight / sum;
    }
    return pdf;
}
