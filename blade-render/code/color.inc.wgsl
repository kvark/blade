fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let c = v * s;
    let x = c * (1.0 - abs((h / 60.0) % 2.0 - 1.0));
    var q = vec3<f32>(v - c);
    if (h < 60.0) {
        q.r += c; q.g += x;
    } else if (h < 120.0) {
        q.g += c; q.r += x;
    } else if (h < 180.0) {
        q.g += c; q.b += x;
    } else if (h < 240.0) {
        q.b += c; q.g += x;
    } else if (h < 300.0) {
        q.b += c; q.r += x;
    } else {
        q.r += c; q.b += x;
    }
    return q;
}
