// Cooperative matrix multiplication: C = A * B
//
// Each workgroup handles one 8x8 output tile.
// The K dimension is iterated in 8-wide steps,
// loading 8x8 tiles of A and B and accumulating into C.

enable wgpu_cooperative_matrix;

const TILE: u32 = 8u;

var<storage, read> matrix_a: array<f32>;
var<storage, read> matrix_b: array<f32>;
var<storage, read_write> matrix_c: array<f32>;

struct Params {
    m: u32,
    n: u32,
    k: u32,
}
var<storage, read> params: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>) {
    let row = wg.x * TILE;
    let col = wg.y * TILE;
    let n = params.n;
    let k = params.k;

    // Zero-initialize accumulator
    let c_offset = row * n + col;
    var acc = coopLoad<coop_mat8x8<f32, C>>(&matrix_c[c_offset], n);

    // Accumulate tiles along K
    for (var t: u32 = 0u; t < k; t += TILE) {
        let a = coopLoad<coop_mat8x8<f32, A>>(&matrix_a[row * k + t], k);
        let b = coopLoad<coop_mat8x8<f32, B>>(&matrix_b[t * n + col], n);
        acc = coopMultiplyAdd(a, b, acc);
    }

    coopStore(acc, &matrix_c[c_offset], n);
}
