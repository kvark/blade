// Cooperative matrix multiplication: C = A * B + C
//
// Each workgroup handles one output tile.
// The K dimension is iterated in TILE-wide steps,
// loading tiles of A and B and accumulating into C.
//
// The host substitutes placeholders based on device capabilities:
//   ENABLE_F16   - "enable f16;" or empty
//   TILE_SIZE    - 8u or 16u
//   INPUT_SCALAR - f32 or f16
//   COOP_MAT     - coop_mat8x8 or coop_mat16x16

enable wgpu_cooperative_matrix;
ENABLE_F16

const TILE: u32 = TILE_SIZE;

var<storage, read> matrix_a: array<INPUT_SCALAR>;
var<storage, read> matrix_b: array<INPUT_SCALAR>;
var<storage, read_write> matrix_c: array<f32>;

struct Params {
    m: u32,
    n: u32,
    k: u32,
}
var<storage, read> params: Params;

// Workgroup X must be a multiple of the subgroup size (32 or 64).
// 64 is the LCM of common subgroup sizes.
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>) {
    let row = wg.x * TILE;
    let col = wg.y * TILE;
    let n = params.n;
    let k = params.k;

    // Zero-initialize accumulator (row-major load with stride = n)
    let c_offset = row * n + col;
    var acc = coopLoadT<COOP_MAT<f32, C>>(&matrix_c[c_offset], n);

    // Accumulate tiles along K (row-major loads)
    for (var t: u32 = 0u; t < k; t += TILE) {
        let a = coopLoadT<COOP_MAT<INPUT_SCALAR, A>>(&matrix_a[row * k + t], k);
        let b = coopLoadT<COOP_MAT<INPUT_SCALAR, B>>(&matrix_b[t * n + col], n);
        acc = coopMultiplyAdd(a, b, acc);
    }

    coopStoreT(acc, &matrix_c[c_offset], n);
}
