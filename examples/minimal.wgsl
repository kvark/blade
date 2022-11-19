struct Output {
    avg: array<u32>,
};

#header

@compute
@workgroup_size(8, 8)
fn main() {
    $output.avg[0] = 5;
}
