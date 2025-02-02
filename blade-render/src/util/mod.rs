mod frame_pacer;

pub use self::frame_pacer::*;

pub fn align_to(offset: u64, alignment: u64) -> u64 {
    let rem = offset & (alignment - 1);
    if rem == 0 {
        offset
    } else {
        offset - rem + alignment
    }
}
