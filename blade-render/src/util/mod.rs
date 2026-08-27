mod frame_pacer;
mod rolling;

pub use self::frame_pacer::*;
pub(crate) use self::rolling::RollingBuffer;

pub fn align_to(offset: u64, alignment: u64) -> u64 {
    let rem = offset & (alignment - 1);
    if rem == 0 {
        offset
    } else {
        offset - rem + alignment
    }
}
