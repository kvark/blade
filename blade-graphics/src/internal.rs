/// Concatenated strings with reusable storage.
#[derive(Default)]
pub struct StringBuffer {
    buf: String,
    lens: Vec<usize>,
}

impl StringBuffer {
    pub fn len(&self) -> usize {
        self.lens.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &str> {
        self.lens.iter().scan(0usize, |start, &len| {
            let s = &self.buf[*start..*start + len];
            *start += len;
            Some(s)
        })
    }

    pub fn push(&mut self, s: &str) {
        self.buf.push_str(s);
        self.lens.push(s.len());
    }

    pub fn truncate(&mut self, len: usize) {
        let end: usize = self.lens[..len].iter().sum();
        self.buf.truncate(end);
        self.lens.truncate(len);
    }

    pub fn drain_prefix(&mut self, count: usize) {
        let prefix: usize = self.lens[..count].iter().sum();
        let _ = self.buf.drain(..prefix);
        let _ = self.lens.drain(..count);
    }

    #[allow(dead_code)]
    pub(crate) fn clear(&mut self) {
        self.buf.clear();
        self.lens.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::StringBuffer;

    #[test]
    fn resource_array_runtime_limit_preserves_reused_slots() {
        let mut array = crate::ResourceArray::<u32, 8>::with_limit(2);
        assert!(array.is_empty());
        assert_eq!(array.alloc(10), 0);
        assert_eq!(array.alloc(20), 1);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| array.alloc(30))).is_err()
        );
        array.free(0);
        assert_eq!(array.alloc(40), 0);
        assert_eq!(array[0], 40);
        array.clear();
        assert!(array.is_empty());
        assert_eq!(array.alloc(50), 0);
    }

    #[test]
    fn drain_prefix_keeps_the_rest() {
        let mut buf = StringBuffer::default();
        buf.push("shadow");
        buf.push("opaque");
        buf.push("debug");
        buf.drain_prefix(2);
        assert_eq!(buf.iter().collect::<Vec<_>>(), ["debug"]);

        buf.truncate(0);
        buf.push("shadow");
        buf.push("opaque");
        buf.drain_prefix(0);
        assert_eq!(buf.iter().collect::<Vec<_>>(), ["shadow", "opaque"]);
    }
}
