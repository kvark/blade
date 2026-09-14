//! Temporary, always-on allocation diagnostics in this isolated branch only.

use std::{
    fmt,
    io::{self, Write},
    sync::atomic::{AtomicU64, Ordering},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

pub(super) struct Trace {
    id: u64,
    started: Instant,
}

pub(super) struct Hex<'a>(pub &'a str);

impl fmt::Display for Hex<'_> {
    fn fmt(&self, writer: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0.as_bytes() {
            write!(writer, "{byte:02x}")?;
        }
        Ok(())
    }
}

fn write_event(
    writer: &mut impl Write,
    header: fmt::Arguments<'_>,
    data: fmt::Arguments<'_>,
) -> io::Result<()> {
    writeln!(writer, "blade_memory_v1 {header} {data}")?;
    writer.flush()
}

impl Trace {
    pub(super) fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self {
            id: NEXT.fetch_add(1, Ordering::Relaxed),
            started: Instant::now(),
        }
    }

    pub(super) fn id(&self) -> u64 {
        self.id
    }

    pub(super) fn mark(&self, phase: &str, data: fmt::Arguments<'_>) {
        write_event(
            &mut io::stderr().lock(),
            format_args!(
                "pid={} request={} unix_ns={} elapsed_ns={} phase={phase}",
                std::process::id(),
                self.id,
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
                self.started.elapsed().as_nanos(),
            ),
            data,
        )
        .expect("allocation diagnostic could not be flushed; stopping");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Sink {
        bytes: Vec<u8>,
        flushes: usize,
    }

    impl Write for Sink {
        fn write(&mut self, data: &[u8]) -> io::Result<usize> {
            self.bytes.extend_from_slice(data);
            Ok(data.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            self.flushes += 1;
            Ok(())
        }
    }

    #[test]
    fn names_cannot_inject_a_record() {
        assert_eq!(format!("{}", Hex("a\nb\tλ")), "610a6209cebb");
        assert_eq!(format!("{}", Hex("")), "");
    }

    #[test]
    fn records_are_flushed_and_single_line() {
        let mut sink = Sink::default();
        write_event(
            &mut sink,
            format_args!("request=3 phase=map.before"),
            format_args!("name_hex={} bytes=4", Hex("a\nb")),
        )
        .unwrap();
        assert_eq!(sink.flushes, 1);
        assert_eq!(
            sink.bytes,
            b"blade_memory_v1 request=3 phase=map.before name_hex=610a62 bytes=4\n"
        );
    }

    #[test]
    fn write_failure_is_not_silenced() {
        struct Broken;
        impl Write for Broken {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                Err(io::ErrorKind::BrokenPipe.into())
            }
            fn flush(&mut self) -> io::Result<()> {
                panic!("must not flush a failed write")
            }
        }
        assert!(
            write_event(
                &mut Broken,
                format_args!("request=1"),
                format_args!("bytes=4")
            )
            .is_err()
        );
    }

    #[test]
    fn flush_failure_is_not_silenced() {
        struct Broken;
        impl Write for Broken {
            fn write(&mut self, data: &[u8]) -> io::Result<usize> {
                Ok(data.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Err(io::ErrorKind::BrokenPipe.into())
            }
        }
        assert!(
            write_event(
                &mut Broken,
                format_args!("request=1"),
                format_args!("bytes=4")
            )
            .is_err()
        );
    }

    #[test]
    fn request_ids_are_distinct() {
        let first = Trace::new();
        let second = Trace::new();
        assert_ne!(first.id(), second.id());
    }
}
