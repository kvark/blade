# Isolated CPU-only allocation observability

This diagnostic branch starts at f6f2729e. Latest upstream 68a23e49 differs only
in the unused renderer; no native runtime fix is omitted. Preserve the user's
original Blade worktree, Kindle main, all historical fixtures and failed jobs.

Add observation only: flushed allocation/map/bind boundaries and cached actual
Vulkan memory placement. Do not change allocation strategy, flags, zeroing,
precision, queue behavior, error handling or GPU lifetime. CPU compilation and
unit tests are permitted. No GPU job, driver query, retry, recovery or adoption
is authorized. The active incident boot is 372a5604 and still faulted.

Keep any future GPU diagnostic separately declared, with explicit recovery,
matched control instrumentation and complete existing qualification gates.
