# Combined initialization control, CPU preparation only

This worktree starts at the source of published Blade 0.9.0, not latest main.
It is the matched historical control. Carry only the existing flushed allocator,
buffer and bind observability. Keep allocator policy and execution unchanged.
New upstream shader-validation fixes belong in a separately identified candidate.

No GPU work, host recovery, runtime qualification or adoption is declared here.
Retain all old failures and completed preparations; never rerun their writers.
