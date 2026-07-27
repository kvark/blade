# Measurement data

`raw/` is **not tracked by git**. Collections are large and grow with every
rerun. Every manifest pins the exact source revision, machine metadata, and
command line, but that does not replace the observations: hardware and driver
state may not be reproducible later. Keep them here locally during collection,
then publish the immutable directories and checksums as a versioned study
artifact before circulating the paper.

`derived/` is regenerated from `raw/` by `analyze.py` and `build-tables.py` and
is likewise untracked. The paper `\input`s `derived/tables/*.tex`, so run
`build-tables.py` before `latexmk`.

What must survive for a result to be admissible is listed in
[`../experiments.md`](../experiments.md): raw rows, machine metadata, source
revision, and the analysis command. A local ignored directory is not sufficient
unless it is also backed up and included in the published artifact.

Software-renderer output may be retained under a directory clearly named
`correctness-only`, but it must not feed performance figures.
