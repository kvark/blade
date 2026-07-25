# Measurement data

`raw/` is **not tracked by git**. Collections are large, they grow with every
rerun, and every manifest pins the exact source revision, machine metadata, and
command line, so a collection can be reproduced rather than shipped. Keep them
here locally, and move them to a study artifact when the paper is circulated.

`derived/` is regenerated from `raw/` by `analyze.py` and `build-tables.py` and
is likewise untracked. The paper `\input`s `derived/tables/*.tex`, so run
`build-tables.py` before `latexmk`.

What must survive for a result to be admissible is listed in
[`../experiments.md`](../experiments.md): raw rows, machine metadata, source
revision, and the analysis command. All four live in the collection directory.

Software-renderer output may be retained under a directory clearly named
`correctness-only`, but it must not feed performance figures.
