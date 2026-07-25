# Study repository layout

A dedicated study repository with pinned submodules is the recommended final
layout.

```text
blade-sync-study/
  .gitmodules
  engines/
    blade/                  # pinned benchmark commit
    wgpu/                   # pinned benchmark commit based on v30
  paper/
  scripts/
  data/
    raw/
    derived/
  captures/                 # manifest or external-storage pointers
  README.md
```

This keeps research scripts, the manuscript, and data independent of either
upstream project while making the two implementation revisions explicit.
Submodules record commit IDs; branch names are only development conveniences.

The two benchmark working trees must first be committed and pushed:

- `kvark/blade`, branch `blade-sync-study`;
- `gfx-rs/wgpu` or an accessible fork, branch `blade-sync-study`, based on
  `v30`.

Then create the study repository and pin the commits:

```sh
git init blade-sync-study
cd blade-sync-study
git submodule add https://github.com/kvark/blade engines/blade
git submodule add https://github.com/gfx-rs/wgpu engines/wgpu
git -C engines/blade checkout BLADE_BENCHMARK_COMMIT
git -C engines/wgpu checkout WGPU_BENCHMARK_COMMIT
git add .gitmodules engines/blade engines/wgpu
```

If the wgpu benchmark branch is hosted on a fork, use that fork as the
submodule URL. Do not configure submodules to follow moving branches for paper
collection.

After the two commits exist, move the current `paper/` directory into the
superproject and adjust its defaults to `engines/blade` and `engines/wgpu`.
The benchmark programs remain inside their respective submodules so each can
be built and profiled independently.

Commit small raw CSV files and metadata directly if repository size remains
reasonable. Store large RenderDoc, RGP, Nsight, and Tracy captures in release
assets, institutional storage, or an archival dataset, with hashes and URLs in
the repository. Do not hide raw numeric measurements behind Git LFS unless the
archive also has a durable non-GitHub location.

Participants clone the complete artifact with:

```sh
git clone --recurse-submodules STUDY_REPOSITORY_URL
```

and update an existing clone with:

```sh
git submodule update --init --recursive
```
