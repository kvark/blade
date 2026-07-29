# Study repository layout

The submitted study uses two content-addressed repositories rather than a
third superproject:

- `kvark/blade`, branch `blade-sync-study`, contains the Blade benchmark,
  collectors, analysis, generated tables, and manuscript; tag
  `sync-study-v1` pins the study code;
- `kvark/wgpu`, branch `blade-sync-study`, tag `blade-sync-study-v1`, contains
  the matched wgpu benchmark.

Branch and tag names are convenience pointers. The full commit hashes in the
paper, collection manifests, and bibliography are the identifiers of record.
The measured revisions are Blade
`87ed06750877f336ad1c54fa5005a0b799f488d7` and wgpu
`7d37a77c086f3d3b8a9dbda6b476cd7ac195fcfc`. The tags pin code rather than
the submitted article revision, which arXiv preserves independently.

The raw measurements are not Git objects. They ship as a checksummed arXiv
ancillary archive and extract to `paper/data/raw/`; generated tables remain
rebuildable from that directory. This keeps the 160 RenderDoc captures and
thousands of CSV observations out of repository history while preserving the
irreplaceable data with the paper itself.

A reader inspecting the measured code can clone both repositories as siblings:

```sh
git clone --branch sync-study-v1 https://github.com/kvark/blade
git clone --branch blade-sync-study-v1 https://github.com/kvark/wgpu
```

and then extract the ancillary data under `blade/paper/data/raw/`. A
superproject with pinned submodules could provide a one-command checkout, but
it would not add archival integrity beyond the recorded commit hashes and
would introduce a third repository that the submitted artifact does not need.
