# JCGT companion: tracking-free wgpu backend, Bevy cell

This directory is the journal manuscript. It is **not** a rewrite of the
archival arXiv paper (`../main.tex`, 2607.26506). That study isolated barrier
placement and stage/access scope on matched synthetic graphs across six GPUs.
This companion is the implementation-and-application half JCGT wants:

- how Blade’s global pass-boundary barriers are exposed as a wgpu *custom
  backend*, so Bevy (and anything else on wgpu) can run without wgpu-core’s
  per-resource tracker;
- what the bind-group / `ShaderData` impedance actually costs, kept *out* of
  the record-and-submit interval;
- a headless Bevy cell with shadows, GPU mesh preprocessing, a depth/normal
  prepass, and SSAO, collected by the same `../collect.py` as the synthetic
  matrix.

Build:

```bash
latexmk -pdf main.tex
```

One clone:

```bash
git clone --recursive https://github.com/kvark/blade-sync-bench
cd blade-sync-bench
python3 collect.py
```

Sources are also the `jcgt-extension` branch of `kvark/blade`,
`kvark/wgpu`, and the `kvark/bevy` fork as siblings.

Collection protocol: `../COLLECTING.md`, section “Running the combined
protocol on another machine”. Do not feed JCGT Bevy CSVs to
`../build-tables.py`; that script still knows only the synthetic schema.
