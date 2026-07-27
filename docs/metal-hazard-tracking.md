# Metal untracked hazard mode on Apple M3

Investigation date: 2026-07-24  
Blade revision: `ba0fb5a`  
Test system: 13-inch MacBook Air (`Mac15,12`), Apple M3 (10-core GPU),
16 GB unified memory, macOS 15.7.3 (24G419), running on battery with Low
Power Mode enabled.

## Summary

Blade should keep using Metal's tracked resource mode.

In this single pilot session, disabling hazard tracking did not show a
consistent independent-pass improvement in CPU encoding, command-buffer commit,
or GPU time. Adding the synchronization that untracked dependent resources
require was substantially slower: at 100 passes,
explicit fences and events cost about 8.5–8.9 times as much CPU encoding time
and about 10 times as much GPU time as Metal's automatic tracking in this
microbenchmark.

This differs from Blade's Vulkan `manual_barriers` optimization. The automatic
Vulkan path inserts a global `ALL_COMMANDS` barrier before every pass. Metal's
automatic tracker instead knows which resources each encoder uses and only
orders real conflicts. Independent tracked resources already expose independent
work to the driver.

There is also an API mismatch. `manual_barriers` belongs to
`CommandEncoderDesc`, but Metal hazard tracking is chosen when each buffer or
texture is created. A resource can outlive one encoder and be used by encoders
with different settings, so the command-encoder flag cannot safely select the
resource's tracking mode.

## Current Blade behavior

Commit `16f0b08` added `CommandEncoderDesc::manual_barriers`. Vulkan stores the
flag and skips its automatic barrier at the beginning of each pass. Metal
currently ignores the flag and `CommandEncoder::barrier()` is a no-op.

That is semantically safe on Metal 3 because Blade creates individual buffers
and textures from `MTLDevice` with the default hazard mode. Individual device
resources default to `MTLHazardTrackingModeTracked`, so Metal resolves
inter-encoder conflicts automatically.

It would not be sufficient to add the
`MTLResourceHazardTrackingModeUntracked` option to resource creation:

1. Untracked resources need explicit synchronization for every real
   read/write or write/write conflict.
2. An `MTLFence` must be updated by the producing pass before that pass's
   encoder ends, then waited on by the consuming pass.
3. Blade's public `encoder.barrier()` is called between RAII pass encoders,
   after the producer has already ended. It is therefore too late to add the
   producer-side fence update.
4. An `MTLEvent` signal/wait pair can be encoded between pass encoders and
   therefore fits the current API, but it has broader scope and was expensive
   in the benchmark below.

## Benchmark

The standalone harness is
[`paper/tools/metal-hazard-bench.swift`](../paper/tools/metal-hazard-bench.swift).
Build and run it with:

```sh
xcrun swiftc -O paper/tools/metal-hazard-bench.swift \
  -o /tmp/metal-hazard-bench
/tmp/metal-hazard-bench \
  --raw-output paper/data/raw/<collection-id>/metal-hazard-r01-raw.csv \
  | tee paper/data/raw/<collection-id>/metal-hazard-r01-summary.csv
```

The raw-output path is required and must be new, preventing a future run from
retaining only summary medians. Repeat with fresh `r02` through `r10` paths for
publication-quality process-level replication.

Each pass is a separate compute encoder containing one single-thread dispatch. Two
workloads are measured:

- **Independent:** every pass increments its own four-byte buffer. No
  synchronization is logically necessary.
- **Dependent:** every pass increments the same buffer. The tracked case relies
  on Metal; the untracked cases update/wait on one `MTLFence`, or signal/wait on
  one `MTLEvent`, between consecutive encoders.

Resources and synchronization objects are reused. Each case receives 10 warmup
runs. Cases are interleaved in alternating order to reduce mode-order and
thermal bias. Results are medians of 200 samples, except the 500-pass cases,
which use 80 samples. CPU encoding excludes `commit()`. Commit time is measured
separately. GPU time comes from `gpuStartTime` and `gpuEndTime`.

The harness queries every test buffer's `hazardTrackingMode` and confirmed that
the requested tracked and untracked modes were active.

### Independent passes

| Passes | Encode tracked | Encode untracked | Delta | GPU tracked | GPU untracked | Delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.125 µs | 4.125 µs | 0.0% | 6.417 µs | 6.417 µs | 0.0% |
| 10 | 6.958 µs | 7.000 µs | +0.6% | 9.458 µs | 9.417 µs | -0.4% |
| 100 | 117.709 µs | 118.917 µs | +1.0% | 310.500 µs | 309.250 µs | -0.4% |
| 500 | 546.292 µs | 541.625 µs | -0.9% | 1,594.542 µs | 1,591.750 µs | -0.2% |

Command-buffer commit medians for tracked/untracked were 1.875/1.834 µs,
2.209/2.125 µs, 15.541/15.542 µs, and 22.000/21.209 µs at 1, 10, 100, and 500
passes respectively. The small differences were inconsistent within this
process's observations and did not translate to an end-to-end win. Independent
process repetitions were not retained, so this is not an equivalence result.

### Dependent passes

| Passes | Mode | Encode | Relative | GPU | Relative | End-to-end |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| 10 | tracked | 6.750 µs | 1.00× | 8.667 µs | 1.00× | 180.958 µs |
| 10 | untracked + fence | 32.833 µs | 4.86× | 83.167 µs | 9.60× | 326.000 µs |
| 10 | untracked + event | 31.833 µs | 4.72× | 83.292 µs | 9.61× | 325.417 µs |
| 100 | tracked | 109.333 µs | 1.00× | 292.500 µs | 1.00× | 809.500 µs |
| 100 | untracked + fence | 971.000 µs | 8.88× | 2,955.375 µs | 10.10× | 4,616.042 µs |
| 100 | untracked + event | 928.041 µs | 8.49× | 3,024.417 µs | 10.34× | 4,611.000 µs |
| 500 | tracked | 489.084 µs | 1.00× | 1,450.500 µs | 1.00× | 2,583.417 µs |
| 500 | untracked + fence | 4,255.417 µs | 8.70× | 16,155.167 µs | 11.14× | 21,093.584 µs |
| 500 | untracked + event | 4,092.167 µs | 8.37× | 16,227.000 µs | 11.19× | 21,138.625 µs |

The harness also contains an intentionally unsynchronized untracked canary. It
happened to produce the expected values on this machine and workload. That is
not evidence that the mode is safe: Metal explicitly makes the application
responsible for these dependencies, and a different workload, schedule, GPU, or
OS version may expose the race.

The machine was in Low Power Mode, so these absolute times are not peak M3
numbers. Interleaving reduces order bias but does not remove thermal, power, or
process-level confounding. Only the printed summaries from this session were
retained. Results should be repeated in fresh processes on AC power with Low
Power Mode disabled, retaining the harness's new raw CSV output, before
publishing performance or equivalence claims.

## Recommendation

Do not wire `CommandEncoderDesc::manual_barriers` to untracked Metal resources,
and do not add automatic Metal fences or events between Blade passes. The
pilot did not reveal an independent-work upside, explicit synchronization was
costly in the dependent cases, and the existing API cannot express
producer-side fence placement correctly.

If a future workload demonstrates meaningful Metal hazard-tracking overhead,
the safer experiment would be a resource-level opt-in paired with an explicit
pass-dependency API. That API needs to identify producer and consumer passes
before their encoders end, allowing Metal to use narrowly scoped fences. It
should be evaluated on a real frame graph with a small number of genuine
dependencies, not a barrier after every pass.

Metal 4 is a separate design point: tracked resource mode does not affect work
submitted to an `MTL4CommandQueue`, whose explicit queue barriers need their own
backend design.

## Article angle

**Suggested title:** “When a Vulkan Optimization Becomes a Metal
Pessimization”

The useful story is not that manual synchronization is universally faster. It
is that the cost model follows the abstraction:

1. Blade's Vulkan fallback is deliberately coarse, so removing global barriers
   can expose overlap.
2. Metal's tracker already has resource-level knowledge, so independent work is
   not blocked by unrelated resources.
3. Opting out throws away that knowledge and makes the application reconstruct
   the dependency graph with synchronization primitives.
4. On this M3, reconstructing the graph costs much more than the tracking it
   replaces.

## References

- [Apple: `MTLHazardTrackingMode`](https://developer.apple.com/documentation/metal/mtlhazardtrackingmode)
- [Apple: untracked hazard mode](https://developer.apple.com/documentation/metal/mtlhazardtrackingmode/untracked)
- [Apple: resource synchronization](https://developer.apple.com/documentation/metal/resource-synchronization)
- [Apple: `MTLFence`](https://developer.apple.com/documentation/metal/mtlfence)
- [Apple: synchronization events](https://developer.apple.com/documentation/metal/about-synchronization-events)
