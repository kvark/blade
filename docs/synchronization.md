# Synchronization in Blade

Blade deliberately does not reconstruct a state machine for every buffer,
texture, or subresource. On Vulkan, ordinary textures stay in `GENERAL` layout
and pass dependencies use global memory barriers. This keeps command recording
small and predictable, but it means that resource-dependency knowledge must
live in the application or its render graph.

The design and its trade-offs are measured in [Global Pass Barriers Without
Per-Resource RHI Tracking: A Cross-Vendor Study with
Blade](global-pass-barriers.pdf). The benchmark and analysis sources are
preserved on the `blade-sync-study` branch and its `sync-study-v1` code tag.

## What Blade does

By default, the Vulkan encoder places one global memory barrier at each pass
boundary. It now keeps a small encoder-wide bitmask of the kinds of producers
recorded since the previous barrier: transfer, acceleration-structure build,
compute, and render. That summary narrows the barrier's pipeline stages and
access masks to the producer kinds and the consumer pass being opened.

This is aggregate access tracking, not resource tracking. It stores no
resource identity, range, layout, lifetime, or dependency edge. It does not
change the public API or the placement of automatic barriers.

With `CommandEncoderDesc::manual_barriers`, an application can suppress the
automatic pass-boundary barriers and call `CommandEncoder::barrier()` only at
dependency cuts. A manual barrier is emitted immediately. Its source is
derived from all producer kinds recorded since the previous barrier, while its
destination remains conservative because the future consumer is not known at
that call.

## Guidance for engine developers

Keep automatic barriers as the conservative default for immediate-mode
rendering or code without a resource graph. If an engine already has a render
or task graph, group independent work into phases and place a global barrier
where one phase joins another. Profile this choice on the target hardware:
the study found large wins from removing redundant barriers in several
independent workloads, but also a regression on one integrated AMD GPU.

A global barrier represents a dependency cut, not a resource edge. It can
unnecessarily constrain an unrelated long-running producer that happens to be
on the source side of the cut. Use a resource-aware graph or a tracked API when
the workload needs arbitrary partially dependent DAG edges.

Blade also leaves transient aliasing, subresource layout transitions,
queue-family ownership, cross-queue synchronization, and exceptional native
layout requirements to a resource-aware layer above the RHI. The internal
pass-kind summary is not a replacement for those responsibilities.
