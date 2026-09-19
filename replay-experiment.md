# Serialized Vulkan replay experiment

This branch is a timing prototype, not a public replay API. It makes Vulkan
`finish` idempotent and records without `ONE_TIME_SUBMIT`. The matching
Meganeura `experiment/llama-replay-2026-09-19` branch resubmits only an idle,
untimed inference encoder when `MEGANEURA_REPLAY=1`.

The GGUF diagnostic updates input contents but keeps its allocations, pipelines,
and plan fixed after setup. It passed the 33-row independent CPU logits check on
RTX 5070 and Arc B570. General use needs explicit opt-in recording, unsupported
backend fallback, and invalidation for rebinding, tuning, profiling, and other
uses of the encoder. Do not enable this prototype in applications.
