# Kindle timing candidate

Current upstream is 92553493. Carry the exact flushed allocator/buffer/bind
observability from 4d8c8bc, matching the completed c96a9a87 control. Retain all
upstream shader-validation fixes. Do not change allocation policy or add GPU
queries. The captured placement is not an allocation lifetime or fault-origin
proof. This branch is not qualified or adopted.

The September 16 user resumes bounded GPU work with NVML disabled. Add only the
encoder timing-enabled accessor needed to use upstream last_timing safely on
contexts with timing disabled. Preserve the actual upstream timestamp fixes.
Use fresh declarations, direct-child host guard and actual-device assertions;
review each GPU result before follow-up. No NVML, blind retry or host recovery.
Never retry the quarantined 0a98775/native 02b600a1 bundle.
Keep the held Pong queue and all correctness/memory/throughput gates intact.
