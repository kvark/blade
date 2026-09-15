# Combined-initialization candidate only

Current upstream is 6ab5fcec. Carry the exact flushed allocator/buffer/bind
observability from 4d8c8bc, matching the completed c96a9a87 control. Retain all
upstream shader-validation fixes. Do not change allocation policy or add GPU
queries. The captured placement is not an allocation lifetime or fault-origin
proof. This branch is not qualified or adopted.

CPU preparation is authorized; GPU work requires a separate explicit one-job
declaration, fresh host/upstream checks and direct-child guard. No follower or
host recovery. Never retry the quarantined 0a98775/native 02b600a1 bundle.
Keep the held Pong queue and all correctness/memory/throughput gates intact.
