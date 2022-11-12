Principles: simple, *opinionated*, convenient, fast

Requirements:
	- native-only, mid to high end platforms
	- oriented towards compute, or low-call rendering
	- a single uniform block for each descriptor set
	- all the data is pushed to the shader, no pre-baked descriptors
	- resources are `Copy`
	- mapping is direct
  
Possibly Vulkan-only? Require dynamic rendering, descriptor templates, inline uniform buffers.

Why not wgpu?
  - async ops are annoying
  - mapping API is too restricted

Why not wgpu-hal?
  - may be difficult to upstream changes?
  - incompatible with a simplified synchronization model (only GENERAL layout)

Why not Sierra/Arcana?
  - tends to have overengineered solutions
