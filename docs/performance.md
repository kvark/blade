# Performance

Blade doesn't expect to be faster than wgpu-hal, but it's important to understand how much the difference is. Testing is done on "bunnymark" example, which is ported from wgpu.

## MBP 2016 with "Intel Iris Graphics 550"

Metal:
  - Blade starts to slow down after about 10K bunnies
  - wgpu-hal starts at 10K bunnies
  - wgpu starts at 5K bunnies

Vulkan Portability:
  - Blade starts to slow down at around 500 bunnies

## Conclusions

So this is quite incredible. We aren't preserving the buffer with bunny data across frames, everything is dynamic, and yet the performance is within a factor of 2 for this worst case. Ergonomically, our example is 370 LOC versus 830 LOC of wgpu-hal.
