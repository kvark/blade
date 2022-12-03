# Performance

Blade doesn't expect to be faster than wgpu-hal, but it's important to understand how much the difference is. Testing is done on "bunnymark" example, which is ported from wgpu. Since every draw call is dynamic in Blade, this benchmark is the worst case of the usage.

## MacBook Pro 2016

GPU: Intel Iris Graphics 550

Metal:
  - Blade starts to slow down after about 10K bunnies
  - wgpu-hal starts at 60K bunnies
  - wgpu starts at 15K bunnies

Vulkan Portability:
  - Blade starts to slow down at around 500 bunnies

## Thinkpad T495s

GPU: Ryzen 3500U

Windows/Vulkan:
  - Blade starts at around 500 bunnies
  - wgpu-hal starts at 60K bunnies
  - wgpu starts at 20K bunnies
