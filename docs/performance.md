# Performance

Blade doesn't expect to be faster than wgpu-hal, but it's important to understand how much the difference is. Testing is done on "bunnymark" example, which is ported from wgpu. Since every draw call is dynamic in Blade, this benchmark is the worst case of the usage.

## MacBook Pro 2016

GPU: Intel Iris Graphics 550

Metal:
  - Blade starts to slow down after about 10K bunnies
  - wgpu-hal starts at 60K bunnies
  - wgpu starts at 15K bunnies

Vulkan Portability:
  - Blade starts to slow down at around 16K bunnies

## Thinkpad T495s

GPU: Ryzen 3500U

Windows/Vulkan:
  - Blade starts at around 20K bunnies
  - wgpu-hal starts at 60K bunnies
  - wgpu starts at 20K bunnies

## Conclusions

Amazingly, Blade performance on the worst case usage is on par with wgpu. This is the best outcome we could hope for. We are curious to see how the best case usage will compare.

Interestingly, on macOS running the test via MoltenVK shows slightly better performance. We suspect this is due to the fact MVK puts data into buffers and is able to bind the same buffer for vertex and fragment stages. We believe this can be optimized in Blade.

Ergonomically, our example is 300 LOC versus 830 LOC of wgpu-hal and 370-750 LOC in wgpu (depending on how we count the example framework).

It's also closer to the hardware (than even wgpu-hal) and easier to debug.