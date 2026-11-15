# Differentiable tracer against Mitsuba 3

Measured gradients of `blade_render::DiffTracer` and of Mitsuba 3.9.1, on
the same scene, against central differences of the same loss in each
renderer. The scene is that of the tests of lights in `tests/diff`: a wall
facing the camera, and a sphere above the view that shades the wall from a
light above it. The camera only sees the wall, so the gradient by the
sphere is entirely that of the shadow it casts. `mitsuba_scene.py` builds
the same meshes, camera, light, loss and checks in Mitsuba.

- Loss: the mean squared error to a target image, with the light and the
  sphere moved. The central differences take the unbiased form
  `sum((A - T) * (B - T)) / 3N`, of two images `A` and `B` of independent
  random numbers.
- Gradient: the mean over iterations of the gradient of one noisy loss,
  with the primal and the adjoint of independent random numbers. The error
  is the standard error of the mean, from the spread of 8 batches.
- Both ran on the same 4-core CPU without a GPU: blade on lavapipe (Mesa's
  software Vulkan, with ray queries), Mitsuba on its LLVM backend
  (`llvm_ad_rgb`). Times are for the gradient iterations alone, after the
  shaders or kernels were compiled.
- The materials differ a little: blade's metallic-roughness BSDF and
  Mitsuba's `principled` BSDF of the same base colors and roughness. The
  central differences of the two renderers differ by up to about 15% for
  that.

## Point light

`diff_point_light_gradients_match_finite_differences`: a point light of
800 at (0.3, 7.7, 0.1), against a target with it at (0, 8, -0.2), and with
the sphere moved by (0.3, 0.25, 0). Blade took 256 iterations, Mitsuba 32
of `direct_projective` with 4 samples per pixel, and its central
differences 8192.

| d loss / d | blade | blade central difference | Mitsuba 3 | Mitsuba central difference |
|---|---|---|---|---|
| sphere x | 0.1436 | 0.1570 | 0 | 0.1821 |
| sphere y | -0.01526 | -0.01465 | 0 | -0.01643 |
| light x | -0.0648 | -0.0742 | 0 | -0.0879 |
| light y | -0.0474 | -0.0480 | 0 | -0.0568 |
| light z | 0.0786 | 0.0779 | 0 | 0.0911 |
| intensity | 2.988e-5 | 2.988e-5 | 3.533e-5 | 3.533e-5 |

Mitsuba 3.9.1 doesn't mark the position of a `point` emitter as
differentiable (`mi.traverse` lists `light.position` without the flag), and
its projective integrators (`direct_projective` here) don't find the
shadows of point lights: the gradients by the sphere and the light come out
as exactly 0, while its own central differences aren't. Blade samples those
shadows at the silhouettes that the light sees.

## Small emissive sphere

`diff_area_light_gradients_match_finite_differences`: the point light is
replaced by an emissive sphere of radius 0.1, of the same power, whose
translation is the parameter of the light. Blade draws the points on it
by area (`DiffConfig::emitters_by_area`), with 8 primal and 4 adjoint
samples per pixel, over 256 iterations. Mitsuba used 4 and 4, over 32
iterations of `direct_projective` and 16 of `prb_projective`, with their
default silhouette sampling.

| d loss / d | blade | central difference | Mitsuba `direct_projective` | Mitsuba `prb_projective` | central difference |
|---|---|---|---|---|---|
| sphere x | 0.1395 ± 0.0040 | 0.1476 | 0.1744 ± 0.0050 | 0.1813 ± 0.0103 | 0.1707 |
| sphere y | -0.0150 ± 0.0008 | -0.0171 | -0.0175 ± 0.0013 | -0.0178 ± 0.0023 | -0.0163 |
| light x | -0.0632 ± 0.0017 | -0.0685 | -0.021 ± 0.011 | -0.019 ± 0.020 | -0.0807 |
| light y | -0.0445 ± 0.0004 | -0.0443 | -0.093 ± 0.005 | -0.076 ± 0.005 | -0.0529 |
| light z | 0.0718 ± 0.0003 | 0.0727 | 0.107 ± 0.012 | 0.128 ± 0.010 | 0.0855 |
| time per iteration | 0.068 s | | 30.8 s | 20.5 s | |

Both renderers get the gradients by the sphere. Mitsuba's gradients by
the translation of the emitting mesh disagree with its own central
differences by 4 to 7 standard errors for two of the three axes, with both
integrators, in this scene. Blade's agree within 8%.

For the gradient by the sphere x, the standard deviation of one iteration
is 0.063 for blade and 0.028 for Mitsuba `direct_projective`, so for a
standard error of 0.01, blade takes 40 iterations, 2.7 s, and Mitsuba 8, 4
minutes.

Without `emitters_by_area`, blade finds the shadow among the directions
drawn from the wall, which rarely meet the small emitter: the same 256
iterations had a standard error of 0.049 for the sphere x, twelve times as
much.

## Not compared

- Shadows seen through a bounce
  (`diff_indirect_light_shadow_gradients_match_finite_differences`) and the
  other scenes of `tests/diff` only ran in blade.
- Mitsuba's CUDA variants, for want of a GPU, and blade on a GPU.
- Mitsuba 3 has what blade doesn't: differentiation of any program of its
  JIT, rather than adjoints written for each term, participating media,
  spectral rendering and polarization.
