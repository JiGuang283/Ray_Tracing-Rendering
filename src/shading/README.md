# Shading Architecture

The renderer separates surface evaluation into four layers:

1. Geometry writes a lightweight `hit_record`.
2. The integrator converts it into `SurfaceInteraction`.
3. A `material` acts as a surface shader and fills `ShadingResult`.
4. The integrator samples/evaluates `ShadingResult::bsdf` and combines it with
   light samples.

`material` classes should own textures and user-facing parameters. They should
not trace shadow rays, select lights, or implement MIS policy.

`BSDF` owns closure sampling and evaluation. New visual effects should usually
enter as a new closure or as a new shader that combines existing closures.
Per-closure sampling, evaluation, and PDF code lives in the shading closure
helpers; `BSDF` itself stays a small fixed-capacity closure container.

`LightSampler` owns light selection policy. Integrators should ask it for light
samples and PDFs instead of manually selecting from the light array.

Texture graphs are the first shader graph layer. Materials consume texture
parameters; nested texture nodes such as mix, multiply, scale, and color ramp
belong in the texture system instead of in integrators or scene geometry.

Area emitters are scene-level lights derived from emissive geometry. Materials
only expose an emission estimate; scene loading decides whether an emissive
surface should be registered with the light sampler.
