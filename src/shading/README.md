# Shading Architecture

The renderer keeps scene description, shader evaluation, scattering, and light
transport as separate layers:

1. JSON texture and material values are parsed into typed scene IR.
2. Scene building compiles that IR into immutable `Texture` and
   `MaterialInstance` resources.
3. Geometry writes a lightweight `hit_record`; the integrator converts it into
   `SurfaceInteraction` and `ShaderEvalContext`.
4. `MaterialProgram::evaluate()` reads an indexed parameter block and writes a
   stack-allocated `MaterialOutput`.
5. Integrators consume only `MaterialOutput::emission` and the BSDF
   `sample/eval/pdf` interface.

`MaterialProgram` objects are stateless and shared. Per-material values live in
`MaterialInstance`, while temporary shader values use worker-local
`ShaderScratch`. A shader must not trace rays, select lights, or implement MIS.

`BSDF` is a fixed-capacity container of typed closure variants. Physical
contribution and lobe selection weight are separate. Continuous closures use a
mixture PDF; delta closures retain both lobe-selection and Fresnel event
probabilities. Integrators use the same `f * factor / pdf` throughput rule for
both measures and treat delta events specially only for MIS.

Image files are immutable `ImageAsset` resources cached by normalized path.
`ImageTexture` is a view that supplies color space, channel, wrapping, and
filtering. sRGB decoding happens before interpolation. Material input semantics
resolve legacy unspecified images as sRGB for color and linear for scalar or
normal data.

glTF materials compile into the same texture views and built-in material
programs as JSON materials. Base color is multiplied by `COLOR_0`; packed
metallic-roughness channels and glTF sampler/UV transforms are represented as
ordinary immutable texture nodes. The integrator therefore has no model-format
specific shading path.

Normal maps perturb only the shading frame. Geometric normals remain the source
of front-face classification and spawned-ray offsets. Tangent frames retain UV
handedness, and normal maps may declare OpenGL or DirectX convention.

Nested texture nodes (`mix`, `multiply`, `scale`, and `color_ramp`) are the
current graph-like layer. A general shader graph compiler or VM is intentionally
outside this foundation.
