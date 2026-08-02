# Shared Render Data

`render_data` is the indexed host/device boundary for the renderer:

```text
JSON -> SceneIR (double source data)
     -> CPU runtime (double polymorphic reference)
     -> CompiledScene (float32 indexed owning arrays)
        -> CompiledSceneView (borrowed host pointers)
        -> DeviceSceneStorage -> DeviceSceneView (device pointers)
```

`raytracer_render_data` depends on Scene IR and the host resource compiler, not
on the CPU scene runtime or renderer. The polymorphic CPU renderer remains the
double-precision reference backend. A compiled scene is immutable after
`compile_scene()` returns.

## Data Contract

- Packed records contain no ownership, virtual functions, `bool`, `size_t`,
  smart pointers, variants, strings, or nested containers.
- All IDs and ranges use 32-bit unsigned values. `UINT32_MAX` is the invalid
  ID. Compilation fails before a buffer or range exceeds this address space.
- Source geometry and scene validation use double precision. Packed values use
  float32. BVH bounds are rounded toward negative/positive infinity so a float
  conversion cannot shrink a source bound.
- `CompiledScene` owns each array. `CompiledSceneView` borrows those arrays and
  must not outlive its owner.
- `validate_compiled_scene()` checks ranges, internal indices, BVH topology,
  material and texture references, and light distributions before upload.

## Geometry And BVH

Meshes share vertex and triangle ranges. Instances reference a mesh, sphere,
moving sphere, or medium plus an affine object/world transform and a material
slot table. Imported mesh assets are deduplicated by resource identity.

Every mesh BLAS and aggregate TLAS is a deterministic 12-bin SAH BVH2. A leaf
contains at most four items and the maximum depth is 64. Nodes use preorder:

- left child: current node + 1
- right child: `PackedBVHNode::first`
- leaf payload: contiguous range beginning at `first`

Triangles and aggregate instance references are physically reordered to leaf
order. Traversal therefore has no separate primitive-order indirection.

`PackedHit` stores only `t`, triangle barycentrics, instance ID, primitive ID,
and hit flags. `reconstruct_compiled_hit()` derives UVs, geometric and shading
normals, derivatives, vertex color, source primitive ID, and material ID.

## Materials And Textures

Material factories expose a typed host `MaterialDescription`; the compiler
does not inspect parameter positions or use `dynamic_cast`. Texture nodes are
deduplicated by object identity and emitted in topological order.

`evaluate_packed_texture()` and `evaluate_packed_material_emission()` are
host-only reference evaluators. They validate the packed texture behavior
before an equivalent device evaluator exists. Images are uncompressed float
texel arrays in this phase; no mipmaps or cross-scene cache are generated.

## Lights

Delta lights are listed in `delta_light_indices` and are evaluated
deterministically. `non_delta_light_indices` corresponds one-to-one with the
global selection probability and CDF arrays. Selection uses 95% power
importance plus a 5% uniform floor.

`emitter_bindings` is parallel to `material_bindings`. Each instance material
slot either contains the corresponding BSDF-hittable light ID or the invalid
index. Reconstructed interactions carry this ID so an emissive hit can
evaluate its complementary light-sampling PDF without searching every light.
Each non-delta `PackedLight` also stores the same selection probability as its
entry in the global probability table.

For triangle and mesh emitters, `element_indices` addresses global packed
triangles. `distribution` stores `N` probabilities followed by `N` CDF values.
Probabilities use the current four-point emission estimate and the 95%
emission-importance plus 5% area mixture.

For an environment light with width `W` and height `H`, `distribution` stores:

```text
W*H conditional probabilities
H*(W+1) conditional CDF values
H marginal probabilities
H+1 marginal CDF values
```

Environment texels remain in the shared image buffers. Flags record LDR sRGB
decoding and square light-probe mapping.

`packed_light_core.h` is the shared host/device implementation for delta,
quad, affine sphere, triangle/mesh, and environment sampling. Area densities
are converted to solid-angle densities at the shading point. Square angular
light probes use their own UV-to-solid-angle Jacobian rather than the
equirectangular Jacobian.

## Transport Reference

`packed_transport_core.h` joins flat traversal, reconstruction, packed
materials, BSDFs, lights, visibility, MIS, eta-aware Russian roulette, and
camera ray generation. The host wrapper is a migration reference and does not
replace the polymorphic CPU renderer. Integrator IDs 0 through 4 are lowered to
the same `IntegratorPolicy` consumed by CPU and CUDA transport; the two
backends intentionally retain separate double and float numerical kernels.

Camera samples use an explicit per-path RNG. `PackedTransportResult` reports a
typed failure instead of turning traversal, shading, or non-finite errors into
black pixels. The CUDA wavefront renderer preserves these state transitions
across kernels.

## Media And Verification

Each constant medium references a private boundary aggregate, density, and
phase material. Flat intersection samples a medium only when the caller passes
an explicit worker-local `RNG*`; a null RNG skips stochastic medium events.

Useful checks:

```bash
./build/scene_data_check --catalog assets/scenes/catalog.json
./build/scene_intersection_check
./build/packed_transport_check
./build-cuda/cuda_scene_check
```

The first command compiles and validates all catalog scenes and reports buffer
sizes. The second compares fixed camera-ray corpora against the polymorphic
double CPU reference for representative analytic, mesh, medium, textured, and
complex glTF scenes.

The CUDA checks upload every packed array and compare intersection, shading,
lighting, and transport against the Packed CPU reference. CUDA remains
optional, but both CPU and CUDA are production `IRenderSession` backends.
