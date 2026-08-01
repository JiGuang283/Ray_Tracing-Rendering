# CUDA Packed Backend

This module uploads an immutable `CompiledScene` and runs the shared packed
geometry and material evaluation code over batches of rays. It currently
provides intersection, surface reconstruction, fixed-stack texture evaluation,
normal mapping, and material closure generation. It does not yet implement
camera sampling, BSDF sampling, light transport, or film accumulation.

## Ownership

- `DeviceBuffer<T>` is move-only RAII ownership for one device allocation.
- `DeviceSceneStorage` owns one device buffer for every `CompiledScene` array.
- `DeviceSceneView` is a trivially-copyable borrowed view passed to kernels.
- Scene upload validates all packed ranges first and commits only after every
  allocation and copy succeeds.
- Empty host arrays become null device pointers with a zero count.

The owning storage must outlive every kernel using its view. Catalog tools
upload and release one scene at a time so large image blobs do not accumulate
in device memory.

## Intersection

CPU and CUDA traversal include `flat_intersector_core.h`. The shared code uses
explicit float operations and `fmaf` ordering so host and device results stay
within the packed reference tolerance. BVH traversal has a fixed 64-entry
stack; overflow is reported as `PackedTraversalStatus::StackOverflow`, never
as a miss.

An optional uint32 RNG state per ray enables constant-medium sampling. A null
state pointer skips media, matching the CPU flat-intersector contract.

## Shading

CPU and CUDA shading include the same three host/device headers:

- `surface_reconstruction_core.h` rebuilds position, UV, normals, tangent
  derivatives, material IDs, and vertex colors from a compact `PackedHit`.
- `packed_texture_core.h` evaluates every packed texture node with a fixed
  16-entry traversal stack and no allocation or recursion.
- `packed_material_core.h` applies normal maps and emits a fixed-capacity
  `PackedMaterialOutput` containing emission and up to eight typed closures.

`reconstruct_hits_cuda()` and `evaluate_materials_cuda()` remain separate
launches. This keeps status reporting and CPU/GPU validation precise. A future
path tracing kernel may fuse them after the transport implementation is
verified.

The packed closure output is a data boundary, not a completed GPU BSDF. The
next transport stage still needs matching closure sample/eval/pdf routines,
light sampling, path state, Russian roulette, and Film accumulation.

## Build And Check

```bash
cmake -S . -B build-cuda -G Ninja \
  -DRAYTRACER_ENABLE_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-cuda -j2
ctest --test-dir build-cuda --output-on-failure
./build-cuda/cuda_scene_check
./build-cuda/cuda_scene_check --all
./build-cuda/cuda_shading_check
./build-cuda/cuda_shading_check --all
```

`cuda_shading_check` compares CPU and GPU traversal status, reconstructed
surface attributes, material status, texture stack depth, emission, shading
frames, and every closure field. It also evaluates one synthetic interaction
for every material so validation is not limited to camera-visible surfaces.

The CUDA option defaults to `OFF`, so normal CPU builds do not require a CUDA
compiler or runtime.
