# CUDA Packed Backend

This module uploads an immutable `CompiledScene` and runs the shared packed
geometry and material evaluation code over batches of rays. It provides
intersection, surface reconstruction, fixed-stack texture evaluation, normal
mapping, material closure generation, BSDF sample/eval/pdf, light sampling,
complete camera light transport, and GPU Film accumulation.

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

CPU and CUDA shading include the same host/device headers:

- `surface_reconstruction_core.h` rebuilds position, UV, normals, tangent
  derivatives, material IDs, and vertex colors from a compact `PackedHit`.
- `packed_texture_core.h` evaluates every packed texture node with a fixed
  16-entry traversal stack and no allocation or recursion.
- `packed_material_core.h` applies normal maps and emits a fixed-capacity
  `PackedMaterialOutput` containing emission and up to eight typed closures.
- `packed_bsdf_core.h` samples and evaluates those closures, including delta
  event probabilities, dielectric eta tracking, and GGX VNDF sampling.
- `packed_light_core.h` evaluates deterministic delta lights and samples the
  global non-delta distribution, textured geometry emitters, and environment
  maps with matching MIS PDFs.

`reconstruct_hits_cuda()` and `evaluate_materials_cuda()` remain separate
validation launches. This keeps status reporting and CPU/GPU comparison
precise while the production path queue evaluates reconstruction, material,
lighting, visibility, and BSDF work together for one bounce.

## Wavefront Transport

`render_wavefront_cuda()` assigns one `PackedPathState` to every active camera
sample in a batch. Two compact index queues alternate between bounces. Each
path owns its RNG state, so atomic queue ordering cannot change its random
sequence. Finished paths are filtered and accumulated into a linear float
Film on the device; only the completed Film and counters are downloaded.

The host stops launching bounce work when the compact queue becomes empty.
Batching caps temporary memory independently of image resolution, while each
sample layer is accumulated in a fixed order for repeatable single-device
results. Transport errors remain typed and are counted rather than silently
turning into successful black paths.

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
./build-cuda/cuda_light_check
./build-cuda/cuda_light_check --all
./build-cuda/cuda_shading_check
./build-cuda/cuda_shading_check --all
./build-cuda/cuda_transport_check
./build-cuda/cuda_transport_check --all --width 4 --height 3 --spp 1
```

`cuda_shading_check` compares CPU and GPU traversal status, reconstructed
surface attributes, material status, texture stack depth, emission, shading
frames, and every closure field. It also evaluates one synthetic interaction
for every material so validation is not limited to camera-visible surfaces.

`cuda_light_check` compares every packed light type, conditional PDFs, global
non-delta selection, and final RNG states on the CPU and GPU.

`cuda_transport_check` compares the CPU packed reference and the compact CUDA
path queue at the Film boundary, checks status counts, and renders each case
twice to verify deterministic GPU accumulation.

The CUDA option defaults to `OFF`, so normal CPU builds do not require a CUDA
compiler or runtime.
