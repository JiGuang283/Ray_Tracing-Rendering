# CUDA Geometry Backend

This module is the first CUDA backend boundary. It uploads an immutable
`CompiledScene` and runs the shared packed BVH intersector over batches of
rays. It does not yet implement camera sampling, shading, light transport, or
film accumulation.

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
```

The CUDA option defaults to `OFF`, so normal CPU builds do not require a CUDA
compiler or runtime.
