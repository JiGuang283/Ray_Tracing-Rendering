#ifndef PACKED_TRANSPORT_H
#define PACKED_TRANSPORT_H

#include "compiled_scene.h"
#include "rng.h"

PackedRay generate_packed_camera_ray(const PackedCamera &camera,
                                     std::uint32_t pixel_x,
                                     std::uint32_t pixel_y,
                                     std::uint32_t width,
                                     std::uint32_t height, RNG &rng);

PackedTransportResult trace_packed_path(
    const CompiledSceneView &scene, const PackedRay &ray,
    const PackedTransportSettings &settings, RNG &rng);

#endif
