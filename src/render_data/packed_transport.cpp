#include "render_data/packed_transport.h"

#include "render_data/packed_transport_core.h"

PackedRay generate_packed_camera_ray(const PackedCamera &camera,
                                     std::uint32_t pixel_x,
                                     std::uint32_t pixel_y,
                                     std::uint32_t width,
                                     std::uint32_t height, RNG &rng) {
    return packed_transport::generate_packed_camera_ray_core(
        camera, pixel_x, pixel_y, width, height, rng);
}

PackedTransportResult trace_packed_path(
    const CompiledSceneView &scene, const PackedRay &ray,
    const PackedTransportSettings &settings, RNG &rng) {
    return packed_transport::trace_packed_path_core(scene, ray, settings,
                                                    rng);
}
