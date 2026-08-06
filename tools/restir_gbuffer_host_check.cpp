#include "restir_gbuffer_host_check.h"

#include "restir_gbuffer_core.h"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace {

bool near(float left, float right, float tolerance = 5e-5f) {
    const float scale =
        std::max(1.0f, std::max(std::abs(left), std::abs(right)));
    return std::abs(left - right) <= tolerance * scale;
}

bool compare_surface(const restir::RestirSurface &device,
                     const restir::RestirSurface &host) {
    if (device.valid() != host.valid() || device.flags != host.flags) {
        return false;
    }
    if (!host.valid()) {
        return true;
    }
    if (device.instance_id != host.instance_id ||
        device.primitive_id != host.primitive_id ||
        device.material_id != host.material_id ||
        device.emitter_id != host.emitter_id ||
        !near(device.position.x, host.position.x) ||
        !near(device.position.y, host.position.y) ||
        !near(device.position.z, host.position.z) ||
        !near(device.view_depth, host.view_depth) ||
        !near(device.hit_t, host.hit_t) ||
        !near(device.barycentric_u, host.barycentric_u) ||
        !near(device.barycentric_v, host.barycentric_v) ||
        !near(device.ray_time, host.ray_time)) {
        return false;
    }
    const Float3 device_geometry =
        restir::unpack_octahedral_normal(device.geometric_normal);
    const Float3 host_geometry =
        restir::unpack_octahedral_normal(host.geometric_normal);
    const Float3 device_shading =
        restir::unpack_octahedral_normal(device.shading_normal);
    const Float3 host_shading =
        restir::unpack_octahedral_normal(host.shading_normal);
    const float geometry_dot = device_geometry.x * host_geometry.x +
                               device_geometry.y * host_geometry.y +
                               device_geometry.z * host_geometry.z;
    const float shading_dot = device_shading.x * host_shading.x +
                              device_shading.y * host_shading.y +
                              device_shading.z * host_shading.z;
    return geometry_dot >= 0.9999f && shading_dot >= 0.9999f;
}

} // namespace

std::uint64_t compare_restir_gbuffer_host(
    const CompiledSceneView &scene, std::uint32_t width,
    std::uint32_t height, std::uint32_t iteration,
    std::uint32_t seed,
    const std::vector<restir::RestirSurface> &device_surfaces) {
    const std::uint32_t pixel_count = width * height;
    if (device_surfaces.size() != pixel_count) {
        return pixel_count;
    }
    std::uint64_t errors = 0;
    for (std::uint32_t pixel = 0; pixel < pixel_count; ++pixel) {
        restir::RestirSurface host_surface;
        (void)restir::build_primary_surface_core(
            scene, width, height, pixel, iteration, seed, host_surface);
        if (!compare_surface(device_surfaces[pixel], host_surface)) {
            if (errors == 0u) {
                const restir::RestirSurface &device =
                    device_surfaces[pixel];
                std::cerr << "RESTIR_GBUFFER_MISMATCH pixel=" << pixel
                          << " device_valid=" << device.valid()
                          << " host_valid=" << host_surface.valid()
                          << " device_flags=" << device.flags
                          << " host_flags=" << host_surface.flags
                          << " device_instance=" << device.instance_id
                          << " host_instance=" << host_surface.instance_id
                          << " device_primitive=" << device.primitive_id
                          << " host_primitive=" << host_surface.primitive_id
                          << " device_t=" << device.hit_t
                          << " host_t=" << host_surface.hit_t
                          << " device_position=" << device.position.x << ','
                          << device.position.y << ',' << device.position.z
                          << " host_position=" << host_surface.position.x << ','
                          << host_surface.position.y << ','
                          << host_surface.position.z
                          << " device_bary=" << device.barycentric_u << ','
                          << device.barycentric_v
                          << " host_bary=" << host_surface.barycentric_u << ','
                          << host_surface.barycentric_v
                          << " device_normals=" << device.geometric_normal
                          << ',' << device.shading_normal
                          << " host_normals=" << host_surface.geometric_normal
                          << ',' << host_surface.shading_normal
                          << " device_ray_time=" << device.ray_time
                          << " host_ray_time=" << host_surface.ray_time
                          << '\n';
            }
            ++errors;
        }
    }
    return errors;
}
