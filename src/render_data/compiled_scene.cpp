#include "compiled_scene.h"

#include <cmath>
#include <limits>
#include <sstream>
#include <type_traits>

namespace {

template <typename T>
PackedArrayView<T> view_of(const std::vector<T> &values) {
    return {values.data(), static_cast<std::uint32_t>(values.size())};
}

template <typename T>
std::uint64_t bytes_of(const std::vector<T> &values) {
    return static_cast<std::uint64_t>(values.size()) * sizeof(T);
}

bool finite(const Float3 &value) {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z);
}

bool valid_range(Range32 range, std::size_t size) {
    return range.offset <= size && range.count <= size - range.offset;
}

void add_range_error(ValidationReport &report, const char *owner,
                     const char *field, std::size_t index) {
    std::ostringstream message;
    message << owner << '[' << index << "]." << field
            << " references data outside its buffer";
    report.errors.push_back(message.str());
}

} // namespace

CompiledSceneView make_scene_view(const CompiledScene &scene) {
    CompiledSceneView view;
    view.camera = scene.camera;
    view.background = scene.background;
    view.scene_time0 = scene.scene_time0;
    view.scene_time1 = scene.scene_time1;
    view.positions = view_of(scene.positions);
    view.normals = view_of(scene.normals);
    view.tangents = view_of(scene.tangents);
    view.uv0 = view_of(scene.uv0);
    view.vertex_colors = view_of(scene.vertex_colors);
    view.triangles = view_of(scene.triangles);
    view.meshes = view_of(scene.meshes);
    view.spheres = view_of(scene.spheres);
    view.moving_spheres = view_of(scene.moving_spheres);
    view.transforms = view_of(scene.transforms);
    view.instances = view_of(scene.instances);
    view.material_bindings = view_of(scene.material_bindings);
    view.aggregates = view_of(scene.aggregates);
    view.aggregate_instance_indices =
        view_of(scene.aggregate_instance_indices);
    view.bvh_nodes = view_of(scene.bvh_nodes);
    view.media = view_of(scene.media);
    view.materials = view_of(scene.materials);
    view.texture_nodes = view_of(scene.texture_nodes);
    view.images = view_of(scene.images);
    view.image_texels = view_of(scene.image_texels);
    view.perlin_tables = view_of(scene.perlin_tables);
    view.perlin_gradients = view_of(scene.perlin_gradients);
    view.perlin_permutations = view_of(scene.perlin_permutations);
    view.lights = view_of(scene.lights);
    view.light_selection_probabilities =
        view_of(scene.light_selection_probabilities);
    view.light_cdf = view_of(scene.light_cdf);
    view.light_distributions = view_of(scene.light_distributions);
    return view;
}

CompiledSceneStats compiled_scene_stats(const CompiledScene &scene) {
    CompiledSceneStats stats;
    stats.vertices = static_cast<std::uint32_t>(scene.positions.size());
    stats.triangles = static_cast<std::uint32_t>(scene.triangles.size());
    stats.bvh_nodes = static_cast<std::uint32_t>(scene.bvh_nodes.size());
    stats.meshes = static_cast<std::uint32_t>(scene.meshes.size());
    stats.instances = static_cast<std::uint32_t>(scene.instances.size());
    stats.materials = static_cast<std::uint32_t>(scene.materials.size());
    stats.textures = static_cast<std::uint32_t>(scene.texture_nodes.size());
    stats.images = static_cast<std::uint32_t>(scene.images.size());
    stats.lights = static_cast<std::uint32_t>(scene.lights.size());

#define ADD_BUFFER_BYTES(field) stats.bytes += bytes_of(scene.field)
    ADD_BUFFER_BYTES(positions);
    ADD_BUFFER_BYTES(normals);
    ADD_BUFFER_BYTES(tangents);
    ADD_BUFFER_BYTES(uv0);
    ADD_BUFFER_BYTES(vertex_colors);
    ADD_BUFFER_BYTES(triangles);
    ADD_BUFFER_BYTES(meshes);
    ADD_BUFFER_BYTES(spheres);
    ADD_BUFFER_BYTES(moving_spheres);
    ADD_BUFFER_BYTES(transforms);
    ADD_BUFFER_BYTES(instances);
    ADD_BUFFER_BYTES(material_bindings);
    ADD_BUFFER_BYTES(aggregates);
    ADD_BUFFER_BYTES(aggregate_instance_indices);
    ADD_BUFFER_BYTES(bvh_nodes);
    ADD_BUFFER_BYTES(media);
    ADD_BUFFER_BYTES(materials);
    ADD_BUFFER_BYTES(texture_nodes);
    ADD_BUFFER_BYTES(images);
    ADD_BUFFER_BYTES(image_texels);
    ADD_BUFFER_BYTES(perlin_tables);
    ADD_BUFFER_BYTES(perlin_gradients);
    ADD_BUFFER_BYTES(perlin_permutations);
    ADD_BUFFER_BYTES(lights);
    ADD_BUFFER_BYTES(light_selection_probabilities);
    ADD_BUFFER_BYTES(light_cdf);
    ADD_BUFFER_BYTES(light_distributions);
#undef ADD_BUFFER_BYTES
    return stats;
}

ValidationReport validate_compiled_scene(const CompiledScene &scene) {
    ValidationReport report;
    if (scene.aggregates.empty()) {
        report.errors.push_back("compiled scene has no world aggregate");
    }
    if (!finite(scene.background)) {
        report.errors.push_back("compiled scene background is non-finite");
    }

    const std::size_t vertex_count = scene.positions.size();
    if (scene.normals.size() != vertex_count ||
        scene.tangents.size() != vertex_count ||
        scene.uv0.size() != vertex_count ||
        scene.vertex_colors.size() != vertex_count) {
        report.errors.push_back(
            "compiled vertex attribute buffers have different sizes");
    }

    for (std::size_t index = 0; index < scene.meshes.size(); ++index) {
        const PackedMesh &mesh = scene.meshes[index];
        if (!valid_range(mesh.vertices, scene.positions.size())) {
            add_range_error(report, "mesh", "vertices", index);
        }
        if (!valid_range(mesh.triangles, scene.triangles.size())) {
            add_range_error(report, "mesh", "triangles", index);
        }
        if (!valid_range(mesh.bvh_nodes, scene.bvh_nodes.size())) {
            add_range_error(report, "mesh", "bvh_nodes", index);
        }
        if (!finite(mesh.bounds_min) || !finite(mesh.bounds_max)) {
            report.errors.push_back("mesh has non-finite bounds");
        }
    }

    for (std::size_t index = 0; index < scene.instances.size(); ++index) {
        const PackedInstance &instance = scene.instances[index];
        if (instance.transform_id >= scene.transforms.size()) {
            add_range_error(report, "instance", "transform_id", index);
        }
        if (!valid_range(instance.material_bindings,
                         scene.material_bindings.size())) {
            add_range_error(report, "instance", "material_bindings", index);
        }
        if (!finite(instance.bounds_min) || !finite(instance.bounds_max)) {
            report.errors.push_back("instance has non-finite bounds");
        }
    }

    for (std::size_t index = 0; index < scene.aggregates.size(); ++index) {
        const PackedAggregate &aggregate = scene.aggregates[index];
        if (!valid_range(aggregate.bvh_nodes, scene.bvh_nodes.size())) {
            add_range_error(report, "aggregate", "bvh_nodes", index);
        }
        if (!valid_range(aggregate.instance_indices,
                         scene.aggregate_instance_indices.size())) {
            add_range_error(report, "aggregate", "instance_indices", index);
        }
    }

    for (std::size_t index = 0; index < scene.media.size(); ++index) {
        const PackedMedium &medium = scene.media[index];
        if (medium.boundary_aggregate >= scene.aggregates.size()) {
            add_range_error(report, "medium", "boundary_aggregate", index);
        }
        if (medium.phase_material >= scene.materials.size()) {
            add_range_error(report, "medium", "phase_material", index);
        }
        if (!std::isfinite(medium.neg_inv_density) ||
            medium.neg_inv_density >= 0.0f) {
            report.errors.push_back("medium has invalid density");
        }
    }

    for (std::size_t index = 0; index < scene.images.size(); ++index) {
        if (!valid_range(scene.images[index].texels,
                         scene.image_texels.size())) {
            add_range_error(report, "image", "texels", index);
        }
    }

    for (std::size_t index = 0; index < scene.perlin_tables.size(); ++index) {
        const PackedPerlinDesc &table = scene.perlin_tables[index];
        if (!valid_range(table.gradients, scene.perlin_gradients.size()) ||
            !valid_range(table.permutation_x,
                         scene.perlin_permutations.size()) ||
            !valid_range(table.permutation_y,
                         scene.perlin_permutations.size()) ||
            !valid_range(table.permutation_z,
                         scene.perlin_permutations.size())) {
            add_range_error(report, "perlin", "data", index);
        }
    }

    if (scene.light_selection_probabilities.size() !=
            scene.light_cdf.size() ||
        scene.light_selection_probabilities.size() > scene.lights.size()) {
        report.errors.push_back("light selection buffers are inconsistent");
    }
    return report;
}
