#include "compiled_scene.h"

#include <algorithm>
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

bool finite(const Float4 &value) {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z) && std::isfinite(value.w);
}

bool valid_range(Range32 range, std::size_t size) {
    return range.offset <= size && range.count <= size - range.offset;
}

bool valid_optional_index(std::uint32_t index, std::size_t size) {
    return index == kInvalidPackedIndex || index < size;
}

void add_range_error(ValidationReport &report, const char *owner,
                     const char *field, std::size_t index) {
    std::ostringstream message;
    message << owner << '[' << index << "]." << field
            << " references data outside its buffer";
    report.errors.push_back(message.str());
}

void add_index_error(ValidationReport &report, const char *owner,
                     const char *field, std::size_t index) {
    std::ostringstream message;
    message << owner << '[' << index << "]." << field
            << " contains an invalid index";
    report.errors.push_back(message.str());
}

void validate_bvh_range(ValidationReport &report,
                        const CompiledScene &scene, Range32 nodes,
                        Range32 payloads, const char *owner,
                        std::size_t owner_index) {
    if (!valid_range(nodes, scene.bvh_nodes.size()) || nodes.count == 0) {
        return;
    }
    const std::uint64_t node_end =
        static_cast<std::uint64_t>(nodes.offset) + nodes.count;
    const std::uint64_t payload_end =
        static_cast<std::uint64_t>(payloads.offset) + payloads.count;
    for (std::uint32_t local = 0; local < nodes.count; ++local) {
        const std::uint32_t node_index = nodes.offset + local;
        const PackedBVHNode &node = scene.bvh_nodes[node_index];
        if (!finite(node.bounds_min) || !finite(node.bounds_max) ||
            node.bounds_min.x > node.bounds_max.x ||
            node.bounds_min.y > node.bounds_max.y ||
            node.bounds_min.z > node.bounds_max.z) {
            add_index_error(report, owner, "bvh_bounds", owner_index);
            return;
        }
        if (node.is_leaf()) {
            if (node.primitive_count() == 0 || node.first < payloads.offset ||
                static_cast<std::uint64_t>(node.first) +
                        node.primitive_count() >
                    payload_end) {
                add_index_error(report, owner, "bvh_leaf", owner_index);
                return;
            }
        } else if (node_index + 1 >= node_end || node.first <= node_index ||
                   node.first >= node_end) {
            add_index_error(report, owner, "bvh_children", owner_index);
            return;
        }
    }
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
    view.emitter_bindings = view_of(scene.emitter_bindings);
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
    view.delta_light_indices = view_of(scene.delta_light_indices);
    view.non_delta_light_indices = view_of(scene.non_delta_light_indices);
    view.light_selection_probabilities =
        view_of(scene.light_selection_probabilities);
    view.light_cdf = view_of(scene.light_cdf);
    view.light_element_indices = view_of(scene.light_element_indices);
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
    ADD_BUFFER_BYTES(emitter_bindings);
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
    ADD_BUFFER_BYTES(delta_light_indices);
    ADD_BUFFER_BYTES(non_delta_light_indices);
    ADD_BUFFER_BYTES(light_selection_probabilities);
    ADD_BUFFER_BYTES(light_cdf);
    ADD_BUFFER_BYTES(light_element_indices);
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
    if (scene.emitter_bindings.size() != scene.material_bindings.size()) {
        report.errors.push_back(
            "material and emitter binding buffers have different sizes");
    }
    for (std::size_t index = 0; index < scene.transforms.size(); ++index) {
        const PackedTransform &transform = scene.transforms[index];
        bool transform_finite = true;
        for (float value : transform.object_to_world) {
            transform_finite = transform_finite && std::isfinite(value);
        }
        for (float value : transform.world_to_object) {
            transform_finite = transform_finite && std::isfinite(value);
        }
        if (!transform_finite) {
            add_index_error(report, "transform", "matrix", index);
        }
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
        if (valid_range(mesh.triangles, scene.triangles.size())) {
            for (std::uint32_t local = 0; local < mesh.triangles.count;
                 ++local) {
                const PackedTriangle &triangle =
                    scene.triangles[mesh.triangles.offset + local];
                if (triangle.vertex0 >= mesh.vertices.count ||
                    triangle.vertex1 >= mesh.vertices.count ||
                    triangle.vertex2 >= mesh.vertices.count) {
                    add_index_error(report, "mesh", "triangle_vertex",
                                    index);
                    break;
                }
                if (triangle.material_slot >= mesh.material_slot_count) {
                    add_index_error(report, "mesh", "material_slot", index);
                    break;
                }
            }
        }
        validate_bvh_range(report, scene, mesh.bvh_nodes, mesh.triangles,
                           "mesh", index);
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
        std::size_t geometry_count = 0;
        std::uint32_t required_materials = 1;
        switch (instance.geometry_type) {
        case PackedGeometryType::Mesh:
            geometry_count = scene.meshes.size();
            if (instance.geometry_index < scene.meshes.size()) {
                required_materials =
                    scene.meshes[instance.geometry_index].material_slot_count;
            }
            break;
        case PackedGeometryType::Sphere:
            geometry_count = scene.spheres.size();
            break;
        case PackedGeometryType::MovingSphere:
            geometry_count = scene.moving_spheres.size();
            break;
        case PackedGeometryType::Medium:
            geometry_count = scene.media.size();
            break;
        default:
            add_index_error(report, "instance", "geometry_type", index);
            continue;
        }
        if (instance.geometry_index >= geometry_count) {
            add_index_error(report, "instance", "geometry_index", index);
        }
        if (instance.material_bindings.count < required_materials) {
            add_index_error(report, "instance", "material_bindings", index);
        } else if (valid_range(instance.material_bindings,
                               scene.material_bindings.size())) {
            for (std::uint32_t binding = 0;
                 binding < instance.material_bindings.count; ++binding) {
                if (scene.material_bindings[
                        instance.material_bindings.offset + binding] >=
                    scene.materials.size()) {
                    add_index_error(report, "instance", "material_id",
                                    index);
                    break;
                }
            }
            if (valid_range(instance.material_bindings,
                            scene.emitter_bindings.size())) {
                for (std::uint32_t binding = 0;
                     binding < instance.material_bindings.count; ++binding) {
                    const std::uint32_t emitter = scene.emitter_bindings[
                        instance.material_bindings.offset + binding];
                    if (!valid_optional_index(emitter,
                                              scene.lights.size())) {
                        add_index_error(report, "instance", "emitter_id",
                                        index);
                        break;
                    }
                    if (emitter != kInvalidPackedIndex) {
                        const PackedLight &light = scene.lights[emitter];
                        const std::uint32_t material = scene.material_bindings[
                            instance.material_bindings.offset + binding];
                        if ((light.flags & PACKED_LIGHT_BSDF_HITTABLE) == 0 ||
                            light.instance_id != index ||
                            light.material_id != material) {
                            add_index_error(report, "instance",
                                            "emitter_binding", index);
                            break;
                        }
                    }
                }
            }
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
        } else {
            for (std::uint32_t local = 0;
                 local < aggregate.instance_indices.count; ++local) {
                if (scene.aggregate_instance_indices[
                        aggregate.instance_indices.offset + local] >=
                    scene.instances.size()) {
                    add_index_error(report, "aggregate", "instance_id",
                                    index);
                    break;
                }
            }
        }
        validate_bvh_range(report, scene, aggregate.bvh_nodes,
                           aggregate.instance_indices, "aggregate", index);
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
        const PackedImageDesc &image = scene.images[index];
        if (!valid_range(image.texels, scene.image_texels.size())) {
            add_range_error(report, "image", "texels", index);
        } else {
            const std::uint64_t expected =
                static_cast<std::uint64_t>(image.width) * image.height *
                image.channels;
            if (image.width == 0 || image.height == 0 || image.channels == 0 ||
                image.channels > 4 || image.texels.count != expected) {
                add_index_error(report, "image", "dimensions", index);
            }
        }
    }

    std::vector<std::uint32_t> texture_depths(scene.texture_nodes.size(), 1);
    for (std::size_t index = 0; index < scene.texture_nodes.size(); ++index) {
        const PackedTextureNode &texture = scene.texture_nodes[index];
        auto valid_input = [&](std::uint32_t input) {
            return input == kInvalidPackedIndex || input < index;
        };
        if (!valid_input(texture.input0) || !valid_input(texture.input1) ||
            !valid_input(texture.input2)) {
            add_index_error(report, "texture", "input", index);
        }
        if (!valid_optional_index(texture.image_id, scene.images.size())) {
            add_index_error(report, "texture", "image_id", index);
        }
        if (!valid_optional_index(texture.perlin_id,
                                  scene.perlin_tables.size())) {
            add_index_error(report, "texture", "perlin_id", index);
        }
        const std::uint32_t inputs[3]{texture.input0, texture.input1,
                                      texture.input2};
        for (std::uint32_t input : inputs) {
            if (input != kInvalidPackedIndex && input < index) {
                texture_depths[index] = std::max(
                    texture_depths[index], texture_depths[input] + 1);
            }
        }
        if (texture_depths[index] > kPackedTextureStackCapacity) {
            std::ostringstream message;
            message << "texture[" << index << "] requires evaluation depth "
                    << texture_depths[index] << ", capacity is "
                    << kPackedTextureStackCapacity;
            report.errors.push_back(message.str());
        }
    }

    for (std::size_t index = 0; index < scene.materials.size(); ++index) {
        for (std::uint32_t texture : scene.materials[index].texture_ids) {
            if (!valid_optional_index(texture, scene.texture_nodes.size())) {
                add_index_error(report, "material", "texture_id", index);
                break;
            }
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
        scene.light_selection_probabilities.size() !=
            scene.non_delta_light_indices.size()) {
        report.errors.push_back("light selection buffers are inconsistent");
    }
    for (std::uint32_t light : scene.delta_light_indices) {
        if (light >= scene.lights.size() ||
            (scene.lights[light].flags & PACKED_LIGHT_DELTA) == 0) {
            report.errors.push_back("delta light index is invalid");
            break;
        }
    }
    for (std::uint32_t light : scene.non_delta_light_indices) {
        if (light >= scene.lights.size() ||
            (scene.lights[light].flags & PACKED_LIGHT_DELTA) != 0) {
            report.errors.push_back("non-delta light index is invalid");
            break;
        }
    }
    for (std::size_t index = 0;
         index < scene.non_delta_light_indices.size(); ++index) {
        const std::uint32_t light_id = scene.non_delta_light_indices[index];
        if (light_id < scene.lights.size() &&
            index < scene.light_selection_probabilities.size() &&
            std::abs(scene.lights[light_id].selection_probability -
                     scene.light_selection_probabilities[index]) > 1e-6f) {
            report.errors.push_back(
                "packed light selection probability is inconsistent");
            break;
        }
    }
    for (std::size_t index = 0; index < scene.lights.size(); ++index) {
        const PackedLight &light = scene.lights[index];
        if (!valid_range(light.distribution,
                         scene.light_distributions.size())) {
            add_range_error(report, "light", "distribution", index);
        }
        if (!valid_range(light.element_indices,
                         scene.light_element_indices.size())) {
            add_range_error(report, "light", "element_indices", index);
        }
        if (!valid_optional_index(light.instance_id, scene.instances.size())) {
            add_index_error(report, "light", "instance_id", index);
        }
        if (!valid_optional_index(light.material_id, scene.materials.size())) {
            add_index_error(report, "light", "material_id", index);
        }
        if (!valid_optional_index(light.image_id, scene.images.size())) {
            add_index_error(report, "light", "image_id", index);
        }
        if (!finite(light.data0) || !finite(light.data1) ||
            !finite(light.data2) || !finite(light.radiance) ||
            !finite(light.power) ||
            !std::isfinite(light.selection_probability) ||
            light.selection_probability < 0.0f) {
            add_index_error(report, "light", "numeric_data", index);
        }
        if (valid_range(light.element_indices,
                        scene.light_element_indices.size())) {
            for (std::uint32_t local = 0;
                 local < light.element_indices.count; ++local) {
                if (scene.light_element_indices[
                        light.element_indices.offset + local] >=
                    scene.triangles.size()) {
                    add_index_error(report, "light", "triangle_id", index);
                    break;
                }
            }
        }
        if (light.type == PackedLightType::TriangleEmitter ||
            light.type == PackedLightType::MeshEmitter) {
            const std::uint32_t count = light.element_indices.count;
            if (count == 0 || light.distribution.count != count * 2u) {
                add_index_error(report, "light", "mesh_distribution",
                                index);
            } else if (valid_range(light.distribution,
                                   scene.light_distributions.size())) {
                float probability_sum = 0.0f;
                float previous = 0.0f;
                for (std::uint32_t local = 0; local < count; ++local) {
                    const float probability = scene.light_distributions[
                        light.distribution.offset + local];
                    const float cdf = scene.light_distributions[
                        light.distribution.offset + count + local];
                    if (!std::isfinite(probability) || probability <= 0.0f ||
                        !std::isfinite(cdf) || cdf < previous) {
                        add_index_error(report, "light", "mesh_cdf", index);
                        break;
                    }
                    probability_sum += probability;
                    previous = cdf;
                }
                if (std::abs(probability_sum - 1.0f) > 1e-4f ||
                    std::abs(previous - 1.0f) > 1e-6f) {
                    add_index_error(report, "light", "mesh_probability",
                                    index);
                }
            }
            if (light.instance_id < scene.instances.size()) {
                const PackedInstance &instance =
                    scene.instances[light.instance_id];
                bool bound = false;
                for (std::uint32_t local = 0;
                     local < instance.material_bindings.count; ++local) {
                    const std::uint32_t binding =
                        instance.material_bindings.offset + local;
                    bound = bound ||
                            (binding < scene.emitter_bindings.size() &&
                             scene.emitter_bindings[binding] == index);
                }
                if (!bound) {
                    add_index_error(report, "light", "emitter_binding",
                                    index);
                }
            }
        } else if (light.type == PackedLightType::Environment) {
            const std::uint64_t width =
                static_cast<std::uint32_t>(light.data0.x);
            const std::uint64_t height =
                static_cast<std::uint32_t>(light.data0.y);
            const std::uint64_t expected =
                width * height + height * (width + 1) + height + height + 1;
            if (width == 0 || height == 0 ||
                light.distribution.count != expected) {
                add_index_error(report, "light", "environment_distribution",
                                index);
            }
        } else if (light.type == PackedLightType::SphereEmitter &&
                   light.instance_id < scene.instances.size()) {
            const PackedInstance &instance = scene.instances[light.instance_id];
            const std::uint32_t binding = instance.material_bindings.offset;
            if (binding >= scene.emitter_bindings.size() ||
                scene.emitter_bindings[binding] != index) {
                add_index_error(report, "light", "emitter_binding", index);
            }
        }
    }
    if (!scene.light_cdf.empty()) {
        float previous = 0.0f;
        float probability_sum = 0.0f;
        for (std::size_t index = 0; index < scene.light_cdf.size(); ++index) {
            const float probability = scene.light_selection_probabilities[index];
            const float cdf = scene.light_cdf[index];
            if (!std::isfinite(probability) || probability <= 0.0f ||
                !std::isfinite(cdf) || cdf < previous) {
                report.errors.push_back(
                    "light selection distribution is invalid");
                break;
            }
            probability_sum += probability;
            previous = cdf;
        }
        if (std::abs(probability_sum - 1.0f) > 1e-4f ||
            std::abs(previous - 1.0f) > 1e-6f) {
            report.errors.push_back(
                "light selection distribution is not normalized");
        }
    }
    return report;
}
