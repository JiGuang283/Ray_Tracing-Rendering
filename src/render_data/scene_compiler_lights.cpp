#include "scene_compiler_internal.h"

namespace scene_compiler_detail {

std::uint32_t SceneCompiler::append_light(PackedLight light)
{
        const std::uint32_t id =
            checked_index(m_scene.lights.size(), "light count");
        m_scene.lights.push_back(light);
        if ((light.flags & PACKED_LIGHT_DELTA) != 0) {
            m_scene.delta_light_indices.push_back(id);
        } else {
            m_scene.non_delta_light_indices.push_back(id);
        }
        return id;
    }

Float3 SceneCompiler::mesh_position(const PackedMesh &mesh,
                         std::uint32_t vertex) const
{
        const Float4 &value = m_scene.positions[mesh.vertices.offset + vertex];
        return {value.x, value.y, value.z};
    }

void SceneCompiler::append_mesh_emitter(std::uint32_t instance_id,
                             std::uint32_t material_slot,
                             std::uint32_t material_id)
{
        struct EmitterTriangle {
            std::uint32_t index = 0;
            double area = 0.0;
            Float3 emission;
        };

        const PackedInstance &instance = m_scene.instances[instance_id];
        const PackedMesh &mesh = m_scene.meshes[instance.geometry_index];
        const PackedTransform &transform =
            m_scene.transforms[instance.transform_id];
        constexpr double sample_points[4][3] = {
            {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0},
            {0.6, 0.2, 0.2},
            {0.2, 0.6, 0.2},
            {0.2, 0.2, 0.6}};
        std::vector<EmitterTriangle> entries;
        double total_area = 0.0;
        double integrated[3]{};
        for (std::uint32_t local = 0; local < mesh.triangles.count; ++local) {
            const std::uint32_t triangle_id = mesh.triangles.offset + local;
            const PackedTriangle &triangle = m_scene.triangles[triangle_id];
            if (triangle.material_slot != material_slot) {
                continue;
            }
            const Float3 object0 = mesh_position(mesh, triangle.vertex0);
            const Float3 object1 = mesh_position(mesh, triangle.vertex1);
            const Float3 object2 = mesh_position(mesh, triangle.vertex2);
            const Float3 world0 = transform_packed_point(transform, object0);
            const Float3 world1 = transform_packed_point(transform, object1);
            const Float3 world2 = transform_packed_point(transform, object2);
            const double area =
                0.5 * packed_length(packed_cross(
                          packed_subtract(world1, world0),
                          packed_subtract(world2, world0)));
            if (!(area > 0.0) || !std::isfinite(area)) {
                continue;
            }

            const Float2 uv0 =
                m_scene.uv0[mesh.vertices.offset + triangle.vertex0];
            const Float2 uv1 =
                m_scene.uv0[mesh.vertices.offset + triangle.vertex1];
            const Float2 uv2 =
                m_scene.uv0[mesh.vertices.offset + triangle.vertex2];
            const Float4 color0 = m_scene.vertex_colors[
                mesh.vertices.offset + triangle.vertex0];
            const Float4 color1 = m_scene.vertex_colors[
                mesh.vertices.offset + triangle.vertex1];
            const Float4 color2 = m_scene.vertex_colors[
                mesh.vertices.offset + triangle.vertex2];
            Float3 estimate{};
            for (const auto &weights : sample_points) {
                PackedTextureEvalContext context;
                context.position =
                    packed_barycentric(world0, world1, world2, weights);
                if ((triangle.flags & PACKED_TRIANGLE_HAS_UV) != 0) {
                    context.uv0 = {
                        static_cast<float>(weights[0] * uv0.x +
                                           weights[1] * uv1.x +
                                           weights[2] * uv2.x),
                        static_cast<float>(weights[0] * uv0.y +
                                           weights[1] * uv1.y +
                                           weights[2] * uv2.y)};
                } else {
                    context.uv0 = {static_cast<float>(weights[1]),
                                   static_cast<float>(weights[2])};
                }
                if ((triangle.flags & PACKED_TRIANGLE_HAS_COLOR) != 0) {
                    context.vertex_color = {
                        static_cast<float>(weights[0] * color0.x +
                                           weights[1] * color1.x +
                                           weights[2] * color2.x),
                        static_cast<float>(weights[0] * color0.y +
                                           weights[1] * color1.y +
                                           weights[2] * color2.y),
                        static_cast<float>(weights[0] * color0.z +
                                           weights[1] * color1.z +
                                           weights[2] * color2.z),
                        static_cast<float>(weights[0] * color0.w +
                                           weights[1] * color1.w +
                                           weights[2] * color2.w)};
                }
                Float3 emission;
                if (!evaluate_packed_material_emission(
                        make_scene_view(m_scene), material_id, context,
                        emission)) {
                    throw std::runtime_error(
                        "failed to evaluate packed mesh emission");
                }
                estimate.x += emission.x;
                estimate.y += emission.y;
                estimate.z += emission.z;
            }
            estimate = finite_nonnegative(
                {estimate.x * 0.25f, estimate.y * 0.25f,
                 estimate.z * 0.25f});
            entries.push_back({triangle_id, area, estimate});
            total_area += area;
            integrated[0] += area * estimate.x;
            integrated[1] += area * estimate.y;
            integrated[2] += area * estimate.z;
        }
        if (entries.empty() || !(total_area > 0.0) ||
            !std::isfinite(total_area)) {
            return;
        }

        double emission_weight_sum = 0.0;
        for (const EmitterTriangle &entry : entries) {
            emission_weight_sum +=
                entry.area * (0.2126 * entry.emission.x +
                              0.7152 * entry.emission.y +
                              0.0722 * entry.emission.z);
        }
        const std::size_t element_offset =
            m_scene.light_element_indices.size();
        const std::size_t distribution_offset =
            m_scene.light_distributions.size();
        std::vector<double> raw_probabilities;
        raw_probabilities.reserve(entries.size());
        for (const EmitterTriangle &entry : entries) {
            const double area_probability = entry.area / total_area;
            const double emission_probability =
                emission_weight_sum > 0.0
                    ? entry.area *
                          (0.2126 * entry.emission.x +
                           0.7152 * entry.emission.y +
                           0.0722 * entry.emission.z) /
                          emission_weight_sum
                    : area_probability;
            raw_probabilities.push_back(
                0.95 * emission_probability + 0.05 * area_probability);
            m_scene.light_element_indices.push_back(entry.index);
        }
        const double minimum_probability =
            static_cast<double>(std::numeric_limits<float>::min());
        double adjusted_sum = 0.0;
        for (double &probability : raw_probabilities) {
            probability = std::max(probability, minimum_probability);
            adjusted_sum += probability;
        }
        std::vector<float> probabilities;
        probabilities.reserve(raw_probabilities.size());
        float float_sum = 0.0f;
        std::size_t largest_index = 0;
        for (std::size_t index = 0; index < raw_probabilities.size();
             ++index) {
            const float probability = static_cast<float>(
                raw_probabilities[index] / adjusted_sum);
            probabilities.push_back(probability);
            float_sum += probability;
            if (probability > probabilities[largest_index]) {
                largest_index = index;
            }
        }
        probabilities[largest_index] += 1.0f - float_sum;
        for (float probability : probabilities) {
            m_scene.light_distributions.push_back(probability);
        }
        double cumulative = 0.0;
        for (float probability : probabilities) {
            cumulative += probability;
            m_scene.light_distributions.push_back(
                static_cast<float>(cumulative));
        }
        m_scene.light_distributions.back() = 1.0f;

        const PackedMaterial &material = m_scene.materials[material_id];
        const float side_factor =
            (material.flags & PACKED_MATERIAL_DOUBLE_SIDED) != 0 ? 2.0f
                                                                  : 1.0f;
        PackedLight light;
        light.type = entries.size() == 1
                         ? PackedLightType::TriangleEmitter
                         : PackedLightType::MeshEmitter;
        light.flags = PACKED_LIGHT_BSDF_HITTABLE;
        if (side_factor == 2.0f) {
            light.flags |= PACKED_LIGHT_DOUBLE_SIDED;
        }
        light.instance_id = instance_id;
        light.material_id = material_id;
        light.element_indices = checked_range(
            element_offset, entries.size(), "light triangle indices");
        light.distribution = checked_range(
            distribution_offset, entries.size() * 2,
            "light triangle distribution");
        light.data0 = {checked_float(total_area, "emitter area"),
                       static_cast<float>(entries.size()), 0.0f, 0.0f};
        light.radiance = {
            checked_float(integrated[0] / total_area, "emitter radiance"),
            checked_float(integrated[1] / total_area, "emitter radiance"),
            checked_float(integrated[2] / total_area, "emitter radiance"),
            0.0f};
        light.power = {
            checked_float(side_factor * pi * integrated[0],
                          "emitter power"),
            checked_float(side_factor * pi * integrated[1],
                          "emitter power"),
            checked_float(side_factor * pi * integrated[2],
                          "emitter power"),
            0.0f};
        const std::uint32_t light_id = append_light(light);
        m_scene.emitter_bindings[instance.material_bindings.offset +
                                 material_slot] = light_id;
    }

void SceneCompiler::append_sphere_emitter(std::uint32_t instance_id,
                               std::uint32_t material_id)
{
        const PackedInstance &instance = m_scene.instances[instance_id];
        const PackedSphere &sphere = m_scene.spheres[instance.geometry_index];
        const PackedTransform &transform =
            m_scene.transforms[instance.transform_id];
        constexpr Float3 normals[6] = {
            {1, 0, 0}, {-1, 0, 0}, {0, 1, 0},
            {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
        const float *m = transform.object_to_world;
        const double determinant =
            m[0] * (m[5] * m[10] - m[6] * m[9]) -
            m[1] * (m[4] * m[10] - m[6] * m[8]) +
            m[2] * (m[4] * m[9] - m[5] * m[8]);
        const float *inverse = transform.world_to_object;
        const double base_patch =
            4.0 * pi * sphere.radius * sphere.radius / 6.0;
        double total_area = 0.0;
        double integrated[3]{};
        for (Float3 normal : normals) {
            const Float3 inverse_transpose{
                inverse[0] * normal.x + inverse[4] * normal.y +
                    inverse[8] * normal.z,
                inverse[1] * normal.x + inverse[5] * normal.y +
                    inverse[9] * normal.z,
                inverse[2] * normal.x + inverse[6] * normal.y +
                    inverse[10] * normal.z};
            const double area = base_patch * std::abs(determinant) *
                                packed_length(inverse_transpose);
            PackedTextureEvalContext context;
            const Float3 object_position{
                sphere.center.x + sphere.radius * normal.x,
                sphere.center.y + sphere.radius * normal.y,
                sphere.center.z + sphere.radius * normal.z};
            context.position =
                transform_packed_point(transform, object_position);
            if ((sphere.flags & PACKED_SPHERE_FLIP_ORIENTATION) != 0) {
                normal = {-normal.x, -normal.y, -normal.z};
            }
            const double theta = std::acos(std::clamp(
                -static_cast<double>(normal.y), -1.0, 1.0));
            const double phi =
                std::atan2(-static_cast<double>(normal.z), normal.x) + pi;
            context.uv0 = {static_cast<float>(phi / (2.0 * pi)),
                           static_cast<float>(theta / pi)};
            Float3 emission;
            if (!evaluate_packed_material_emission(
                    make_scene_view(m_scene), material_id, context,
                    emission)) {
                throw std::runtime_error(
                    "failed to evaluate packed sphere emission");
            }
            emission = finite_nonnegative(emission);
            total_area += area;
            integrated[0] += area * emission.x;
            integrated[1] += area * emission.y;
            integrated[2] += area * emission.z;
        }
        if (!(total_area > 0.0) || !std::isfinite(total_area)) {
            return;
        }
        const PackedMaterial &material = m_scene.materials[material_id];
        const float side_factor =
            (material.flags & PACKED_MATERIAL_DOUBLE_SIDED) != 0 ? 2.0f
                                                                  : 1.0f;
        PackedLight light;
        light.type = PackedLightType::SphereEmitter;
        light.flags = PACKED_LIGHT_BSDF_HITTABLE;
        if (side_factor == 2.0f) {
            light.flags |= PACKED_LIGHT_DOUBLE_SIDED;
        }
        light.instance_id = instance_id;
        light.material_id = material_id;
        light.data0 = {sphere.center.x, sphere.center.y, sphere.center.z,
                       sphere.radius};
        light.data1.x = checked_float(total_area, "sphere emitter area");
        light.radiance = {
            checked_float(integrated[0] / total_area,
                          "sphere emitter radiance"),
            checked_float(integrated[1] / total_area,
                          "sphere emitter radiance"),
            checked_float(integrated[2] / total_area,
                          "sphere emitter radiance"),
            0.0f};
        light.power = {
            checked_float(side_factor * pi * integrated[0],
                          "sphere emitter power"),
            checked_float(side_factor * pi * integrated[1],
                          "sphere emitter power"),
            checked_float(side_factor * pi * integrated[2],
                          "sphere emitter power"),
            0.0f};
        const std::uint32_t light_id = append_light(light);
        m_scene.emitter_bindings[instance.material_bindings.offset] =
            light_id;
    }

void SceneCompiler::append_environment_light(const EnvironmentLightIR &environment)
{
        const std::string path = resolve_asset_path(
            m_context.source_path, environment.path);
        const auto image = m_context.resources.load_image(path);
        const ImageId image_id = m_resources.compile_image(image);
        const int width = image->width();
        const int height = image->height();
        auto linear_component = [&](int x, int y, int channel) {
            double value = image->component(x, y, channel);
            if (!image->is_hdr()) {
                value = value <= 0.04045
                            ? value / 12.92
                            : std::pow((value + 0.055) / 1.055, 2.4);
            }
            return value;
        };
        std::vector<double> weights(
            static_cast<std::size_t>(width) * height);
        std::vector<double> row_sums(static_cast<std::size_t>(height), 0.0);
        double total_weight = 0.0;
        const bool is_probe = width == height;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const double luminance =
                    0.2126 * linear_component(x, y, 0) +
                    0.7152 * linear_component(x, y, 1) +
                    0.0722 * linear_component(x, y, 2);
                double solid_angle_weight = 0.0;
                if (is_probe) {
                    const double centered_x =
                        2.0 * (static_cast<double>(x) + 0.5) / width - 1.0;
                    const double centered_y =
                        1.0 - 2.0 * (static_cast<double>(y) + 0.5) / height;
                    const double radial = std::sqrt(
                        centered_x * centered_x + centered_y * centered_y);
                    if (radial <= 1.0) {
                        solid_angle_weight =
                            radial > 1e-12 ? std::sin(pi * radial) / radial
                                           : pi;
                    }
                } else {
                    solid_angle_weight = std::sin(
                        pi * (static_cast<double>(y) + 0.5) / height);
                }
                const double weight =
                    std::max(0.0, luminance * solid_angle_weight);
                weights[static_cast<std::size_t>(y) * width + x] = weight;
                row_sums[static_cast<std::size_t>(y)] += weight;
                total_weight += weight;
            }
        }

        const std::size_t offset = m_scene.light_distributions.size();
        for (int y = 0; y < height; ++y) {
            const double row_sum = row_sums[static_cast<std::size_t>(y)];
            for (int x = 0; x < width; ++x) {
                const double value =
                    weights[static_cast<std::size_t>(y) * width + x];
                m_scene.light_distributions.push_back(
                    row_sum > 0.0 ? static_cast<float>(value / row_sum)
                                  : 0.0f);
            }
        }
        for (int y = 0; y < height; ++y) {
            float cumulative = 0.0f;
            m_scene.light_distributions.push_back(0.0f);
            const double row_sum = row_sums[static_cast<std::size_t>(y)];
            for (int x = 0; x < width; ++x) {
                const double value =
                    weights[static_cast<std::size_t>(y) * width + x];
                cumulative += row_sum > 0.0
                                  ? static_cast<float>(value / row_sum)
                                  : 0.0f;
                m_scene.light_distributions.push_back(cumulative);
            }
            if (row_sum > 0.0) {
                m_scene.light_distributions.back() = 1.0f;
            }
        }
        for (double row_sum : row_sums) {
            m_scene.light_distributions.push_back(
                total_weight > 0.0
                    ? static_cast<float>(row_sum / total_weight)
                    : 0.0f);
        }
        float cumulative = 0.0f;
        m_scene.light_distributions.push_back(0.0f);
        for (double row_sum : row_sums) {
            cumulative += total_weight > 0.0
                              ? static_cast<float>(row_sum / total_weight)
                              : 0.0f;
            m_scene.light_distributions.push_back(cumulative);
        }
        if (total_weight > 0.0) {
            m_scene.light_distributions.back() = 1.0f;
        }

        PackedLight light;
        light.type = PackedLightType::Environment;
        light.flags = PACKED_LIGHT_INFINITE | PACKED_LIGHT_BSDF_HITTABLE;
        if (is_probe) {
            light.flags |= PACKED_LIGHT_ENVIRONMENT_PROBE;
        }
        if (!image->is_hdr()) {
            light.flags |= PACKED_LIGHT_ENVIRONMENT_SRGB;
        }
        light.image_id = image_id.value;
        light.distribution = checked_range(
            offset, m_scene.light_distributions.size() - offset,
            "environment distribution");
        light.data0 = {static_cast<float>(width),
                       static_cast<float>(height), 0.0f, 0.0f};
        const double jacobian_scale = is_probe ? 4.0 * pi : 2.0 * pi * pi;
        const double total_power =
            total_weight * jacobian_scale /
            (static_cast<double>(width) * height);
        light.power = {checked_float(total_power, "environment power"),
                       checked_float(total_power, "environment power"),
                       checked_float(total_power, "environment power"),
                       0.0f};
        append_light(light);
    }

void SceneCompiler::compile_explicit_lights()
{
        for (const LightIR &source : m_ir.lights) {
            std::visit(
                [&](const auto &light_ir) {
                    using T = std::decay_t<decltype(light_ir)>;
                    PackedLight light;
                    if constexpr (std::is_same_v<T, PointLightIR>) {
                        light.type = PackedLightType::Point;
                        light.flags = PACKED_LIGHT_DELTA;
                        light.data0 = pack_vec4(light_ir.position, 1.0);
                        light.radiance = pack_vec4(light_ir.intensity, 0.0);
                        light.power =
                            pack_vec4(4.0 * pi * light_ir.intensity, 0.0);
                        append_light(light);
                    } else if constexpr (std::is_same_v<T,
                                                        DirectionalLightIR>) {
                        if (light_ir.direction.near_zero()) {
                            throw std::runtime_error(
                                "directional light has a zero direction");
                        }
                        light.type = PackedLightType::Directional;
                        light.flags = PACKED_LIGHT_DELTA;
                        light.data0 =
                            pack_vec4(unit_vector(light_ir.direction), 0.0);
                        light.radiance = pack_vec4(light_ir.radiance, 0.0);
                        light.power = pack_vec4(light_ir.radiance, 0.0);
                        append_light(light);
                    } else if constexpr (std::is_same_v<T, SpotLightIR>) {
                        if (light_ir.direction.near_zero()) {
                            throw std::runtime_error(
                                "spot light has a zero direction");
                        }
                        light.type = PackedLightType::Spot;
                        light.flags = PACKED_LIGHT_DELTA;
                        light.data0 = pack_vec4(
                            light_ir.position,
                            std::cos(degrees_to_radians(light_ir.cutoff)));
                        light.data1 =
                            pack_vec4(unit_vector(light_ir.direction), 0.0);
                        light.radiance = pack_vec4(light_ir.intensity, 0.0);
                        light.power = pack_vec4(light_ir.intensity, 0.0);
                        append_light(light);
                    } else if constexpr (std::is_same_v<T, QuadLightIR>) {
                        const double area =
                            cross(light_ir.u, light_ir.v).length();
                        light.type = PackedLightType::Quad;
                        light.data0 = pack_vec4(light_ir.origin, 1.0);
                        light.data1 = pack_vec4(light_ir.u, 0.0);
                        light.data2 = pack_vec4(light_ir.v, 0.0);
                        light.radiance = pack_vec4(light_ir.intensity, 0.0);
                        light.power = pack_vec4(
                            pi * area * light_ir.intensity, 0.0);
                        append_light(light);
                    } else if constexpr (std::is_same_v<
                                             T, EnvironmentLightIR>) {
                        append_environment_light(light_ir);
                    }
                },
                source.data);
        }
    }

void SceneCompiler::compile_auto_emitters()
{
        if (!m_ir.auto_emitters || m_scene.aggregates.empty()) {
            return;
        }
        const PackedAggregate &world = m_scene.aggregates[0];
        for (std::uint32_t local = 0; local < world.instance_indices.count;
             ++local) {
            const std::uint32_t instance_id =
                m_scene.aggregate_instance_indices[
                    world.instance_indices.offset + local];
            const PackedInstance &instance = m_scene.instances[instance_id];
            if (instance.geometry_type == PackedGeometryType::Sphere) {
                const std::uint32_t material_id =
                    m_scene.material_bindings[
                        instance.material_bindings.offset];
                if ((m_scene.materials[material_id].flags &
                     PACKED_MATERIAL_EMISSIVE) != 0) {
                    append_sphere_emitter(instance_id, material_id);
                }
            } else if (instance.geometry_type == PackedGeometryType::Mesh) {
                const PackedMesh &mesh =
                    m_scene.meshes[instance.geometry_index];
                for (std::uint32_t slot = 0;
                     slot < mesh.material_slot_count; ++slot) {
                    const std::uint32_t material_id =
                        m_scene.material_bindings[
                            instance.material_bindings.offset + slot];
                    if ((m_scene.materials[material_id].flags &
                         PACKED_MATERIAL_EMISSIVE) != 0) {
                        append_mesh_emitter(instance_id, slot, material_id);
                    }
                }
            }
        }
    }

void SceneCompiler::build_light_selection_distribution()
{
        if (m_scene.non_delta_light_indices.empty()) {
            return;
        }
        std::vector<double> weights;
        weights.reserve(m_scene.non_delta_light_indices.size());
        double total = 0.0;
        for (std::uint32_t light_id : m_scene.non_delta_light_indices) {
            const double weight =
                packed_luminance(m_scene.lights[light_id].power);
            weights.push_back(weight);
            total += weight;
        }
        const double uniform =
            1.0 / static_cast<double>(weights.size());
        std::vector<float> probabilities;
        probabilities.reserve(weights.size());
        float float_sum = 0.0f;
        std::size_t largest_index = 0;
        for (std::size_t index = 0; index < weights.size(); ++index) {
            const double weight = weights[index];
            const double power_probability =
                total > 0.0 ? weight / total : uniform;
            const float probability = static_cast<float>(
                0.95 * power_probability + 0.05 * uniform);
            probabilities.push_back(probability);
            float_sum += probability;
            if (probability > probabilities[largest_index]) {
                largest_index = index;
            }
        }
        probabilities[largest_index] += 1.0f - float_sum;

        float cumulative = 0.0f;
        for (std::size_t index = 0; index < probabilities.size(); ++index) {
            const float probability = probabilities[index];
            m_scene.light_selection_probabilities.push_back(probability);
            cumulative += probability;
            m_scene.light_cdf.push_back(cumulative);
            const std::uint32_t light_id =
                m_scene.non_delta_light_indices[index];
            m_scene.lights[light_id].selection_probability = probability;
        }
        m_scene.light_cdf.back() = 1.0f;
    }

void SceneCompiler::compile_lights()
{
        compile_explicit_lights();
        compile_auto_emitters();
        build_light_selection_distribution();
    }

} // namespace scene_compiler_detail
