#include "external/json.hpp"
#include "material.h"
#include "render_data/flat_intersector.h"
#include "render_data/scene_compiler.h"
#include "scene_description.h"
#include "scene_ir.h"
#include "triangle_scale_fixture.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

struct Options {
    std::filesystem::path catalog = "assets/scenes/catalog.json";
    std::set<int> scene_ids{1, 7, 8, 23, 59, 62, 64};
    int grid_width = 9;
    int grid_height = 7;
};

std::set<int> parse_ids(const std::string &value) {
    std::set<int> ids;
    std::istringstream stream(value);
    std::string part;
    while (std::getline(stream, part, ',')) {
        ids.insert(std::stoi(part));
    }
    if (ids.empty()) {
        throw std::runtime_error("--ids requires at least one scene ID");
    }
    return ids;
}

Options parse_options(int argc, char **argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto require_value = [&]() -> std::string {
            if (++index >= argc) {
                throw std::runtime_error(argument + " requires a value");
            }
            return argv[index];
        };
        if (argument == "--catalog") {
            options.catalog = require_value();
        } else if (argument == "--ids") {
            options.scene_ids = parse_ids(require_value());
        } else if (argument == "--grid-width") {
            options.grid_width = std::stoi(require_value());
        } else if (argument == "--grid-height") {
            options.grid_height = std::stoi(require_value());
        } else if (argument == "--help" || argument == "-h") {
            std::cout
                << "Usage: scene_intersection_check [--catalog PATH] "
                   "[--ids 1,7,...] [--grid-width N] [--grid-height N]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + argument);
        }
    }
    if (options.grid_width <= 0 || options.grid_height <= 0) {
        throw std::runtime_error("ray grid dimensions must be positive");
    }
    return options;
}

nlohmann::json load_json(const std::filesystem::path &path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open catalog: " + path.string());
    }
    nlohmann::json json;
    input >> json;
    return json;
}

std::filesystem::path resolve_scene_path(
    const std::filesystem::path &catalog,
    const std::filesystem::path &scene_path) {
    if (scene_path.is_absolute() || std::filesystem::exists(scene_path)) {
        return scene_path;
    }
    const std::filesystem::path relative = catalog.parent_path() / scene_path;
    return std::filesystem::exists(relative) ? relative : scene_path;
}

PackedMaterialType material_type(const MaterialDescription &description) {
    if (std::holds_alternative<LambertianMaterialDescription>(description)) {
        return PackedMaterialType::Lambertian;
    }
    if (std::holds_alternative<MetalMaterialDescription>(description)) {
        return PackedMaterialType::Metal;
    }
    if (std::holds_alternative<DielectricMaterialDescription>(description)) {
        return PackedMaterialType::Dielectric;
    }
    if (std::holds_alternative<DiffuseLightMaterialDescription>(description)) {
        return PackedMaterialType::DiffuseLight;
    }
    if (std::holds_alternative<PrincipledMaterialDescription>(description)) {
        return PackedMaterialType::Principled;
    }
    if (std::holds_alternative<IsotropicMaterialDescription>(description)) {
        return PackedMaterialType::Isotropic;
    }
    throw std::runtime_error("reference hit has no material description");
}

double normal_dot(Float3 packed, const vec3 &reference) {
    return packed.x * reference.x() + packed.y * reference.y() +
           packed.z * reference.z();
}

PackedRay make_camera_ray(const PackedCamera &camera, int x, int y,
                          int width, int height) {
    const float u = (static_cast<float>(x) + 0.37f) / width;
    const float v = (static_cast<float>(y) + 0.61f) / height;
    PackedRay ray;
    ray.origin = camera.origin;
    ray.direction = {
        camera.lower_left_corner.x + u * camera.horizontal.x +
            v * camera.vertical.x - camera.origin.x,
        camera.lower_left_corner.y + u * camera.horizontal.y +
            v * camera.vertical.y - camera.origin.y,
        camera.lower_left_corner.z + u * camera.horizontal.z +
            v * camera.vertical.z - camera.origin.z};
    ray.t_min = 0.001f;
    ray.t_max = 1.0e30f;
    ray.time = camera.time0 + 0.43f * (camera.time1 - camera.time0);
    return ray;
}

struct ComparisonResult {
    std::size_t rays = 0;
    std::size_t hits = 0;
    std::size_t errors = 0;
};

ComparisonResult compare_triangle_scale_fixture() {
    const triangle_scale_fixture::Fixture fixture =
        triangle_scale_fixture::make_fixture();
    const CompiledSceneView view = make_scene_view(fixture.packed);
    ComparisonResult result;
    std::size_t reported_errors = 0;
    auto report = [&](std::size_t index, const std::string &message) {
        ++result.errors;
        if (reported_errors++ < 8) {
            std::cerr << "TRIANGLE_SCALE_MISMATCH ray=" << index
                      << " error=" << message << '\n';
        }
    };

    const ValidationReport validation =
        validate_compiled_scene(fixture.packed);
    if (!validation.ok()) {
        report(0, validation.errors.front());
        return result;
    }

    for (std::size_t index = 0; index < fixture.rays.size(); ++index) {
        ++result.rays;
        const PackedRay &packed_ray = fixture.rays[index];
        const triangle_scale_fixture::ExpectedHit &expected =
            fixture.expected[index];
        const ray reference_ray(
            point3(packed_ray.origin.x, packed_ray.origin.y,
                   packed_ray.origin.z),
            vec3(packed_ray.direction.x, packed_ray.direction.y,
                 packed_ray.direction.z));
        hit_record reference_hit;
        const bool reference_found = fixture.reference->hit(
            reference_ray, packed_ray.t_min, packed_ray.t_max,
            reference_hit);
        PackedHit packed_hit;
        const bool packed_found =
            intersect_compiled_scene(view, packed_ray, packed_hit);
        if (reference_found != expected.found ||
            packed_found != expected.found) {
            std::ostringstream message;
            message << "hit/miss differs expected=" << expected.found
                    << " reference=" << reference_found
                    << " packed=" << packed_found;
            report(index, message.str());
            continue;
        }
        if (!expected.found) {
            continue;
        }
        ++result.hits;
        const double t_tolerance =
            5e-5 * std::max(1.0, std::abs(static_cast<double>(expected.t)));
        if (std::abs(reference_hit.t - expected.t) > t_tolerance ||
            std::abs(static_cast<double>(packed_hit.t) - expected.t) >
                t_tolerance) {
            report(index, "hit distance differs");
        }
        if (std::abs(reference_hit.u - expected.barycentric_u) > 2e-5 ||
            std::abs(reference_hit.v - expected.barycentric_v) > 2e-5 ||
            std::abs(packed_hit.barycentric_u - expected.barycentric_u) >
                2e-5f ||
            std::abs(packed_hit.barycentric_v - expected.barycentric_v) >
                2e-5f) {
            report(index, "barycentrics differ");
        }
        if (reference_hit.primitive_id != 0 || packed_hit.instance_id != 0 ||
            packed_hit.primitive_id != 0) {
            report(index, "primitive or instance ID differs");
        }
    }
    return result;
}

ComparisonResult compare_scene(const std::filesystem::path &path,
                               const Options &options) {
    const SceneDescription description = load_scene_description(path.string());
    const SceneIR ir = parse_scene_ir(description);
    const CompiledScene packed = compile_scene(ir);
    const CompiledSceneView view = make_scene_view(packed);
    const SceneConfig reference = build_scene_config(ir);
    ComparisonResult result;
    std::size_t reported_errors = 0;
    auto report = [&](int x, int y, const std::string &message) {
        ++result.errors;
        if (reported_errors++ < 8) {
            std::cerr << "RAY_MISMATCH scene=" << path.string()
                      << " x=" << x << " y=" << y
                      << " error=" << message << '\n';
        }
    };

    for (int y = 0; y < options.grid_height; ++y) {
        for (int x = 0; x < options.grid_width; ++x) {
            ++result.rays;
            const PackedRay packed_ray = make_camera_ray(
                packed.camera, x, y, options.grid_width,
                options.grid_height);
            const ray reference_ray(
                point3(packed_ray.origin.x, packed_ray.origin.y,
                       packed_ray.origin.z),
                vec3(packed_ray.direction.x, packed_ray.direction.y,
                     packed_ray.direction.z),
                packed_ray.time);
            const std::uint32_t seed = static_cast<std::uint32_t>(
                1234 + y * options.grid_width + x);
            RNG reference_rng(seed);
            hit_record reference_hit;
            const bool reference_found = reference.scene.world->hit(
                reference_ray, packed_ray.t_min, packed_ray.t_max,
                reference_hit, reference_rng);
            RNG packed_rng(seed);
            PackedHit packed_hit;
            const bool packed_found = intersect_compiled_scene(
                view, packed_ray, packed_hit, &packed_rng);
            if (reference_found != packed_found) {
                report(x, y, "hit/miss differs");
                continue;
            }
            if (!packed_found) {
                continue;
            }
            ++result.hits;
            if (std::abs(static_cast<double>(packed_hit.t) - reference_hit.t) >
                5e-4) {
                report(x, y, "t differs");
                continue;
            }
            PackedSurfaceInteraction surface;
            if (!reconstruct_compiled_hit(view, packed_ray, packed_hit,
                                          surface)) {
                report(x, y, "packed hit reconstruction failed");
                continue;
            }
            const double uv_tolerance =
                (surface.flags & PACKED_HIT_SPHERE) != 0 ? 1e-3 : 5e-4;
            if (std::abs(static_cast<double>(surface.uv.x) - reference_hit.u) >
                    uv_tolerance ||
                std::abs(static_cast<double>(surface.uv.y) - reference_hit.v) >
                    uv_tolerance) {
                std::ostringstream message;
                message << "UV differs reference=(" << reference_hit.u
                        << ',' << reference_hit.v << ") packed=("
                        << surface.uv.x << ',' << surface.uv.y
                        << ") reference_primitive="
                        << reference_hit.primitive_id
                        << " packed_triangle=" << packed_hit.primitive_id
                        << " packed_source_primitive="
                        << surface.primitive_id << " reference_t="
                        << reference_hit.t << " packed_t=" << packed_hit.t
                        << " packed_barycentric=("
                        << packed_hit.barycentric_u << ','
                        << packed_hit.barycentric_v << ')';
                report(x, y, message.str());
            }
            if (normal_dot(surface.geometric_normal,
                           reference_hit.geometric_normal) < 0.999) {
                report(x, y, "geometric normal differs");
            }
            if (normal_dot(surface.shading_normal, reference_hit.normal) <
                0.999) {
                report(x, y, "shading normal differs");
            }
            if (reference_hit.mat_ptr == nullptr ||
                surface.material_id >= packed.materials.size() ||
                material_type(reference_hit.mat_ptr->description()) !=
                    packed.materials[surface.material_id].type) {
                report(x, y, "material type differs");
            }
        }
    }
    return result;
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Options options = parse_options(argc, argv);
        const nlohmann::json catalog = load_json(options.catalog);
        std::size_t passed = 0;
        std::size_t failed = 0;
        const ComparisonResult scale_result =
            compare_triangle_scale_fixture();
        const bool scale_ok = scale_result.errors == 0;
        std::cout << "TRIANGLE_SCALE_CHECK rays=" << scale_result.rays
                  << " hits=" << scale_result.hits
                  << " errors=" << scale_result.errors << '\n';
        for (const nlohmann::json &entry : catalog.at("scenes")) {
            const int id = entry.at("id").get<int>();
            if (options.scene_ids.count(id) == 0) {
                continue;
            }
            const std::filesystem::path path = resolve_scene_path(
                options.catalog,
                entry.at("path").get<std::string>());
            const ComparisonResult result = compare_scene(path, options);
            const bool ok = result.errors == 0;
            ok ? ++passed : ++failed;
            std::cout << "INTERSECTION_CHECK id=" << id
                      << " rays=" << result.rays << " hits=" << result.hits
                      << " errors=" << result.errors << '\n';
        }
        std::cout << "INTERSECTION_CHECK_SUMMARY passed=" << passed
                  << " failed=" << failed << '\n';
        return failed == 0 && scale_ok ? 0 : 1;
    } catch (const std::exception &error) {
        std::cerr << "scene_intersection_check: " << error.what() << '\n';
        return 1;
    }
}
