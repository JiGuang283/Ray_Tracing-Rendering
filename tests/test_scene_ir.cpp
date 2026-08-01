#include "scene_ir.h"
#include "test_harness.h"

#include <stdexcept>

namespace {

SceneDescription description_from(const std::string &source) {
    SceneDescription description;
    description.source_path = "test_scene.json";
    description.root = nlohmann::json::parse(source);
    return description;
}

bool parse_throws(const std::string &source) {
    try {
        (void)parse_scene_ir(description_from(source));
    } catch (const std::runtime_error &) {
        return true;
    }
    return false;
}

} // namespace

TEST_CASE(scene_ir_lowers_objects_and_lights_to_typed_nodes) {
    const SceneIR ir = parse_scene_ir(description_from(R"({
        "materials": {
            "mat": {"type": "lambertian", "color": [0.4, 0.5, 0.6]}
        },
        "objects": [{
            "type": "transform",
            "transform": {
                "translation": [2, 0, 0],
                "scale": [2, 1, 1]
            },
            "object": {
                "type": "sphere",
                "center": [0, 0, 0],
                "radius": 1,
                "material": "mat"
            }
        }],
        "lights": [{
            "type": "point",
            "position": [0, 3, -2],
            "intensity": [10, 9, 8]
        }]
    })"));

    REQUIRE(ir.objects.size() == 1);
    REQUIRE(ir.object_nodes.size() == 2);
    const auto *transform = std::get_if<TransformObjectIR>(
        &ir.object_nodes[ir.objects[0]].data);
    REQUIRE(transform != nullptr);
    REQUIRE(transform->child < ir.object_nodes.size());
    const auto *sphere = std::get_if<SphereObjectIR>(
        &ir.object_nodes[transform->child].data);
    REQUIRE(sphere != nullptr);
    REQUIRE(sphere->material == "mat");
    REQUIRE_NEAR(transform->transform.point_to_world(point3(1, 0, 0)).x(),
                 4.0, 1e-12);

    REQUIRE(ir.lights.size() == 1);
    const auto *light = std::get_if<PointLightIR>(&ir.lights[0].data);
    REQUIRE(light != nullptr);
    REQUIRE_NEAR(light->intensity.y(), 9.0, 1e-12);
}

TEST_CASE(scene_ir_builds_transformed_runtime_geometry) {
    const SceneIR ir = parse_scene_ir(description_from(R"({
        "world_accel": false,
        "materials": {
            "mat": {"type": "lambertian", "color": [0.4, 0.5, 0.6]}
        },
        "objects": [{
            "type": "transform",
            "transform": {
                "translation": [2, 0, 0],
                "scale": [2, 1, 1]
            },
            "object": {
                "type": "sphere",
                "center": [0, 0, 0],
                "radius": 1,
                "material": "mat"
            }
        }]
    })"));
    const SceneConfig config = build_scene_config(ir);

    hit_record record;
    REQUIRE(config.scene.world->hit(
        ray(point3(2, 0, -5), vec3(0, 0, 1)), 0.001, infinity, record));
    REQUIRE_NEAR(record.t, 4.0, 1e-9);
    REQUIRE_NEAR(record.p.x(), 2.0, 1e-9);
    REQUIRE(record.mat_ptr != nullptr);
}

TEST_CASE(scene_ir_parses_model_resources_without_building_them) {
    const SceneIR ir = parse_scene_ir(description_from(R"({
        "materials": {},
        "objects": [{
            "type": "model",
            "path": "models/example.glb",
            "scene": 2,
            "transform": {"translation": [1, 2, 3]},
            "material_overrides": {"Body": "paint"}
        }]
    })"));

    const auto *model =
        std::get_if<ModelObjectIR>(&ir.object_nodes[ir.objects[0]].data);
    REQUIRE(model != nullptr);
    REQUIRE(model->path == "models/example.glb");
    REQUIRE(model->scene_index == 2);
    REQUIRE(model->material_overrides.at("Body") == "paint");
    REQUIRE_NEAR(model->transform.point_to_world(point3(0, 0, 0)).z(), 3.0,
                 1e-12);
}

TEST_CASE(scene_ir_rejects_unknown_geometry_types) {
    REQUIRE(parse_throws(R"({
        "materials": {},
        "objects": [{"type": "mystery_shape"}]
    })"));
}

TEST_CASE(scene_ir_rejects_invalid_transform_matrices) {
    REQUIRE(parse_throws(R"({
        "materials": {},
        "objects": [{
            "type": "transform",
            "transform": {"matrix": [1, 0, 0]},
            "object": {"type": "sphere", "center": [0, 0, 0],
                       "material": "unused"}
        }]
    })"));
}

TEST_CASE(scene_ir_parses_and_validates_sample_clamp) {
    const SceneIR ir = parse_scene_ir(description_from(R"({
        "render": {"sample_clamp": 12.5},
        "materials": {},
        "objects": []
    })"));
    REQUIRE_NEAR(ir.preset.sample_clamp, 12.5, 1e-12);
    REQUIRE(parse_throws(R"({
        "render": {"sample_clamp": -1},
        "materials": {},
        "objects": []
    })"));
}
