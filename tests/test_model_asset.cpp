#include "mesh_instance.h"
#include "resource_registry.h"
#include "test_harness.h"

#include <filesystem>
#include <fstream>

namespace {

const char *kTriangleBuffer =
    "AAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAAAAAgD8AAAAAAAAAAAAAAAAAAIA/"
    "AAAAAAAAAAAAAIA/AAAAAAAAAAAAAIA/AAAAAAAAAAAAAIA/AAAAAAAAAAAAAIA/"
    "AAABAAIA";

std::filesystem::path write_test_model() {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        "raytracer_model_asset_test.gltf";
    std::ofstream output(path);
    output << R"({
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"name": "Main", "nodes": [0]}],
        "nodes": [{"name": "Moved", "mesh": 0,
                   "translation": [2, 0, 0]}],
        "meshes": [{
            "name": "Triangle",
            "primitives": [{
                "attributes": {"POSITION": 0, "NORMAL": 1,
                               "TEXCOORD_0": 2},
                "indices": 3,
                "material": 0
            }]
        }],
        "materials": [{
            "name": "Paint",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.8, 0.2, 0.1, 1],
                "metallicFactor": 0,
                "roughnessFactor": 0.6
            }
        }],
        "buffers": [{
            "byteLength": 102,
            "uri": "data:application/octet-stream;base64,)"
           << kTriangleBuffer << R"("
        }],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 36,
             "target": 34962},
            {"buffer": 0, "byteOffset": 36, "byteLength": 36,
             "target": 34962},
            {"buffer": 0, "byteOffset": 72, "byteLength": 24,
             "target": 34962},
            {"buffer": 0, "byteOffset": 96, "byteLength": 6,
             "target": 34963}
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3,
             "type": "VEC3", "min": [0, 0, 0], "max": [1, 1, 0]},
            {"bufferView": 1, "componentType": 5126, "count": 3,
             "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": 3,
             "type": "VEC2"},
            {"bufferView": 3, "componentType": 5123, "count": 3,
             "type": "SCALAR"}
        ]
    })";
    return path;
}

} // namespace

TEST_CASE(gltf_importer_builds_geometry_materials_and_hierarchy) {
    const std::filesystem::path path = write_test_model();
    ResourceRegistry resources;
    ModelImportOptions options;
    std::string error;
    const auto model = resources.load_model(path.string(), options, error);
    REQUIRE(model != nullptr);
    REQUIRE(error.empty());
    REQUIRE(model->meshes().size() == 1);
    REQUIRE(model->nodes().size() == 1);
    REQUIRE(model->scenes().size() == 1);
    REQUIRE(model->material_names().size() == 2);
    REQUIRE(model->material_names()[0] == "Paint");

    const auto &mesh = model->meshes()[0].geometry;
    REQUIRE(mesh->vertex_count() == 3);
    REQUIRE(mesh->triangle_count() == 1);
    REQUIRE(has_mesh_attribute(mesh->triangle(0).attributes,
                               MESH_ATTRIBUTE_TANGENT));
    REQUIRE_NEAR(model->nodes()[0]
                     .local_transform.point_to_world(point3(0, 0, 0))
                     .x(),
                 2.0, 1e-12);

    MeshInstance instance(mesh, model->materials(),
                          model->nodes()[0].local_transform);
    hit_record hit;
    REQUIRE(instance.hit(ray(point3(2.2, 0.2, 1), vec3(0, 0, -1)),
                         0.001, 10.0, hit));
    REQUIRE(hit.material_id == 0);
    REQUIRE(hit.mat_ptr == model->materials()[0].get());
    std::filesystem::remove(path);
}

TEST_CASE(resource_registry_reuses_model_and_embedded_image_assets) {
    const std::filesystem::path path = write_test_model();
    ResourceRegistry resources;
    ModelImportOptions options;
    std::string error;
    const auto first = resources.load_model(path.string(), options, error);
    const auto second = resources.load_model(
        (path.parent_path() / "." / path.filename()).string(), options,
        error);
    REQUIRE(first != nullptr);
    REQUIRE(first == second);
    REQUIRE(resources.model_count() == 1);

    const std::vector<std::uint8_t> ppm = {
        'P', '6', '\n', '1', ' ', '1', '\n', '2', '5', '5', '\n',
        255, 0, 0};
    const auto image_a = resources.load_image_from_memory(
        "embedded:red", ppm.data(), ppm.size());
    const auto image_b = resources.load_image_from_memory(
        "embedded:red", nullptr, 0);
    REQUIRE(image_a == image_b);
    REQUIRE_NEAR(image_a->component(0, 0, 0), 1.0, 1e-12);
    std::filesystem::remove(path);
}
