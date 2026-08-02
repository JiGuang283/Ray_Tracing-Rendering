#ifndef TRIANGLE_SCALE_FIXTURE_H
#define TRIANGLE_SCALE_FIXTURE_H

#include "material_programs.h"
#include "mesh_instance.h"
#include "render_data/compiled_scene.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace triangle_scale_fixture {

struct ExpectedHit {
    bool found = false;
    float t = 0.0f;
    float barycentric_u = 0.0f;
    float barycentric_v = 0.0f;
};

struct Fixture {
    std::shared_ptr<const MeshInstance> reference;
    CompiledScene packed;
    std::vector<PackedRay> rays;
    std::vector<ExpectedHit> expected;
};

inline PackedRay make_ray(Float3 origin, Float3 direction, float t_min,
                          float t_max) {
    PackedRay ray;
    ray.origin = origin;
    ray.direction = direction;
    ray.t_min = t_min;
    ray.t_max = t_max;
    return ray;
}

inline Fixture make_fixture() {
    // Exact powers of two keep the shared edge identical in float and double.
    constexpr float local_scale = 1.0f / 8192.0f;
    constexpr float instance_scale = 8192.0f;

    std::vector<MeshVertex> reference_vertices(3);
    reference_vertices[0].position = point3(0, 0, 0);
    reference_vertices[1].position = point3(local_scale, 0, 0);
    reference_vertices[2].position = point3(0, local_scale, 0);
    MeshTriangle reference_triangle;
    reference_triangle.vertices[0] = 0;
    reference_triangle.vertices[1] = 1;
    reference_triangle.vertices[2] = 2;
    const auto reference_asset = std::make_shared<MeshAsset>(
        std::move(reference_vertices),
        std::vector<MeshTriangle>{reference_triangle});
    const MaterialHandle reference_material =
        make_lambertian_material(color(0.5, 0.5, 0.5));

    Fixture fixture;
    fixture.reference = std::make_shared<MeshInstance>(
        reference_asset, std::vector<MaterialHandle>{reference_material},
        Transform::scale(
            vec3(instance_scale, instance_scale, instance_scale)));

    CompiledScene &scene = fixture.packed;
    scene.positions = {{0, 0, 0, 1},
                       {local_scale, 0, 0, 1},
                       {0, local_scale, 0, 1}};
    scene.normals.assign(3, Float4{0, 0, 1, 0});
    scene.tangents.assign(3, Float4{1, 0, 0, 1});
    scene.uv0.assign(3, Float2{});
    scene.vertex_colors.assign(3, Float4{1, 1, 1, 1});

    PackedTriangle triangle;
    triangle.vertex0 = 0;
    triangle.vertex1 = 1;
    triangle.vertex2 = 2;
    triangle.material_slot = 0;
    triangle.primitive_id = 37;
    scene.triangles.push_back(triangle);

    PackedBVHNode mesh_node;
    mesh_node.bounds_min = {0, 0, -1e-8f};
    mesh_node.bounds_max = {local_scale, local_scale, 1e-8f};
    mesh_node.first = 0;
    mesh_node.meta = PACKED_BVH_LEAF_BIT | 1u;
    scene.bvh_nodes.push_back(mesh_node);

    PackedMesh mesh;
    mesh.vertices = {0, 3};
    mesh.triangles = {0, 1};
    mesh.bvh_nodes = {0, 1};
    mesh.material_slot_count = 1;
    mesh.bounds_min = mesh_node.bounds_min;
    mesh.bounds_max = mesh_node.bounds_max;
    scene.meshes.push_back(mesh);

    PackedTransform transform;
    transform.object_to_world[0] = instance_scale;
    transform.object_to_world[5] = instance_scale;
    transform.object_to_world[10] = instance_scale;
    transform.world_to_object[0] = local_scale;
    transform.world_to_object[5] = local_scale;
    transform.world_to_object[10] = local_scale;
    scene.transforms.push_back(transform);

    scene.materials.push_back(PackedMaterial{});
    scene.material_bindings.push_back(0);
    scene.emitter_bindings.push_back(kInvalidPackedIndex);

    PackedInstance instance;
    instance.geometry_type = PackedGeometryType::Mesh;
    instance.geometry_index = 0;
    instance.transform_id = 0;
    instance.material_bindings = {0, 1};
    instance.source_object_id = 19;
    instance.bounds_min = {0, 0, -1e-4f};
    instance.bounds_max = {1, 1, 1e-4f};
    scene.instances.push_back(instance);
    scene.aggregate_instance_indices.push_back(0);

    PackedBVHNode aggregate_node;
    aggregate_node.bounds_min = instance.bounds_min;
    aggregate_node.bounds_max = instance.bounds_max;
    aggregate_node.first = 0;
    aggregate_node.meta = PACKED_BVH_LEAF_BIT | 1u;
    scene.bvh_nodes.push_back(aggregate_node);
    PackedAggregate aggregate;
    aggregate.bvh_nodes = {1, 1};
    aggregate.instance_indices = {0, 1};
    scene.aggregates.push_back(aggregate);

    fixture.rays = {
        make_ray({0.25f, 0.25f, 1.0f}, {0, 0, -1.0f}, 0.0f, 2.0f),
        make_ray({0.2f, 0.3f, 1.0f}, {0, 0, -1000.0f}, 0.0f, 2.0f),
        make_ray({0.2f, 0.3f, 1.0f}, {0, 0, -0.001f}, 0.0f, 2000.0f),
        make_ray({0.5f, 0.5f, 1.0f}, {0, 0, -1.0f}, 0.0f, 2.0f),
        make_ray({0.0f, 0.0f, -1.0f}, {0, 0, 1.0f}, 0.0f, 2.0f),
        make_ray({0.8f, 0.4f, 1.0f}, {0, 0, -1.0f}, 0.0f, 2.0f)};
    fixture.expected = {{true, 1.0f, 0.25f, 0.25f},
                        {true, 0.001f, 0.2f, 0.3f},
                        {true, 1000.0f, 0.2f, 0.3f},
                        {true, 1.0f, 0.5f, 0.5f},
                        {true, 1.0f, 0.0f, 0.0f},
                        {false, 0.0f, 0.0f, 0.0f}};
    return fixture;
}

} // namespace triangle_scale_fixture

#endif
