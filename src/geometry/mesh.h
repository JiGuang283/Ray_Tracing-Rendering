#ifndef MESH_H
#define MESH_H

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "hittable.h"
#include "material.h"

struct FlatTriangleData {
    point3 v0, v1, v2;
    vec3 edge1, edge2;
    vec3 face_normal;
    vec3 n0{0, 0, 0}, n1{0, 0, 0}, n2{0, 0, 0};
    vec2 uv0{0, 0}, uv1{0, 0}, uv2{0, 0};
    bool has_vertex_normals = false;
    bool has_texcoords = false;
    aabb bbox;
    point3 centroid;
};

struct FlatMeshBVHNode {
    aabb bbox;
    int left = -1;
    int right = -1;
    int first = 0;
    int count = 0;

    bool is_leaf() const {
        return count > 0;
    }
};

class FlatMesh : public hittable {
  public:
    FlatMesh() = default;
    FlatMesh(std::vector<FlatTriangleData> faces, shared_ptr<material> mat,
             bool build_bvh = true);

    static shared_ptr<FlatMesh>
    load_from_obj(const std::string &filename, shared_ptr<material> mat,
                  const vec3 &translation = vec3(0, 0, 0),
                  const vec3 &scale = vec3(1, 1, 1),
                  bool build_bvh = true, bool use_vertex_normals = true);

    bool hit(const ray &r, double t_min, double t_max,
             hit_record &rec) const override;
    bool hit(const ray &r, double t_min, double t_max, hit_record &rec,
             RNG &rng) const override;

    bool bounding_box(double time0, double time1,
                      aabb &output_box) const override;

  private:
    static constexpr int kLeafSize = 4;
    std::vector<FlatTriangleData> triangles;
    std::vector<int> indices;
    std::vector<FlatMeshBVHNode> nodes;
    shared_ptr<material> mat_ptr;

    int build_node(int start, int end);
    bool hit_triangle(int triangle_index, const ray &r, double t_min,
                      double t_max, hit_record &rec) const;
};

class mesh : public hittable {
  public:
    mesh() = default;

    mesh(std::vector<shared_ptr<hittable>> faces, double time0 = 0.0,
         double time1 = 1.0, bool build_bvh = true);

    static shared_ptr<mesh>
    load_from_obj(const std::string &filename, shared_ptr<material> mat,
                  const vec3 &translation = vec3(0, 0, 0),
                  const vec3 &scale = vec3(1, 1, 1),
                  bool build_bvh = true, bool use_vertex_normals = true);

    bool hit(const ray &r, double t_min, double t_max,
             hit_record &rec) const override;
    bool hit(const ray &r, double t_min, double t_max, hit_record &rec,
             RNG &rng) const override;

    bool bounding_box(double time0, double time1,
                      aabb &output_box) const override;

  private:
    std::vector<shared_ptr<hittable>> triangles;
    shared_ptr<hittable> accelerator;
};

#endif
