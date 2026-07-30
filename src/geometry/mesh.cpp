#include "mesh.h"

#include <algorithm>
#include <iostream>
#include <utility>

#include "geometry_transform.h"
#include "tiny_obj_loader.h"

FlatMesh::FlatMesh(std::vector<FlatTriangleData> faces,
                   shared_ptr<material> mat, bool build_bvh)
    : triangles(std::move(faces)), mat_ptr(std::move(mat)) {
    indices.resize(triangles.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = static_cast<int>(i);
    }
    if (build_bvh && !indices.empty()) {
        build_node(0, static_cast<int>(indices.size()));
    }
}

bool FlatMesh::bounding_box(double /*time0*/, double /*time1*/,
                            aabb &output_box) const {
    if (triangles.empty()) {
        return false;
    }
    if (!nodes.empty()) {
        output_box = nodes[0].bbox;
        return true;
    }
    output_box = triangles[0].bbox;
    for (size_t i = 1; i < triangles.size(); ++i) {
        output_box = surrounding_box(output_box, triangles[i].bbox);
    }
    return true;
}

std::vector<TriangleSurface> FlatMesh::light_triangles() const {
    std::vector<TriangleSurface> result;
    result.reserve(triangles.size());
    for (const auto &triangle : triangles) {
        result.push_back({triangle.v0, triangle.v1, triangle.v2});
    }
    return result;
}

int FlatMesh::build_node(int start, int end) {
    int node_index = static_cast<int>(nodes.size());
    nodes.emplace_back();

    aabb bounds = triangles[indices[start]].bbox;
    aabb centroid_bounds(triangles[indices[start]].centroid,
                         triangles[indices[start]].centroid);
    for (int i = start + 1; i < end; ++i) {
        const auto &tri = triangles[indices[i]];
        bounds = surrounding_box(bounds, tri.bbox);
        centroid_bounds =
            surrounding_box(centroid_bounds, aabb(tri.centroid, tri.centroid));
    }
    nodes[node_index].bbox = bounds;

    int count = end - start;
    if (count <= kLeafSize) {
        nodes[node_index].first = start;
        nodes[node_index].count = count;
        return node_index;
    }

    vec3 extent = centroid_bounds.max() - centroid_bounds.min();
    int axis = 0;
    if (extent.y() > extent.x() && extent.y() > extent.z()) {
        axis = 1;
    } else if (extent.z() > extent.x()) {
        axis = 2;
    }

    std::sort(indices.begin() + start, indices.begin() + end,
              [&](int a, int b) {
                  return triangles[a].centroid[axis] <
                         triangles[b].centroid[axis];
              });

    int mid = start + count / 2;
    int left_index = build_node(start, mid);
    int right_index = build_node(mid, end);
    nodes[node_index].left = left_index;
    nodes[node_index].right = right_index;
    return node_index;
}

bool FlatMesh::hit_triangle(int triangle_index, const ray &r, double t_min,
                            double t_max, hit_record &rec) const {
    const auto &tri = triangles[triangle_index];
    const double eps = 1e-8;
    vec3 pvec = cross(r.direction(), tri.edge2);
    double det = dot(tri.edge1, pvec);

    if (fabs(det) < eps) {
        return false;
    }

    double inv_det = 1.0 / det;
    vec3 tvec = r.origin() - tri.v0;
    double bary_u = dot(tvec, pvec) * inv_det;
    if (bary_u < 0.0 || bary_u > 1.0) {
        return false;
    }

    vec3 qvec = cross(tvec, tri.edge1);
    double bary_v = dot(r.direction(), qvec) * inv_det;
    if (bary_v < 0.0 || bary_u + bary_v > 1.0) {
        return false;
    }

    double t = dot(tri.edge2, qvec) * inv_det;
    if (t < t_min || t > t_max) {
        return false;
    }

    rec.t = t;
    rec.p = r.at(t);
    rec.mat_ptr = mat_ptr.get();
    rec.primitive_id = triangle_index;

    double w = 1.0 - bary_u - bary_v;
    if (tri.has_texcoords) {
        rec.u = w * tri.uv0.x() + bary_u * tri.uv1.x() +
                bary_v * tri.uv2.x();
        rec.v = w * tri.uv0.y() + bary_u * tri.uv1.y() +
                bary_v * tri.uv2.y();
    } else {
        rec.u = bary_u;
        rec.v = bary_v;
    }

    vec3 shading_normal = tri.face_normal;
    if (tri.has_vertex_normals) {
        shading_normal =
            unit_vector(w * tri.n0 + bary_u * tri.n1 + bary_v * tri.n2);
    }

    rec.front_face = dot(r.direction(), tri.face_normal) < 0;
    rec.geometric_normal = rec.front_face ? tri.face_normal : -tri.face_normal;
    rec.normal = rec.front_face ? shading_normal : -shading_normal;
    rec.dpdu = tri.dpdu;
    rec.dpdv = tri.dpdv;
    return true;
}

bool FlatMesh::hit(const ray &r, double t_min, double t_max,
                   hit_record &rec) const {
    RNG rng(make_thread_seed());
    return hit(r, t_min, t_max, rec, rng);
}

bool FlatMesh::hit(const ray &r, double t_min, double t_max, hit_record &rec,
                   RNG & /*rng*/) const {
    if (triangles.empty()) {
        return false;
    }

    bool hit_anything = false;
    double closest_so_far = t_max;
    hit_record temp_rec;

    auto visit_leaf = [&](const FlatMeshBVHNode &node) {
        for (int i = node.first; i < node.first + node.count; ++i) {
            int triangle_index = indices[i];
            if (hit_triangle(triangle_index, r, t_min, closest_so_far,
                             temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
    };

    if (nodes.empty()) {
        FlatMeshBVHNode all;
        all.first = 0;
        all.count = static_cast<int>(indices.size());
        visit_leaf(all);
        return hit_anything;
    }

    std::vector<int> stack;
    stack.reserve(128);
    int current = 0;
    while (true) {
        const auto &node = nodes[current];
        if (node.bbox.hit(r, t_min, closest_so_far)) {
            if (node.is_leaf()) {
                visit_leaf(node);
                if (stack.empty()) {
                    break;
                }
                current = stack.back();
                stack.pop_back();
            } else {
                stack.push_back(node.right);
                current = node.left;
            }
        } else {
            if (stack.empty()) {
                break;
            }
            current = stack.back();
            stack.pop_back();
        }
    }

    return hit_anything;
}


shared_ptr<FlatMesh>
FlatMesh::load_from_obj(const std::string &filename, shared_ptr<material> mat,
                        const vec3 &translation, const vec3 &scale,
                        bool build_bvh, bool use_vertex_normals) {
    tinyobj::ObjReaderConfig reader_config;
    reader_config.triangulate = true;

    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(filename, reader_config)) {
        std::cerr << "[TinyObjLoader] " << reader.Error() << std::endl;
        return nullptr;
    }

    const auto &attrib = reader.GetAttrib();
    const auto &shapes = reader.GetShapes();

    std::vector<FlatTriangleData> faces;
    faces.reserve(shapes.size() * 3);

    auto make_triangle = [](const point3 &p0, const point3 &p1,
                            const point3 &p2, const vec3 &n0,
                            const vec3 &n1, const vec3 &n2,
                            bool has_normals, const vec2 &uv0,
                            const vec2 &uv1, const vec2 &uv2,
                            bool has_uvs) {
        FlatTriangleData tri;
        tri.v0 = p0;
        tri.v1 = p1;
        tri.v2 = p2;
        tri.edge1 = p1 - p0;
        tri.edge2 = p2 - p0;
        tri.face_normal = unit_vector(cross(tri.edge1, tri.edge2));
        tri.n0 = n0;
        tri.n1 = n1;
        tri.n2 = n2;
        tri.has_vertex_normals = has_normals;
        tri.uv0 = uv0;
        tri.uv1 = uv1;
        tri.uv2 = uv2;
        tri.has_texcoords = has_uvs;
        if (has_uvs) {
            vec2 duv1 = uv1 - uv0;
            vec2 duv2 = uv2 - uv0;
            double determinant =
                duv1.x() * duv2.y() - duv1.y() * duv2.x();
            if (fabs(determinant) > 1e-10) {
                double inv_det = 1.0 / determinant;
                tri.dpdu =
                    (duv2.y() * tri.edge1 - duv1.y() * tri.edge2) * inv_det;
                tri.dpdv =
                    (-duv2.x() * tri.edge1 + duv1.x() * tri.edge2) * inv_det;
            } else {
                tri.dpdu = tri.edge1;
                tri.dpdv = tri.edge2;
            }
        } else {
            tri.dpdu = tri.edge1;
            tri.dpdv = tri.edge2;
        }

        double min_x = fmin(p0.x(), fmin(p1.x(), p2.x()));
        double min_y = fmin(p0.y(), fmin(p1.y(), p2.y()));
        double min_z = fmin(p0.z(), fmin(p1.z(), p2.z()));
        double max_x = fmax(p0.x(), fmax(p1.x(), p2.x()));
        double max_y = fmax(p0.y(), fmax(p1.y(), p2.y()));
        double max_z = fmax(p0.z(), fmax(p1.z(), p2.z()));
        const double padding = 1e-4;
        tri.bbox =
            aabb(point3(min_x - padding, min_y - padding, min_z - padding),
                 point3(max_x + padding, max_y + padding, max_z + padding));
        tri.centroid = (p0 + p1 + p2) / 3.0;
        return tri;
    };

    for (const auto &shape : shapes) {
        size_t index_offset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
            int fv = shape.mesh.num_face_vertices[f];
            if (fv != 3) {
                index_offset += fv;
                continue;
            }

            tinyobj::index_t idx0 = shape.mesh.indices[index_offset + 0];
            tinyobj::index_t idx1 = shape.mesh.indices[index_offset + 1];
            tinyobj::index_t idx2 = shape.mesh.indices[index_offset + 2];
            index_offset += 3;

            if (idx0.vertex_index < 0 || idx1.vertex_index < 0 ||
                idx2.vertex_index < 0) {
                continue;
            }

            auto fetch_vertex = [&](const tinyobj::index_t &idx) {
                if (idx.vertex_index < 0 ||
                    attrib.vertices.size() <=
                        static_cast<size_t>(3 * idx.vertex_index + 2)) {
                    return point3(0, 0, 0);
                }
                size_t v_base = static_cast<size_t>(3 * idx.vertex_index);
                point3 p(attrib.vertices[v_base + 0],
                         attrib.vertices[v_base + 1],
                         attrib.vertices[v_base + 2]);
                p[0] = p.x() * scale.x() + translation.x();
                p[1] = p.y() * scale.y() + translation.y();
                p[2] = p.z() * scale.z() + translation.z();
                return p;
            };

            auto fetch_normal = [&](const tinyobj::index_t &idx) {
                if (idx.normal_index < 0 ||
                    attrib.normals.size() <=
                        static_cast<size_t>(3 * idx.normal_index + 2)) {
                    return vec3(0, 0, 0);
                }
                size_t n_base = static_cast<size_t>(3 * idx.normal_index);
                return transform_normal_by_inverse_scale(
                    vec3(attrib.normals[n_base + 0],
                         attrib.normals[n_base + 1],
                         attrib.normals[n_base + 2]),
                    scale);
            };

            auto fetch_texcoord = [&](const tinyobj::index_t &idx,
                                      bool &valid) {
                if (idx.texcoord_index < 0 ||
                    attrib.texcoords.size() <=
                        static_cast<size_t>(2 * idx.texcoord_index + 1)) {
                    valid = false;
                    return vec2(0, 0);
                }
                valid = true;
                size_t t_base = static_cast<size_t>(2 * idx.texcoord_index);
                return vec2(attrib.texcoords[t_base + 0],
                            attrib.texcoords[t_base + 1]);
            };

            point3 p0 = fetch_vertex(idx0);
            point3 p1 = fetch_vertex(idx1);
            point3 p2 = fetch_vertex(idx2);

            vec3 n0 = fetch_normal(idx0);
            vec3 n1 = fetch_normal(idx1);
            vec3 n2 = fetch_normal(idx2);

            bool uv0_valid = false, uv1_valid = false, uv2_valid = false;
            vec2 uv0 = fetch_texcoord(idx0, uv0_valid);
            vec2 uv1 = fetch_texcoord(idx1, uv1_valid);
            vec2 uv2 = fetch_texcoord(idx2, uv2_valid);

            bool has_uvs = uv0_valid && uv1_valid && uv2_valid;
            bool has_normals = use_vertex_normals && n0.length_squared() > 0 &&
                               n1.length_squared() > 0 &&
                               n2.length_squared() > 0;

            if (has_normals) {
                n0 = unit_vector(n0);
                n1 = unit_vector(n1);
                n2 = unit_vector(n2);
            }

            faces.push_back(make_triangle(p0, p1, p2, n0, n1, n2,
                                          has_normals, uv0, uv1, uv2,
                                          has_uvs));
        }
    }

    if (faces.empty()) {
        std::cerr << "[TinyObjLoader] No valid triangles parsed from "
                  << filename << std::endl;
        return nullptr;
    }

    return make_shared<FlatMesh>(faces, mat, build_bvh);
}
