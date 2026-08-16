#include "bvh.h"

#include <algorithm>
#include <array>
#include <stdexcept>

bool LinearBVH::bounding_box(double /*time0*/, double /*time1*/,
                             aabb &output_box) const {
    if (nodes.empty()) {
        return false;
    }
    output_box = nodes[0].bbox;
    return true;
}

void LinearBVH::build(const std::vector<shared_ptr<hittable>> &src_objects,
                      double time0, double time1) {
    nodes.clear();
    primitives.clear();

    if (src_objects.empty()) {
        return;
    }

    std::vector<PrimitiveInfo> primitive_infos;
    primitive_infos.reserve(src_objects.size());

    for (const auto &primitive : src_objects) {
        aabb bbox;
        if (!primitive->bounding_box(time0, time1, bbox)) {
            throw std::runtime_error(
                "LinearBVH build error: primitive has no bounding box.");
        }
        primitive_infos.push_back(
            {primitive, bbox, 0.5 * (bbox.min() + bbox.max())});
    }

    nodes.reserve(src_objects.size() * 2);
    build_node(primitive_infos, 0, static_cast<int>(primitive_infos.size()));

    primitives.reserve(primitive_infos.size());
    for (const auto &info : primitive_infos) {
        primitives.push_back(info.primitive);
    }
}

int LinearBVH::build_node(std::vector<PrimitiveInfo> &primitive_infos,
                          int start, int end) {
    int node_index = static_cast<int>(nodes.size());
    nodes.emplace_back();

    aabb bounds = primitive_infos[start].bbox;
    aabb centroid_bounds(primitive_infos[start].centroid,
                         primitive_infos[start].centroid);

    for (int i = start + 1; i < end; ++i) {
        bounds = surrounding_box(bounds, primitive_infos[i].bbox);
        centroid_bounds = surrounding_box(
            centroid_bounds,
            aabb(primitive_infos[i].centroid, primitive_infos[i].centroid));
    }
    nodes[node_index].bbox = bounds;

    int count = end - start;
    if (count <= kLeafSize) {
        nodes[node_index].primitives_offset = start;
        nodes[node_index].n_primitives = count;
        return node_index;
    }

    vec3 extent = centroid_bounds.max() - centroid_bounds.min();
    int axis = 0;
    if (extent.y() > extent.x() && extent.y() > extent.z()) {
        axis = 1;
    } else if (extent.z() > extent.x()) {
        axis = 2;
    }

    std::sort(primitive_infos.begin() + start, primitive_infos.begin() + end,
              [axis](const PrimitiveInfo &a, const PrimitiveInfo &b) {
                  if (a.centroid[axis] == b.centroid[axis]) {
                      return a.primitive.get() < b.primitive.get();
                  }
                  return a.centroid[axis] < b.centroid[axis];
              });

    int mid = start + count / 2;
    int left_index = build_node(primitive_infos, start, mid);
    if (left_index != node_index + 1) {
        throw std::runtime_error(
            "LinearBVH build error: left child is not contiguous.");
    }
    int right_index = build_node(primitive_infos, mid, end);
    nodes[node_index].second_child_offset = right_index;
    return node_index;
}

bool LinearBVH::hit(const ray &r, double t_min, double t_max,
                    hit_record &rec) const {
    RNG rng(make_thread_seed());
    return hit(r, t_min, t_max, rec, rng);
}

bool LinearBVH::hit(const ray &r, double t_min, double t_max, hit_record &rec,
                    RNG &rng) const {
    if (nodes.empty()) {
        return false;
    }

    bool hit_anything = false;
    double closest_so_far = t_max;
    hit_record temp_rec;

    // Deliberately not value-initialized: writing 512 bytes on every ray is
    // slower than the per-ray allocation it replaces. stack_size controls
    // which slots are live.
    std::array<int, 128> stack;
    std::size_t stack_size = 0;
    int current = 0;

    while (true) {
        const LinearBVHNode &node = nodes[current];

        if (node.bbox.hit(r, t_min, closest_so_far)) {
            if (node.is_leaf()) {
                int begin = node.primitives_offset;
                int end = begin + node.n_primitives;
                for (int i = begin; i < end; ++i) {
                    if (primitives[i]->hit(r, t_min, closest_so_far, temp_rec,
                                           rng)) {
                        hit_anything = true;
                        closest_so_far = temp_rec.t;
                        rec = temp_rec;
                    }
                }

                if (stack_size == 0) {
                    break;
                }
                current = stack[--stack_size];
            } else {
                if (stack_size == stack.size()) {
                    throw std::runtime_error(
                        "LinearBVH traversal stack overflow.");
                }
                stack[stack_size++] = node.second_child_offset;
                current = current + 1;
            }
        } else {
            if (stack_size == 0) {
                break;
            }
            current = stack[--stack_size];
        }
    }

    return hit_anything;
}

bool LinearBVH::occluded(const ray &r, double t_min, double t_max,
                         RNG &rng) const {
    if (nodes.empty()) {
        return false;
    }

    double closest_so_far = t_max;
    std::array<int, 128> stack;
    std::size_t stack_size = 0;
    int current = 0;

    while (true) {
        const LinearBVHNode &node = nodes[current];
        if (!node.bbox.hit(r, t_min, closest_so_far)) {
            if (stack_size == 0) {
                return false;
            }
            current = stack[--stack_size];
            continue;
        }

        if (node.is_leaf()) {
            const int begin = node.primitives_offset;
            const int end = begin + node.n_primitives;
            for (int i = begin; i < end; ++i) {
                if (primitives[i]->occluded(r, t_min, closest_so_far, rng)) {
                    return true;
                }
            }
            if (stack_size == 0) {
                return false;
            }
            current = stack[--stack_size];
        } else {
            if (stack_size == stack.size()) {
                throw std::runtime_error(
                    "LinearBVH occlusion traversal stack overflow.");
            }
            stack[stack_size++] = node.second_child_offset;
            current = current + 1;
        }
    }
}
