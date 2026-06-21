#include "bvh.h"

#include <algorithm>
#include <stdexcept>

bool bvh_node::bounding_box(double /*time0*/, double /*time1*/,
                            aabb &output_box) const {
    output_box = box;
    return true;
}

bool bvh_node::hit(const ray &r, double t_min, double t_max,
                   hit_record &rec) const {
    RNG rng(make_thread_seed());
    return hit(r, t_min, t_max, rec, rng);
}

bool bvh_node::hit(const ray &r, double t_min, double t_max, hit_record &rec,
                   RNG &rng) const {
    if (!box.hit(r, t_min, t_max)) {
        return false;
    }

    bool hit_left = left->hit(r, t_min, t_max, rec, rng);
    bool hit_right =
        right->hit(r, t_min, hit_left ? rec.t : t_max, rec, rng);
    return hit_left || hit_right;
}

bvh_node::bvh_node(const std::vector<shared_ptr<hittable>> &src_objects,
                   size_t start, size_t end, double time0, double time1) {
    if (end <= start) {
        throw std::runtime_error("BVH build error: empty range [start, end).");
    }

    auto objects = src_objects;

    bool have_any_centroid = false;
    aabb centroid_bounds;

    for (size_t i = start; i < end; ++i) {
        aabb obj_box;
        if (!objects[i]->bounding_box(time0, time1, obj_box)) {
            throw std::runtime_error(
                "BVH build error: object has no bounding box. "
                "Implement bounding_box() for all hittables used in BVH.");
        }

        point3 c = 0.5 * (obj_box.min() + obj_box.max());
        aabb c_box(c, c);

        if (!have_any_centroid) {
            centroid_bounds = c_box;
            have_any_centroid = true;
        } else {
            centroid_bounds = surrounding_box(centroid_bounds, c_box);
        }
    }

    vec3 extent = centroid_bounds.max() - centroid_bounds.min();
    int axis = 0;
    if (extent.y() > extent.x() && extent.y() > extent.z()) {
        axis = 1;
    } else if (extent.z() > extent.x()) {
        axis = 2;
    }

    auto comparator = [axis, time0, time1](const shared_ptr<hittable> &a,
                                           const shared_ptr<hittable> &b) {
        aabb box_a, box_b;
        bool ok_a = a->bounding_box(time0, time1, box_a);
        bool ok_b = b->bounding_box(time0, time1, box_b);

        if (!ok_a || !ok_b) {
            return a.get() < b.get();
        }

        point3 ca = 0.5 * (box_a.min() + box_a.max());
        point3 cb = 0.5 * (box_b.min() + box_b.max());
        return ca[axis] < cb[axis];
    };

    const size_t object_span = end - start;

    if (object_span == 1) {
        left = right = objects[start];
    } else if (object_span == 2) {
        if (comparator(objects[start], objects[start + 1])) {
            left = objects[start];
            right = objects[start + 1];
        } else {
            left = objects[start + 1];
            right = objects[start];
        }
    } else {
        std::sort(objects.begin() + start, objects.begin() + end, comparator);

        const size_t mid = start + object_span / 2;
        left = make_shared<bvh_node>(objects, start, mid, time0, time1);
        right = make_shared<bvh_node>(objects, mid, end, time0, time1);
    }

    aabb box_left, box_right;
    if (!left->bounding_box(time0, time1, box_left)) {
        throw std::runtime_error(
            "BVH build error: left child has no bounding box.");
    }
    if (!right->bounding_box(time0, time1, box_right)) {
        throw std::runtime_error(
            "BVH build error: right child has no bounding box.");
    }

    box = surrounding_box(box_left, box_right);
}

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

    std::vector<int> stack;
    stack.reserve(128);
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

                if (stack.empty()) {
                    break;
                }
                current = stack.back();
                stack.pop_back();
            } else {
                stack.push_back(node.second_child_offset);
                current = current + 1;
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
