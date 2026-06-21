#ifndef BVH_H
#define BVH_H

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <typeinfo>
#include <vector>

#include "hittable.h"
#include "hittable_list.h"
#include "ray.h"
#include "rtweekend.h"
#include "vec3.h"

// BVH node
class bvh_node : public hittable {
  public:
    bvh_node(const hittable_list& list, double time0, double time1)
        : bvh_node(list.objects, 0, list.objects.size(), time0, time1) {}

    bvh_node(const std::vector<shared_ptr<hittable>>& src_objects,
             size_t start, size_t end, double time0, double time1);

    bool hit(const ray& r, double t_min, double t_max,
             hit_record& rec) const override;

    bool bounding_box(double time0, double time1,
                      aabb& output_box) const override;

  public:
    shared_ptr<hittable> left;
    shared_ptr<hittable> right;
    aabb box;
};

struct LinearBVHNode {
    aabb bbox;
    int second_child_offset = -1;
    int primitives_offset = -1;
    int n_primitives = 0;

    bool is_leaf() const {
        return n_primitives > 0;
    }
};

class LinearBVH : public hittable {
  public:
    LinearBVH(const hittable_list &list, double time0, double time1)
        : LinearBVH(make_shared<bvh_node>(list, time0, time1), time0, time1) {
    }

    LinearBVH(shared_ptr<bvh_node> root, double time0 = 0.0,
              double time1 = 1.0) {
        if (!root) {
            return;
        }
        nodes.reserve(128);
        primitives.reserve(128);
        flatten_node(root, time0, time1);
    }

    bool hit(const ray &r, double t_min, double t_max,
             hit_record &rec) const override;

    bool bounding_box(double /*time0*/, double /*time1*/,
                      aabb &output_box) const override {
        if (nodes.empty()) {
            return false;
        }
        output_box = nodes[0].bbox;
        return true;
    }

  private:
    std::vector<LinearBVHNode> nodes;
    std::vector<shared_ptr<hittable>> primitives;

    int flatten_node(const shared_ptr<hittable> &node, double time0,
                     double time1);
    void make_leaf(int node_index, const shared_ptr<hittable> &primitive);
    void make_leaf(int node_index, const shared_ptr<hittable> &a,
                   const shared_ptr<hittable> &b);
};

inline bool bvh_node::bounding_box(double /*time0*/, double /*time1*/,
                                   aabb& output_box) const {
    output_box = box;
    return true;
}

inline bool bvh_node::hit(const ray& r, double t_min, double t_max,
                          hit_record& rec) const {
    if (!box.hit(r, t_min, t_max)) {
        return false;
    }

    bool hit_left = left->hit(r, t_min, t_max, rec);
    bool hit_right = right->hit(r, t_min, hit_left ? rec.t : t_max, rec);
    return hit_left || hit_right;
}

inline bvh_node::bvh_node(const std::vector<shared_ptr<hittable>>& src_objects,
                          size_t start, size_t end, double time0, double time1) {
    if (end <= start) {
        throw std::runtime_error("BVH build error: empty range [start, end).");
    }

    // NOTE: Copy because we sort locally
    auto objects = src_objects;

    // Compute centroid bounds to choose split axis (SAH-lite)
    bool have_any_centroid = false;
    aabb centroid_bounds;

    for (size_t i = start; i < end; ++i) {
        aabb obj_box;
        if (!objects[i]->bounding_box(time0, time1, obj_box)) {
            // HARD-FAIL: BVH requires valid AABB for every primitive.
            // This will tell you exactly which path is missing bounding_box().
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

    // Choose axis with largest extent
    vec3 extent = centroid_bounds.max() - centroid_bounds.min();
    int axis = 0;
    if (extent.y() > extent.x() && extent.y() > extent.z()) {
        axis = 1;
    } else if (extent.z() > extent.x()) {
        axis = 2;
    }

    // Comparator: safe + deterministic + uses time0/time1
    auto comparator = [axis, time0, time1](const shared_ptr<hittable>& a,
                                          const shared_ptr<hittable>& b) {
        aabb box_a, box_b;
        bool ok_a = a->bounding_box(time0, time1, box_a);
        bool ok_b = b->bounding_box(time0, time1, box_b);

        // If either fails (shouldn't due to hard-fail earlier), provide stable order
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

    // Build node bbox from children; MUST be valid
    aabb box_left, box_right;
    if (!left->bounding_box(time0, time1, box_left)) {
        throw std::runtime_error("BVH build error: left child has no bounding box.");
    }
    if (!right->bounding_box(time0, time1, box_right)) {
        throw std::runtime_error("BVH build error: right child has no bounding box.");
    }

    box = surrounding_box(box_left, box_right);
}

inline void LinearBVH::make_leaf(int node_index,
                                 const shared_ptr<hittable> &primitive) {
    nodes[node_index].primitives_offset = static_cast<int>(primitives.size());
    nodes[node_index].n_primitives = 1;
    primitives.push_back(primitive);
}

inline void LinearBVH::make_leaf(int node_index,
                                 const shared_ptr<hittable> &a,
                                 const shared_ptr<hittable> &b) {
    nodes[node_index].primitives_offset = static_cast<int>(primitives.size());
    primitives.push_back(a);
    if (a != b) {
        primitives.push_back(b);
        nodes[node_index].n_primitives = 2;
    } else {
        nodes[node_index].n_primitives = 1;
    }
}

inline int LinearBVH::flatten_node(const shared_ptr<hittable> &node,
                                   double time0, double time1) {
    int node_index = static_cast<int>(nodes.size());
    nodes.emplace_back();

    aabb node_box;
    if (!node->bounding_box(time0, time1, node_box)) {
        throw std::runtime_error("LinearBVH build error: node has no bbox.");
    }
    nodes[node_index].bbox = node_box;

    auto bvh = std::dynamic_pointer_cast<bvh_node>(node);
    if (!bvh) {
        make_leaf(node_index, node);
        return node_index;
    }

    auto left_bvh = std::dynamic_pointer_cast<bvh_node>(bvh->left);
    auto right_bvh = std::dynamic_pointer_cast<bvh_node>(bvh->right);

    if (bvh->left == bvh->right) {
        make_leaf(node_index, bvh->left);
        return node_index;
    }

    if (!left_bvh && !right_bvh) {
        make_leaf(node_index, bvh->left, bvh->right);
        return node_index;
    }

    int left_index = flatten_node(bvh->left, time0, time1);
    if (left_index != node_index + 1) {
        throw std::runtime_error(
            "LinearBVH build error: left child is not contiguous.");
    }
    int right_index = flatten_node(bvh->right, time0, time1);
    nodes[node_index].second_child_offset = right_index;
    return node_index;
}

inline bool LinearBVH::hit(const ray &r, double t_min, double t_max,
                           hit_record &rec) const {
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
                    if (primitives[i]->hit(r, t_min, closest_so_far,
                                           temp_rec)) {
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

#endif // BVH_H
