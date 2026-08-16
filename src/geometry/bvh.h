#ifndef BVH_H
#define BVH_H

#include <vector>

#include "hittable.h"
#include "hittable_list.h"
#include "ray.h"
#include "rtweekend.h"
#include "vec3.h"

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
        : LinearBVH(list.objects, time0, time1) {
    }

    LinearBVH(const std::vector<shared_ptr<hittable>> &src_objects,
              double time0, double time1) {
        build(src_objects, time0, time1);
    }

    bool hit(const ray &r, double t_min, double t_max,
             hit_record &rec) const override;
    bool hit(const ray &r, double t_min, double t_max, hit_record &rec,
             RNG &rng) const override;
    bool occluded(const ray &r, double t_min, double t_max,
                  RNG &rng) const override;

    bool bounding_box(double time0, double time1,
                      aabb &output_box) const override;

  private:
    static constexpr int kLeafSize = 4;

    struct PrimitiveInfo {
        shared_ptr<hittable> primitive;
        aabb bbox;
        point3 centroid;
    };

    std::vector<LinearBVHNode> nodes;
    std::vector<shared_ptr<hittable>> primitives;

    void build(const std::vector<shared_ptr<hittable>> &src_objects,
               double time0, double time1);
    int build_node(std::vector<PrimitiveInfo> &primitive_infos, int start,
                   int end);
};

#endif // BVH_H
