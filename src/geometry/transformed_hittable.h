#ifndef TRANSFORMED_HITTABLE_H
#define TRANSFORMED_HITTABLE_H

#include "hittable.h"
#include "transform.h"

#include <memory>

class TransformedHittable final : public hittable {
  public:
    TransformedHittable(shared_ptr<hittable> child,
                        Transform object_to_world);

    bool hit(const ray &world_ray, double t_min, double t_max,
             hit_record &record) const override;
    bool hit(const ray &world_ray, double t_min, double t_max,
             hit_record &record, RNG &rng) const override;
    bool bounding_box(double time0, double time1,
                      aabb &output_box) const override;

  private:
    void transform_record(const ray &world_ray, hit_record &record) const;

    shared_ptr<hittable> m_child;
    Transform m_object_to_world;
};

#endif
