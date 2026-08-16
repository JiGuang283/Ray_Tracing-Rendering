#ifndef MESH_INSTANCE_H
#define MESH_INSTANCE_H

#include "hittable.h"
#include "material.h"
#include "mesh_asset.h"
#include "transform.h"

#include <memory>
#include <vector>

class MeshInstance final : public hittable {
  public:
    MeshInstance(std::shared_ptr<const MeshAsset> asset,
                 std::vector<MaterialHandle> materials,
                 Transform object_to_world = Transform());

    bool hit(const ray &world_ray, double t_min, double t_max,
             hit_record &record) const override;
    bool hit(const ray &world_ray, double t_min, double t_max,
             hit_record &record, RNG &rng) const override;
    bool occluded(const ray &world_ray, double t_min, double t_max,
                  RNG &rng) const override;
    bool bounding_box(double time0, double time1,
                      aabb &output_box) const override;

    const std::shared_ptr<const MeshAsset> &asset() const;
    const Transform &object_to_world() const;
    const std::vector<MaterialHandle> &materials() const;

  private:
    void populate_record(const ray &world_ray,
                         const MeshIntersection &intersection,
                         hit_record &record) const;

    std::shared_ptr<const MeshAsset> m_asset;
    std::vector<MaterialHandle> m_materials;
    Transform m_object_to_world;
    aabb m_world_bounds;
};

#endif
