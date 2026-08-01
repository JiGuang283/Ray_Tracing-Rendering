#include "flat_intersector.h"

#include "flat_intersector_core.h"
#include "surface_reconstruction_core.h"

PackedTraversalStatus intersect_compiled_scene_status(
    const CompiledSceneView &scene, const PackedRay &ray, PackedHit &hit,
    RNG *rng) {
    return packed_intersector::intersect_compiled_scene_core(scene, ray, hit,
                                                              rng);
}

bool intersect_compiled_scene(const CompiledSceneView &scene,
                              const PackedRay &ray, PackedHit &hit,
                              RNG *rng) {
    return intersect_compiled_scene_status(scene, ray, hit, rng) ==
           PackedTraversalStatus::Hit;
}

PackedShadingStatus reconstruct_compiled_hit_status(
    const CompiledSceneView &scene, const PackedRay &ray,
    const PackedHit &hit, PackedSurfaceInteraction &surface) {
    return packed_reconstruction::reconstruct_compiled_hit_core(
        scene, ray, hit, surface);
}

bool reconstruct_compiled_hit(const CompiledSceneView &scene,
                              const PackedRay &ray, const PackedHit &hit,
                              PackedSurfaceInteraction &surface) {
    return reconstruct_compiled_hit_status(scene, ray, hit, surface) ==
           PackedShadingStatus::Success;
}
