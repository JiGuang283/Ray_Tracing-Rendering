#ifndef FLAT_INTERSECTOR_H
#define FLAT_INTERSECTOR_H

#include "compiled_scene.h"
#include "rng.h"

PackedTraversalStatus intersect_compiled_scene_status(
    const CompiledSceneView &scene, const PackedRay &ray, PackedHit &hit,
    RNG *rng = nullptr);

bool intersect_compiled_scene(const CompiledSceneView &scene,
                              const PackedRay &ray, PackedHit &hit,
                              RNG *rng = nullptr);

bool reconstruct_compiled_hit(const CompiledSceneView &scene,
                              const PackedRay &ray, const PackedHit &hit,
                              PackedSurfaceInteraction &surface);

#endif
