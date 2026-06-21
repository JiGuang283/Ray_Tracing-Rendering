#ifndef SCENE_PROCEDURAL_LOADER_H
#define SCENE_PROCEDURAL_LOADER_H

#include "hittable.h"

namespace scene_loader_internal {

shared_ptr<hittable> build_random_scene_generator(bool emissive_variant);
shared_ptr<hittable> build_final_scene_generator(bool nee_variant);

} // namespace scene_loader_internal

#endif
