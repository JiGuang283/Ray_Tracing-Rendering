#ifndef PACKED_BVH_H
#define PACKED_BVH_H

#include "aabb.h"
#include "compiled_scene.h"
#include "packed_types.h"

#include <cstdint>
#include <vector>

struct PackedBVHPrimitive {
    aabb bounds;
    std::uint32_t payload = 0;
};

struct PackedBVHBuildResult {
    std::vector<PackedBVHNode> nodes;
    std::vector<std::uint32_t> ordered_payloads;
    std::uint32_t max_depth = 0;
};

PackedBVHBuildResult
build_packed_bvh(std::vector<PackedBVHPrimitive> primitives);

PackedBVHNode pack_packed_bounds(const aabb &bounds);

Range32 append_packed_bvh(CompiledScene &scene,
                          const PackedBVHBuildResult &build,
                          std::uint32_t payload_base);

#endif
