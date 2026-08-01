#include "packed_bvh.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

constexpr std::uint32_t kLeafSize = 4;
constexpr std::uint32_t kMaxDepth = 64;
constexpr int kBucketCount = 12;

struct BuildPrimitive {
    aabb bounds;
    point3 centroid;
    std::uint32_t payload = 0;
};

struct Bucket {
    aabb bounds;
    std::uint32_t count = 0;
    bool initialized = false;
};

bool finite_vec(const vec3 &value) {
    return std::isfinite(value.x()) && std::isfinite(value.y()) &&
           std::isfinite(value.z());
}

double surface_area(const aabb &bounds) {
    const vec3 extent = bounds.max() - bounds.min();
    return 2.0 * (extent.x() * extent.y() +
                  extent.y() * extent.z() + extent.z() * extent.x());
}

void expand(aabb &target, bool &initialized, const aabb &value) {
    if (!initialized) {
        target = value;
        initialized = true;
    } else {
        target = surrounding_box(target, value);
    }
}

int largest_axis(const vec3 &extent) {
    if (extent.y() > extent.x() && extent.y() > extent.z()) {
        return 1;
    }
    return extent.z() > extent.x() ? 2 : 0;
}

float outward_min(double value) {
    const float converted = static_cast<float>(value);
    if (!std::isfinite(converted)) {
        throw std::runtime_error("BVH bound is not representable as float32");
    }
    return std::nextafter(converted,
                          -std::numeric_limits<float>::infinity());
}

float outward_max(double value) {
    const float converted = static_cast<float>(value);
    if (!std::isfinite(converted)) {
        throw std::runtime_error("BVH bound is not representable as float32");
    }
    return std::nextafter(converted,
                          std::numeric_limits<float>::infinity());
}

PackedBVHNode pack_node_bounds(const aabb &bounds) {
    PackedBVHNode node;
    node.bounds_min = {outward_min(bounds.min().x()),
                       outward_min(bounds.min().y()),
                       outward_min(bounds.min().z())};
    node.bounds_max = {outward_max(bounds.max().x()),
                       outward_max(bounds.max().y()),
                       outward_max(bounds.max().z())};
    return node;
}

class Builder {
  public:
    explicit Builder(std::vector<BuildPrimitive> primitives)
        : m_primitives(std::move(primitives)) {
        m_result.nodes.reserve(m_primitives.size() * 2);
        m_result.ordered_payloads.reserve(m_primitives.size());
    }

    PackedBVHBuildResult build() {
        if (!m_primitives.empty()) {
            build_node(0, static_cast<std::uint32_t>(m_primitives.size()), 1);
        }
        return std::move(m_result);
    }

  private:
    std::uint32_t build_node(std::uint32_t begin, std::uint32_t end,
                             std::uint32_t depth) {
        if (depth > kMaxDepth) {
            throw std::runtime_error("Packed BVH exceeds traversal depth 64");
        }
        m_result.max_depth = std::max(m_result.max_depth, depth);

        bool bounds_initialized = false;
        bool centroid_initialized = false;
        aabb bounds;
        aabb centroid_bounds;
        for (std::uint32_t index = begin; index < end; ++index) {
            expand(bounds, bounds_initialized, m_primitives[index].bounds);
            const point3 centroid = m_primitives[index].centroid;
            expand(centroid_bounds, centroid_initialized,
                   aabb(centroid, centroid));
        }

        const std::uint32_t node_index =
            static_cast<std::uint32_t>(m_result.nodes.size());
        m_result.nodes.push_back(pack_node_bounds(bounds));
        const std::uint32_t count = end - begin;
        if (count <= kLeafSize) {
            PackedBVHNode &node = m_result.nodes[node_index];
            node.first = static_cast<std::uint32_t>(
                m_result.ordered_payloads.size());
            node.meta = PACKED_BVH_LEAF_BIT | count;
            for (std::uint32_t index = begin; index < end; ++index) {
                m_result.ordered_payloads.push_back(
                    m_primitives[index].payload);
            }
            return node_index;
        }

        const vec3 centroid_extent =
            centroid_bounds.max() - centroid_bounds.min();
        const int axis = largest_axis(centroid_extent);
        std::uint32_t middle = begin;
        if (centroid_extent[axis] > 1e-12) {
            middle = partition_sah(begin, end, axis, centroid_bounds, bounds);
        }
        if (middle == begin || middle == end) {
            middle = begin + count / 2;
            std::stable_sort(
                m_primitives.begin() + begin, m_primitives.begin() + end,
                [axis](const BuildPrimitive &left,
                       const BuildPrimitive &right) {
                    if (left.centroid[axis] == right.centroid[axis]) {
                        return left.payload < right.payload;
                    }
                    return left.centroid[axis] < right.centroid[axis];
                });
        }

        const std::uint32_t left = build_node(begin, middle, depth + 1);
        if (left != node_index + 1) {
            throw std::runtime_error(
                "Packed BVH preorder invariant was violated");
        }
        const std::uint32_t right = build_node(middle, end, depth + 1);
        m_result.nodes[node_index].first = right;
        m_result.nodes[node_index].meta = static_cast<std::uint32_t>(axis);
        return node_index;
    }

    std::uint32_t partition_sah(std::uint32_t begin, std::uint32_t end,
                                int axis, const aabb &centroid_bounds,
                                const aabb &parent_bounds) {
        std::array<Bucket, kBucketCount> buckets{};
        const double minimum = centroid_bounds.min()[axis];
        const double extent =
            centroid_bounds.max()[axis] - centroid_bounds.min()[axis];
        auto bucket_index = [&](const BuildPrimitive &primitive) {
            int bucket = static_cast<int>(
                kBucketCount * (primitive.centroid[axis] - minimum) / extent);
            return std::max(0, std::min(kBucketCount - 1, bucket));
        };

        for (std::uint32_t index = begin; index < end; ++index) {
            Bucket &bucket = buckets[bucket_index(m_primitives[index])];
            expand(bucket.bounds, bucket.initialized,
                   m_primitives[index].bounds);
            ++bucket.count;
        }

        std::array<double, kBucketCount - 1> costs{};
        for (int split = 0; split < kBucketCount - 1; ++split) {
            aabb left_bounds;
            aabb right_bounds;
            bool left_initialized = false;
            bool right_initialized = false;
            std::uint32_t left_count = 0;
            std::uint32_t right_count = 0;
            for (int bucket = 0; bucket <= split; ++bucket) {
                if (buckets[bucket].count > 0) {
                    expand(left_bounds, left_initialized,
                           buckets[bucket].bounds);
                    left_count += buckets[bucket].count;
                }
            }
            for (int bucket = split + 1; bucket < kBucketCount; ++bucket) {
                if (buckets[bucket].count > 0) {
                    expand(right_bounds, right_initialized,
                           buckets[bucket].bounds);
                    right_count += buckets[bucket].count;
                }
            }
            if (!left_initialized || !right_initialized) {
                costs[split] = std::numeric_limits<double>::infinity();
            } else {
                costs[split] =
                    1.0 +
                    (left_count * surface_area(left_bounds) +
                     right_count * surface_area(right_bounds)) /
                        surface_area(parent_bounds);
            }
        }

        const int best_split = static_cast<int>(
            std::min_element(costs.begin(), costs.end()) - costs.begin());
        auto middle = std::stable_partition(
            m_primitives.begin() + begin, m_primitives.begin() + end,
            [&](const BuildPrimitive &primitive) {
                return bucket_index(primitive) <= best_split;
            });
        return static_cast<std::uint32_t>(middle - m_primitives.begin());
    }

    std::vector<BuildPrimitive> m_primitives;
    PackedBVHBuildResult m_result;
};

} // namespace

PackedBVHBuildResult
build_packed_bvh(std::vector<PackedBVHPrimitive> primitives) {
    std::vector<BuildPrimitive> build_primitives;
    build_primitives.reserve(primitives.size());
    for (const PackedBVHPrimitive &primitive : primitives) {
        if (!finite_vec(primitive.bounds.min()) ||
            !finite_vec(primitive.bounds.max()) ||
            primitive.bounds.min().x() > primitive.bounds.max().x() ||
            primitive.bounds.min().y() > primitive.bounds.max().y() ||
            primitive.bounds.min().z() > primitive.bounds.max().z()) {
            throw std::runtime_error("Packed BVH received invalid bounds");
        }
        build_primitives.push_back(
            {primitive.bounds,
             0.5 * (primitive.bounds.min() + primitive.bounds.max()),
             primitive.payload});
    }
    return Builder(std::move(build_primitives)).build();
}

PackedBVHNode pack_packed_bounds(const aabb &bounds) {
    if (!finite_vec(bounds.min()) || !finite_vec(bounds.max()) ||
        bounds.min().x() > bounds.max().x() ||
        bounds.min().y() > bounds.max().y() ||
        bounds.min().z() > bounds.max().z()) {
        throw std::runtime_error("cannot pack invalid bounds");
    }
    return pack_node_bounds(bounds);
}

Range32 append_packed_bvh(CompiledScene &scene,
                          const PackedBVHBuildResult &build,
                          std::uint32_t payload_base) {
    if (scene.bvh_nodes.size() + build.nodes.size() >=
        kInvalidPackedIndex) {
        throw std::overflow_error("packed BVH node buffer exceeds 32 bits");
    }
    const std::uint32_t node_base =
        static_cast<std::uint32_t>(scene.bvh_nodes.size());
    for (PackedBVHNode node : build.nodes) {
        if (node.is_leaf()) {
            node.first += payload_base;
        } else {
            node.first += node_base;
        }
        scene.bvh_nodes.push_back(node);
    }
    return {node_base, static_cast<std::uint32_t>(build.nodes.size())};
}
