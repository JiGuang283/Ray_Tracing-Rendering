#include "gltf_importer.h"

#include "cgltf/cgltf.h"
#include "material_programs.h"
#include "mikktspace.h"
#include "resource_registry.h"
#include "texture.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using TriangleIndices = std::array<std::uint32_t, 3>;

struct CgltfDeleter {
    void operator()(cgltf_data *data) const {
        cgltf_free(data);
    }
};

using CgltfData = std::unique_ptr<cgltf_data, CgltfDeleter>;

std::string result_name(cgltf_result result) {
    switch (result) {
    case cgltf_result_success:
        return "success";
    case cgltf_result_data_too_short:
        return "data too short";
    case cgltf_result_unknown_format:
        return "unknown format";
    case cgltf_result_invalid_json:
        return "invalid JSON";
    case cgltf_result_invalid_gltf:
        return "invalid glTF";
    case cgltf_result_invalid_options:
        return "invalid options";
    case cgltf_result_file_not_found:
        return "file not found";
    case cgltf_result_io_error:
        return "I/O error";
    case cgltf_result_out_of_memory:
        return "out of memory";
    case cgltf_result_legacy_gltf:
        return "legacy glTF 1.x";
    case cgltf_result_max_enum:
        break;
    }
    return "unknown cgltf error";
}

int hex_value(char value) {
    if (value >= '0' && value <= '9') {
        return value - '0';
    }
    value = static_cast<char>(std::tolower(static_cast<unsigned char>(value)));
    if (value >= 'a' && value <= 'f') {
        return value - 'a' + 10;
    }
    return -1;
}

std::string decode_uri_component(const std::string &value) {
    std::string result;
    result.reserve(value.size());
    for (std::size_t index = 0; index < value.size(); ++index) {
        if (value[index] == '%' && index + 2 < value.size()) {
            const int high = hex_value(value[index + 1]);
            const int low = hex_value(value[index + 2]);
            if (high >= 0 && low >= 0) {
                result.push_back(static_cast<char>((high << 4) | low));
                index += 2;
                continue;
            }
        }
        result.push_back(value[index]);
    }
    return result;
}

std::vector<std::uint8_t> decode_base64(const std::string &encoded) {
    static constexpr signed char table[256] = {
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 62, -1, -1, -1, 63,
        52, 53, 54, 55, 56, 57, 58, 59, 60, 61, -1, -1, -1, -2, -1, -1,
        -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
        15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, -1, -1, -1, -1, -1,
        -1, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, -1, -1, -1, -1, -1,
    };

    std::vector<std::uint8_t> decoded;
    decoded.reserve(encoded.size() * 3 / 4);
    std::uint32_t accumulator = 0;
    int bits = 0;
    for (unsigned char character : encoded) {
        if (std::isspace(character)) {
            continue;
        }
        if (character >= 128) {
            throw std::runtime_error("invalid base64 image data");
        }
        const signed char value = table[character];
        if (value == -2) {
            break;
        }
        if (value < 0) {
            throw std::runtime_error("invalid base64 image data");
        }
        accumulator = (accumulator << 6) | static_cast<unsigned>(value);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            decoded.push_back(static_cast<std::uint8_t>(
                (accumulator >> bits) & 0xffu));
        }
    }
    return decoded;
}

std::vector<std::uint8_t> decode_data_uri(const std::string &uri) {
    const std::size_t comma = uri.find(',');
    if (comma == std::string::npos) {
        throw std::runtime_error("malformed image data URI");
    }
    const std::string metadata = uri.substr(0, comma);
    const std::string payload = uri.substr(comma + 1);
    if (metadata.find(";base64") != std::string::npos) {
        return decode_base64(payload);
    }
    const std::string decoded = decode_uri_component(payload);
    return std::vector<std::uint8_t>(decoded.begin(), decoded.end());
}

std::string object_name(const char *name, const std::string &fallback) {
    return name && *name ? std::string(name) : fallback;
}

void require_accessor(const cgltf_accessor *accessor, cgltf_size count,
                      const std::string &attribute,
                      const std::string &context) {
    if (!accessor) {
        throw std::runtime_error(context + " is missing " + attribute + ".");
    }
    if (accessor->count != count) {
        throw std::runtime_error(context + " has a mismatched " + attribute +
                                 " accessor count.");
    }
}

std::array<float, 4> read_attribute(const cgltf_accessor *accessor,
                                    cgltf_size index,
                                    const std::string &context) {
    std::array<float, 4> value{0, 0, 0, 1};
    if (!cgltf_accessor_read_float(accessor, index, value.data(),
                                   value.size())) {
        throw std::runtime_error("Failed to read " + context + ".");
    }
    for (float component : value) {
        if (!std::isfinite(component)) {
            throw std::runtime_error("Non-finite value in " + context + ".");
        }
    }
    return value;
}

std::vector<std::uint32_t>
read_indices(const cgltf_primitive &primitive, std::size_t vertex_count,
             const std::string &context) {
    if (primitive.indices) {
        const cgltf_component_type component =
            primitive.indices->component_type;
        if (primitive.indices->type != cgltf_type_scalar ||
            (component != cgltf_component_type_r_8u &&
             component != cgltf_component_type_r_16u &&
             component != cgltf_component_type_r_32u)) {
            throw std::runtime_error(
                context + " has an invalid index accessor type.");
        }
    }
    const std::size_t count = primitive.indices
                                  ? primitive.indices->count
                                  : vertex_count;
    std::vector<std::uint32_t> result;
    result.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        const std::size_t value = primitive.indices
                                      ? cgltf_accessor_read_index(
                                            primitive.indices, index)
                                      : index;
        if (value >= vertex_count) {
            throw std::runtime_error(context +
                                     " contains an out-of-range index.");
        }
        result.push_back(static_cast<std::uint32_t>(value));
    }
    return result;
}

std::vector<TriangleIndices>
triangulate(const cgltf_primitive &primitive, std::size_t vertex_count,
            const std::vector<MeshVertex> &vertices,
            const std::string &context) {
    const std::vector<std::uint32_t> indices =
        read_indices(primitive, vertex_count, context);
    std::vector<TriangleIndices> triangles;

    auto append = [&](std::uint32_t a, std::uint32_t b, std::uint32_t c) {
        if (a == b || b == c || a == c) {
            return;
        }
        const vec3 cross_product =
            cross(vertices[b].position - vertices[a].position,
                  vertices[c].position - vertices[a].position);
        if (cross_product.length_squared() <= 1e-24) {
            return;
        }
        triangles.push_back({a, b, c});
    };

    if (primitive.type == cgltf_primitive_type_triangles) {
        if (indices.size() % 3 != 0) {
            throw std::runtime_error(context +
                                     " triangle index count is not divisible by 3.");
        }
        for (std::size_t index = 0; index < indices.size(); index += 3) {
            append(indices[index], indices[index + 1], indices[index + 2]);
        }
    } else if (primitive.type == cgltf_primitive_type_triangle_strip) {
        for (std::size_t index = 2; index < indices.size(); ++index) {
            if ((index & 1u) == 0) {
                append(indices[index - 2], indices[index - 1], indices[index]);
            } else {
                append(indices[index - 1], indices[index - 2], indices[index]);
            }
        }
    } else if (primitive.type == cgltf_primitive_type_triangle_fan) {
        for (std::size_t index = 2; index < indices.size(); ++index) {
            append(indices[0], indices[index - 1], indices[index]);
        }
    } else {
        throw std::runtime_error(context +
                                 " uses a non-triangle primitive mode.");
    }
    return triangles;
}

void generate_normals(std::vector<MeshVertex> &vertices,
                      const std::vector<TriangleIndices> &triangles) {
    std::vector<vec3> sums(vertices.size(), vec3(0, 0, 0));
    for (const TriangleIndices &triangle : triangles) {
        const vec3 normal =
            cross(vertices[triangle[1]].position -
                      vertices[triangle[0]].position,
                  vertices[triangle[2]].position -
                      vertices[triangle[0]].position);
        for (std::uint32_t vertex : triangle) {
            sums[vertex] += normal;
        }
    }
    for (std::size_t index = 0; index < vertices.size(); ++index) {
        vertices[index].normal = sums[index].near_zero()
                                     ? vec3(0, 0, 1)
                                     : unit_vector(sums[index]);
    }
}

struct MikkUserData {
    std::vector<MeshVertex> *vertices = nullptr;
    const std::vector<TriangleIndices> *triangles = nullptr;
    std::vector<vec3> tangent_sums;
    std::vector<double> sign_sums;
    std::vector<std::uint32_t> counts;
};

const MeshVertex &mikk_vertex(const SMikkTSpaceContext *context, int face,
                             int corner) {
    const auto *data = static_cast<const MikkUserData *>(context->m_pUserData);
    const std::uint32_t index =
        (*data->triangles)[static_cast<std::size_t>(face)]
                          [static_cast<std::size_t>(corner)];
    return (*data->vertices)[index];
}

int mikk_face_count(const SMikkTSpaceContext *context) {
    const auto *data = static_cast<const MikkUserData *>(context->m_pUserData);
    return static_cast<int>(data->triangles->size());
}

int mikk_vertex_count(const SMikkTSpaceContext *, int) {
    return 3;
}

void mikk_position(const SMikkTSpaceContext *context, float output[], int face,
                   int corner) {
    const point3 &position = mikk_vertex(context, face, corner).position;
    output[0] = static_cast<float>(position.x());
    output[1] = static_cast<float>(position.y());
    output[2] = static_cast<float>(position.z());
}

void mikk_normal(const SMikkTSpaceContext *context, float output[], int face,
                 int corner) {
    const vec3 &normal = mikk_vertex(context, face, corner).normal;
    output[0] = static_cast<float>(normal.x());
    output[1] = static_cast<float>(normal.y());
    output[2] = static_cast<float>(normal.z());
}

void mikk_uv(const SMikkTSpaceContext *context, float output[], int face,
             int corner) {
    const vec2 &uv = mikk_vertex(context, face, corner).uv0;
    output[0] = static_cast<float>(uv.x());
    output[1] = static_cast<float>(uv.y());
}

void mikk_tangent(const SMikkTSpaceContext *context, const float tangent[],
                  float sign, int face, int corner) {
    auto *data = static_cast<MikkUserData *>(context->m_pUserData);
    const std::uint32_t index =
        (*data->triangles)[static_cast<std::size_t>(face)]
                          [static_cast<std::size_t>(corner)];
    data->tangent_sums[index] +=
        vec3(tangent[0], tangent[1], tangent[2]);
    data->sign_sums[index] += sign;
    ++data->counts[index];
}

bool generate_tangents(std::vector<MeshVertex> &vertices,
                       const std::vector<TriangleIndices> &triangles) {
    if (triangles.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return false;
    }
    MikkUserData user;
    user.vertices = &vertices;
    user.triangles = &triangles;
    user.tangent_sums.assign(vertices.size(), vec3(0, 0, 0));
    user.sign_sums.assign(vertices.size(), 0.0);
    user.counts.assign(vertices.size(), 0);

    SMikkTSpaceInterface callbacks{};
    callbacks.m_getNumFaces = mikk_face_count;
    callbacks.m_getNumVerticesOfFace = mikk_vertex_count;
    callbacks.m_getPosition = mikk_position;
    callbacks.m_getNormal = mikk_normal;
    callbacks.m_getTexCoord = mikk_uv;
    callbacks.m_setTSpaceBasic = mikk_tangent;
    SMikkTSpaceContext context{&callbacks, &user};
    if (!genTangSpaceDefault(&context)) {
        return false;
    }

    for (std::size_t index = 0; index < vertices.size(); ++index) {
        vec3 tangent = user.tangent_sums[index];
        tangent = tangent - dot(tangent, vertices[index].normal) *
                                vertices[index].normal;
        if (user.counts[index] == 0 || tangent.near_zero()) {
            return false;
        }
        vertices[index].tangent = unit_vector(tangent);
        vertices[index].tangent_sign =
            user.sign_sums[index] < 0.0 ? -1.0 : 1.0;
    }
    return true;
}

SamplerState sampler_state(const cgltf_texture *texture) {
    SamplerState state;
    state.flip_v = false;
    if (!texture || !texture->sampler) {
        return state;
    }
    const cgltf_sampler &sampler = *texture->sampler;
    auto wrap = [](cgltf_wrap_mode mode) {
        if (mode == cgltf_wrap_mode_clamp_to_edge) {
            return WrapMode::Clamp;
        }
        if (mode == cgltf_wrap_mode_mirrored_repeat) {
            return WrapMode::Mirror;
        }
        return WrapMode::Repeat;
    };
    state.wrap_u = wrap(sampler.wrap_s);
    state.wrap_v = wrap(sampler.wrap_t);
    if (sampler.mag_filter == cgltf_filter_type_nearest ||
        sampler.min_filter == cgltf_filter_type_nearest ||
        sampler.min_filter == cgltf_filter_type_nearest_mipmap_nearest ||
        sampler.min_filter == cgltf_filter_type_nearest_mipmap_linear) {
        state.filter = FilterMode::Nearest;
    }
    return state;
}

class ImportContext {
  public:
    ImportContext(const std::string &filename,
                  const ModelImportOptions &options,
                  ResourceRegistry &resources, const cgltf_data &data)
        : m_filename(filename), m_options(options), m_resources(resources),
          m_data(data), m_default_material_slot(data.materials_count) {
    }

    std::shared_ptr<const ModelAsset> build() {
        validate_required_extensions();
        reject_unsupported_compression();
        if (m_data.animations_count > 0) {
            std::cerr << "WARNING: glTF animations are ignored by the static "
                         "model importer.\n";
        }
        build_materials();
        build_meshes();
        build_nodes();
        build_scenes();
        if (m_meshes.empty()) {
            throw std::runtime_error("glTF contains no renderable meshes.");
        }
        return std::make_shared<ModelAsset>(
            std::move(m_meshes), std::move(m_nodes), std::move(m_scenes),
            m_default_scene, std::move(m_material_names),
            std::move(m_materials));
    }

  private:
    void validate_required_extensions() const {
        static const std::array<const char *, 3> supported = {
            "KHR_texture_transform", "KHR_materials_clearcoat",
            "KHR_materials_emissive_strength"};
        for (cgltf_size index = 0;
             index < m_data.extensions_required_count; ++index) {
            const std::string required = m_data.extensions_required[index];
            const bool found =
                std::find_if(supported.begin(), supported.end(),
                             [&](const char *value) {
                                 return required == value;
                             }) != supported.end();
            if (!found) {
                throw std::runtime_error(
                    "glTF requires unsupported extension '" + required +
                    "'.");
            }
        }
    }

    void reject_unsupported_compression() const {
        for (cgltf_size index = 0; index < m_data.buffer_views_count;
             ++index) {
            if (m_data.buffer_views[index].has_meshopt_compression) {
                throw std::runtime_error(
                    "EXT_meshopt_compression is not supported.");
            }
        }
        for (cgltf_size mesh_index = 0; mesh_index < m_data.meshes_count;
             ++mesh_index) {
            const cgltf_mesh &mesh = m_data.meshes[mesh_index];
            for (cgltf_size primitive_index = 0;
                 primitive_index < mesh.primitives_count;
                 ++primitive_index) {
                if (mesh.primitives[primitive_index]
                        .has_draco_mesh_compression) {
                    throw std::runtime_error(
                        "KHR_draco_mesh_compression is not supported.");
                }
            }
        }
    }

    std::shared_ptr<const ImageAsset>
    load_image(const cgltf_texture *texture) {
        if (!texture) {
            return nullptr;
        }
        const cgltf_image *image = texture->image;
        if (!image && texture->has_webp) {
            image = texture->webp_image;
        }
        if (!image && texture->has_basisu) {
            image = texture->basisu_image;
        }
        if (!image) {
            return nullptr;
        }

        const std::size_t image_index = cgltf_image_index(&m_data, image);
        const std::string resource_key =
            m_filename + "#image=" + std::to_string(image_index);
        if (image->buffer_view) {
            const std::uint8_t *bytes =
                cgltf_buffer_view_data(image->buffer_view);
            if (!bytes) {
                throw std::runtime_error("Embedded glTF image has no data.");
            }
            return m_resources.load_image_from_memory(
                resource_key, bytes, image->buffer_view->size);
        }
        if (!image->uri) {
            throw std::runtime_error("glTF image has no URI or buffer view.");
        }
        const std::string uri(image->uri);
        if (uri.rfind("data:", 0) == 0) {
            const std::vector<std::uint8_t> decoded = decode_data_uri(uri);
            return m_resources.load_image_from_memory(
                resource_key, decoded.data(), decoded.size());
        }
        if (uri.find("://") != std::string::npos) {
            throw std::runtime_error(
                "Remote glTF image URIs are not supported: " + uri);
        }
        const std::filesystem::path image_path =
            std::filesystem::path(m_filename).parent_path() /
            decode_uri_component(uri);
        return m_resources.load_image(image_path.string());
    }

    TextureHandle texture_view(const cgltf_texture_view &view,
                               ColorSpace color_space,
                               TextureChannel channel) {
        if (!view.texture) {
            return nullptr;
        }
        const int texcoord = view.has_transform &&
                                     view.transform.has_texcoord
                                 ? view.transform.texcoord
                                 : view.texcoord;
        if (texcoord != 0) {
            throw std::runtime_error(
                "Only glTF TEXCOORD_0 is currently supported.");
        }
        TextureHandle result = std::make_shared<ImageTexture>(
            load_image(view.texture), color_space,
            sampler_state(view.texture), channel);
        if (view.has_transform) {
            result = std::make_shared<UVTransformTexture>(
                result,
                vec2(view.transform.offset[0], view.transform.offset[1]),
                vec2(view.transform.scale[0], view.transform.scale[1]),
                view.transform.rotation);
        }
        return result;
    }

    static TextureHandle solid(const color &value) {
        return std::make_shared<SolidColorTexture>(value);
    }

    static TextureHandle scaled(TextureHandle texture, double factor) {
        if (!texture) {
            return solid(color(factor, factor, factor));
        }
        return std::make_shared<ScaleTexture>(std::move(texture), factor);
    }

    static TextureHandle multiplied(TextureHandle texture,
                                    const color &factor) {
        TextureHandle factor_texture = solid(factor);
        if (!texture) {
            return factor_texture;
        }
        return std::make_shared<MultiplyTexture>(std::move(texture),
                                                 std::move(factor_texture));
    }

    MaterialHandle build_material(const cgltf_material &material,
                                  std::size_t index) {
        if (material.alpha_mode != cgltf_alpha_mode_opaque) {
            std::cerr << "WARNING: glTF material '"
                      << object_name(material.name,
                                     "material_" + std::to_string(index))
                      << "' uses alpha; it is rendered opaque.\n";
        }
        if (material.has_transmission || material.has_volume) {
            std::cerr << "WARNING: glTF transmission/volume extensions are "
                         "not yet represented by the current Principled BSDF.\n";
        }
        if (material.has_sheen || material.has_specular ||
            material.has_iridescence || material.has_anisotropy ||
            material.has_dispersion ||
            material.has_pbr_specular_glossiness) {
            std::cerr << "WARNING: glTF material '"
                      << object_name(material.name,
                                     "material_" + std::to_string(index))
                      << "' contains unsupported shading extensions; core "
                         "metallic-roughness inputs will be used.\n";
        }
        if (material.unlit) {
            std::cerr << "WARNING: glTF unlit material '"
                      << object_name(material.name,
                                     "material_" + std::to_string(index))
                      << "' is approximated as Lambertian.\n";
        }

        const cgltf_pbr_metallic_roughness &pbr =
            material.pbr_metallic_roughness;
        TextureHandle base = texture_view(
            pbr.base_color_texture, ColorSpace::SRGB, TextureChannel::RGB);
        base = multiplied(
            std::move(base),
            color(pbr.base_color_factor[0], pbr.base_color_factor[1],
                  pbr.base_color_factor[2]));
        base = std::make_shared<MultiplyTexture>(
            std::move(base), std::make_shared<VertexColorTexture>());

        TextureHandle roughness = texture_view(
            pbr.metallic_roughness_texture, ColorSpace::Linear,
            TextureChannel::G);
        roughness = scaled(std::move(roughness), pbr.roughness_factor);
        TextureHandle metallic = texture_view(
            pbr.metallic_roughness_texture, ColorSpace::Linear,
            TextureChannel::B);
        metallic = scaled(std::move(metallic), pbr.metallic_factor);

        TextureHandle normal = texture_view(
            material.normal_texture, ColorSpace::Linear,
            TextureChannel::RGB);
        NormalMapSettings normal_settings;
        normal_settings.strength = material.normal_texture.texture
                                       ? material.normal_texture.scale
                                       : 1.0;

        const color emissive_factor(material.emissive_factor[0],
                                    material.emissive_factor[1],
                                    material.emissive_factor[2]);
        TextureHandle emission;
        if (emissive_factor.length_squared() > 0.0) {
            emission = multiplied(
                texture_view(material.emissive_texture, ColorSpace::SRGB,
                             TextureChannel::RGB),
                emissive_factor);
        }
        const double emission_strength =
            material.has_emissive_strength
                ? material.emissive_strength.emissive_strength
                : 1.0;

        TextureHandle clearcoat;
        TextureHandle clearcoat_roughness;
        if (material.has_clearcoat) {
            clearcoat = scaled(
                texture_view(material.clearcoat.clearcoat_texture,
                             ColorSpace::Linear, TextureChannel::R),
                material.clearcoat.clearcoat_factor);
            clearcoat_roughness = scaled(
                texture_view(
                    material.clearcoat.clearcoat_roughness_texture,
                    ColorSpace::Linear, TextureChannel::G),
                material.clearcoat.clearcoat_roughness_factor);
        }

        if (material.unlit) {
            return make_lambertian_material(std::move(base));
        }
        return make_principled_material(
            std::move(base), std::move(roughness), std::move(metallic),
            std::move(normal), std::move(emission), emission_strength,
            std::move(clearcoat), std::move(clearcoat_roughness),
            normal_settings, material.double_sided != 0);
    }

    void build_materials() {
        m_material_names.reserve(m_data.materials_count + 1);
        m_materials.reserve(m_data.materials_count + 1);
        for (cgltf_size index = 0; index < m_data.materials_count; ++index) {
            const cgltf_material &material = m_data.materials[index];
            m_material_names.push_back(object_name(
                material.name, "material_" + std::to_string(index)));
            m_materials.push_back(build_material(material, index));
        }
        m_material_names.push_back("__default");
        TextureHandle base = std::make_shared<MultiplyTexture>(
            solid(color(0.8, 0.8, 0.8)),
            std::make_shared<VertexColorTexture>());
        m_materials.push_back(make_principled_material(
            std::move(base), solid(color(1, 1, 1)), solid(color(0, 0, 0))));
    }

    std::shared_ptr<const MeshAsset>
    build_mesh(const cgltf_mesh &mesh, std::size_t mesh_index) {
        std::vector<MeshVertex> vertices;
        std::vector<MeshTriangle> triangles;
        std::vector<MeshPrimitive> primitives;

        for (cgltf_size primitive_index = 0;
             primitive_index < mesh.primitives_count; ++primitive_index) {
            const cgltf_primitive &primitive =
                mesh.primitives[primitive_index];
            const std::string context =
                "mesh '" +
                object_name(mesh.name, "mesh_" + std::to_string(mesh_index)) +
                "' primitive " + std::to_string(primitive_index);
            if (primitive.type != cgltf_primitive_type_triangles &&
                primitive.type != cgltf_primitive_type_triangle_strip &&
                primitive.type != cgltf_primitive_type_triangle_fan) {
                std::cerr << "WARNING: Skipping " << context
                          << " because it is not a triangle primitive.\n";
                continue;
            }
            if (primitive.targets_count > 0) {
                std::cerr << "WARNING: Morph targets in " << context
                          << " are ignored; base geometry is imported.\n";
            }

            const cgltf_accessor *positions = cgltf_find_accessor(
                &primitive, cgltf_attribute_type_position, 0);
            if (!positions || positions->type != cgltf_type_vec3) {
                throw std::runtime_error(context +
                                         " requires a VEC3 POSITION accessor.");
            }
            const std::size_t vertex_count = positions->count;
            if (vertex_count >
                static_cast<std::size_t>(
                    std::numeric_limits<std::uint32_t>::max())) {
                throw std::runtime_error(context +
                                         " exceeds 32-bit mesh limits.");
            }
            const cgltf_accessor *normals = cgltf_find_accessor(
                &primitive, cgltf_attribute_type_normal, 0);
            const cgltf_accessor *tangents = cgltf_find_accessor(
                &primitive, cgltf_attribute_type_tangent, 0);
            const cgltf_accessor *uv0 = cgltf_find_accessor(
                &primitive, cgltf_attribute_type_texcoord, 0);
            const cgltf_accessor *color0 = cgltf_find_accessor(
                &primitive, cgltf_attribute_type_color, 0);
            if (normals) {
                require_accessor(normals, vertex_count, "NORMAL", context);
                if (normals->type != cgltf_type_vec3) {
                    throw std::runtime_error(
                        context + " requires a VEC3 NORMAL accessor.");
                }
            }
            if (tangents) {
                require_accessor(tangents, vertex_count, "TANGENT", context);
                if (tangents->type != cgltf_type_vec4) {
                    throw std::runtime_error(
                        context + " requires a VEC4 TANGENT accessor.");
                }
            }
            if (uv0) {
                require_accessor(uv0, vertex_count, "TEXCOORD_0", context);
                if (uv0->type != cgltf_type_vec2) {
                    throw std::runtime_error(
                        context + " requires a VEC2 TEXCOORD_0 accessor.");
                }
            }
            if (color0) {
                require_accessor(color0, vertex_count, "COLOR_0", context);
                if (color0->type != cgltf_type_vec3 &&
                    color0->type != cgltf_type_vec4) {
                    throw std::runtime_error(
                        context + " requires a VEC3 or VEC4 COLOR_0 accessor.");
                }
            }

            std::vector<MeshVertex> primitive_vertices(vertex_count);
            for (std::size_t index = 0; index < vertex_count; ++index) {
                MeshVertex &vertex = primitive_vertices[index];
                const auto position =
                    read_attribute(positions, index, context + " POSITION");
                vertex.position = point3(position[0], position[1], position[2]);
                if (normals) {
                    const auto value =
                        read_attribute(normals, index, context + " NORMAL");
                    const vec3 normal(value[0], value[1], value[2]);
                    vertex.normal = normal.near_zero()
                                        ? vec3(0, 0, 1)
                                        : unit_vector(normal);
                }
                if (tangents) {
                    const auto value = read_attribute(
                        tangents, index, context + " TANGENT");
                    const vec3 tangent(value[0], value[1], value[2]);
                    vertex.tangent = tangent.near_zero()
                                         ? vec3(1, 0, 0)
                                         : unit_vector(tangent);
                    vertex.tangent_sign = value[3] < 0.0f ? -1.0 : 1.0;
                }
                if (uv0) {
                    const auto value =
                        read_attribute(uv0, index, context + " TEXCOORD_0");
                    vertex.uv0 = vec2(value[0], value[1]);
                }
                if (color0) {
                    const auto value =
                        read_attribute(color0, index, context + " COLOR_0");
                    vertex.color0 = color(value[0], value[1], value[2]);
                    vertex.color_alpha = value[3];
                }
            }

            const std::vector<TriangleIndices> local_triangles =
                triangulate(primitive, vertex_count, primitive_vertices,
                            context);
            if (local_triangles.empty()) {
                continue;
            }
            bool has_normals = normals != nullptr;
            if (!has_normals && m_options.generate_normals) {
                generate_normals(primitive_vertices, local_triangles);
                has_normals = true;
            }
            bool has_tangents = tangents != nullptr;
            if (!has_tangents && m_options.generate_tangents && has_normals &&
                uv0) {
                has_tangents = generate_tangents(primitive_vertices,
                                                 local_triangles);
            }

            const std::size_t max_index =
                std::numeric_limits<std::uint32_t>::max();
            if (local_triangles.size() > max_index ||
                vertices.size() > max_index - primitive_vertices.size() ||
                triangles.size() > max_index - local_triangles.size()) {
                throw std::runtime_error(context +
                                         " exceeds 32-bit mesh limits.");
            }
            const std::uint32_t first_vertex =
                static_cast<std::uint32_t>(vertices.size());
            const std::uint32_t first_triangle =
                static_cast<std::uint32_t>(triangles.size());
            vertices.insert(vertices.end(), primitive_vertices.begin(),
                            primitive_vertices.end());

            std::uint8_t attributes = MESH_ATTRIBUTE_NONE;
            if (has_normals) {
                attributes |= MESH_ATTRIBUTE_NORMAL;
            }
            if (has_tangents) {
                attributes |= MESH_ATTRIBUTE_TANGENT;
            }
            if (uv0) {
                attributes |= MESH_ATTRIBUTE_UV0;
            }
            if (color0) {
                attributes |= MESH_ATTRIBUTE_COLOR0;
            }
            const std::uint32_t material_slot =
                primitive.material
                    ? static_cast<std::uint32_t>(
                          cgltf_material_index(&m_data, primitive.material))
                    : static_cast<std::uint32_t>(m_default_material_slot);
            for (const TriangleIndices &local : local_triangles) {
                MeshTriangle triangle;
                triangle.vertices[0] = first_vertex + local[0];
                triangle.vertices[1] = first_vertex + local[1];
                triangle.vertices[2] = first_vertex + local[2];
                triangle.primitive_index =
                    static_cast<std::uint32_t>(primitives.size());
                triangle.material_slot = material_slot;
                triangle.attributes = attributes;
                triangles.push_back(triangle);
            }
            primitives.push_back(
                {context, first_triangle,
                 static_cast<std::uint32_t>(triangles.size()) - first_triangle,
                 material_slot});
        }
        if (triangles.empty()) {
            return nullptr;
        }
        return std::make_shared<MeshAsset>(
            std::move(vertices), std::move(triangles), std::move(primitives),
            m_options.build_bvh);
    }

    void build_meshes() {
        m_mesh_mapping.assign(m_data.meshes_count, -1);
        for (cgltf_size index = 0; index < m_data.meshes_count; ++index) {
            std::shared_ptr<const MeshAsset> geometry =
                build_mesh(m_data.meshes[index], index);
            if (!geometry) {
                continue;
            }
            m_mesh_mapping[index] = static_cast<int>(m_meshes.size());
            m_meshes.push_back(
                {object_name(m_data.meshes[index].name,
                             "mesh_" + std::to_string(index)),
                 std::move(geometry)});
        }
    }

    void build_nodes() {
        m_nodes.resize(m_data.nodes_count);
        for (cgltf_size index = 0; index < m_data.nodes_count; ++index) {
            const cgltf_node &source = m_data.nodes[index];
            if (source.skin) {
                throw std::runtime_error(
                    "Skinned glTF nodes are not supported yet.");
            }
            if (source.has_mesh_gpu_instancing) {
                throw std::runtime_error(
                    "EXT_mesh_gpu_instancing is not supported yet.");
            }
            ModelNode &node = m_nodes[index];
            node.name = object_name(source.name,
                                    "node_" + std::to_string(index));
            float local[16];
            cgltf_node_transform_local(&source, local);
            double matrix[16];
            for (std::size_t element = 0; element < 16; ++element) {
                matrix[element] = local[element];
                if (!std::isfinite(matrix[element])) {
                    throw std::runtime_error(
                        "glTF node contains a non-finite transform.");
                }
            }
            node.local_transform =
                Transform(Matrix4::from_column_major(matrix));
            if (source.mesh) {
                const std::size_t source_mesh =
                    cgltf_mesh_index(&m_data, source.mesh);
                node.mesh_index = m_mesh_mapping[source_mesh];
            }
            node.children.reserve(source.children_count);
            for (cgltf_size child = 0; child < source.children_count; ++child) {
                node.children.push_back(
                    cgltf_node_index(&m_data, source.children[child]));
            }
        }
    }

    void build_scenes() {
        if (m_data.scenes_count == 0) {
            ModelScene scene;
            scene.name = "default";
            for (cgltf_size index = 0; index < m_data.nodes_count; ++index) {
                if (!m_data.nodes[index].parent) {
                    scene.roots.push_back(index);
                }
            }
            m_scenes.push_back(std::move(scene));
            m_default_scene = 0;
            return;
        }

        m_scenes.reserve(m_data.scenes_count);
        for (cgltf_size index = 0; index < m_data.scenes_count; ++index) {
            const cgltf_scene &source = m_data.scenes[index];
            ModelScene scene;
            scene.name = object_name(source.name,
                                     "scene_" + std::to_string(index));
            scene.roots.reserve(source.nodes_count);
            for (cgltf_size root = 0; root < source.nodes_count; ++root) {
                scene.roots.push_back(
                    cgltf_node_index(&m_data, source.nodes[root]));
            }
            m_scenes.push_back(std::move(scene));
        }
        m_default_scene = m_data.scene
                              ? static_cast<int>(
                                    cgltf_scene_index(&m_data, m_data.scene))
                              : 0;
    }

    std::string m_filename;
    ModelImportOptions m_options;
    ResourceRegistry &m_resources;
    const cgltf_data &m_data;
    std::size_t m_default_material_slot = 0;
    std::vector<int> m_mesh_mapping;
    std::vector<ModelMesh> m_meshes;
    std::vector<ModelNode> m_nodes;
    std::vector<ModelScene> m_scenes;
    int m_default_scene = 0;
    std::vector<std::string> m_material_names;
    std::vector<MaterialHandle> m_materials;
};

} // namespace

std::shared_ptr<const ModelAsset>
load_gltf_model_asset(const std::string &filename,
                      const ModelImportOptions &options,
                      ResourceRegistry &resources, std::string &error) {
    error.clear();
    try {
        cgltf_options parse_options{};
        cgltf_data *raw_data = nullptr;
        cgltf_result result =
            cgltf_parse_file(&parse_options, filename.c_str(), &raw_data);
        if (result != cgltf_result_success) {
            error = "cgltf parse failed: " + result_name(result);
            return nullptr;
        }
        CgltfData data(raw_data);
        result = cgltf_load_buffers(&parse_options, data.get(),
                                    filename.c_str());
        if (result != cgltf_result_success) {
            error = "cgltf buffer load failed: " + result_name(result);
            return nullptr;
        }
        result = cgltf_validate(data.get());
        if (result != cgltf_result_success) {
            error = "cgltf validation failed: " + result_name(result);
            return nullptr;
        }
        return ImportContext(filename, options, resources, *data).build();
    } catch (const std::exception &exception) {
        error = exception.what();
        return nullptr;
    }
}
