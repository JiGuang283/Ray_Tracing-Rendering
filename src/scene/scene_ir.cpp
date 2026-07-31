#include "scene_ir.h"

#include "scene_loader_internal.h"

#include <stdexcept>
#include <type_traits>
#include <unordered_map>

using json = nlohmann::json;

namespace {

using namespace scene_loader_internal;

ToneMappingMode parse_tone_mapping(const std::string &mode,
                                   const std::string &context) {
    if (mode == "linear") {
        return ToneMappingMode::Linear;
    }
    if (mode == "reinhard") {
        return ToneMappingMode::Reinhard;
    }
    if (mode == "aces" || mode == "ACES") {
        return ToneMappingMode::ACES;
    }
    throw std::runtime_error("Scene file error: unknown tone_mapping '" +
                             mode + "' in " + context + ".");
}

void apply_color_pipeline_json(const json &pipeline_json,
                               ColorPipelineSettings &settings,
                               const std::string &context) {
    settings.exposure =
        read_double_or(pipeline_json, "exposure", settings.exposure);
    settings.gamma =
        read_double_or(pipeline_json, "gamma", settings.gamma);
    if (pipeline_json.contains("tone_mapping")) {
        settings.tone_mapping = parse_tone_mapping(
            pipeline_json["tone_mapping"].get<std::string>(),
            context + ".tone_mapping");
    }
}

void apply_camera_json(const json &root, SceneIR &ir) {
    if (!root.contains("camera")) {
        return;
    }
    const auto &camera_json = root["camera"];
    ir.camera.lookfrom =
        read_vec3_or(camera_json, "lookfrom", ir.camera.lookfrom, "camera");
    ir.camera.lookat =
        read_vec3_or(camera_json, "lookat", ir.camera.lookat, "camera");
    ir.camera.vup =
        read_vec3_or(camera_json, "vup", ir.camera.vup, "camera");
    ir.camera.vfov =
        read_double_or(camera_json, "vfov", ir.camera.vfov);
    ir.camera.aperture =
        read_double_or(camera_json, "aperture", ir.camera.aperture);
    ir.camera.focus_dist =
        read_double_or(camera_json, "focus_dist", ir.camera.focus_dist);
    ir.camera.aspect_ratio = read_double_or(
        camera_json, "aspect_ratio", ir.camera.aspect_ratio);
}

void apply_render_json(const json &root, SceneIR &ir) {
    if (!root.contains("render")) {
        return;
    }
    const auto &render_json = root["render"];
    ir.preset.image_width =
        read_int_or(render_json, "width", ir.preset.image_width);
    ir.preset.samples_per_pixel =
        read_int_or(render_json, "spp", ir.preset.samples_per_pixel);
    ir.preset.background =
        read_vec3_or(render_json, "background", ir.preset.background,
                     "render");
    apply_color_pipeline_json(render_json, ir.preset.color_pipeline,
                              "render");
    if (render_json.contains("color_pipeline")) {
        const auto &pipeline_json = render_json["color_pipeline"];
        if (!pipeline_json.is_object()) {
            throw std::runtime_error(
                "Scene file error: render.color_pipeline must be an object.");
        }
        apply_color_pipeline_json(pipeline_json, ir.preset.color_pipeline,
                                  "render.color_pipeline");
    }
}

WrapMode parse_wrap(const std::string &value, const std::string &context) {
    if (value == "repeat") {
        return WrapMode::Repeat;
    }
    if (value == "clamp") {
        return WrapMode::Clamp;
    }
    if (value == "mirror") {
        return WrapMode::Mirror;
    }
    throw std::runtime_error("Scene file error: unknown wrap mode '" + value +
                             "' in " + context + ".");
}

FilterMode parse_filter(const std::string &value,
                        const std::string &context) {
    if (value == "nearest") {
        return FilterMode::Nearest;
    }
    if (value == "bilinear") {
        return FilterMode::Bilinear;
    }
    throw std::runtime_error("Scene file error: unknown filter mode '" + value +
                             "' in " + context + ".");
}

TextureChannel parse_channel(const std::string &value,
                             const std::string &context) {
    if (value == "rgb") {
        return TextureChannel::RGB;
    }
    if (value == "r") {
        return TextureChannel::R;
    }
    if (value == "g") {
        return TextureChannel::G;
    }
    if (value == "b") {
        return TextureChannel::B;
    }
    if (value == "a") {
        return TextureChannel::A;
    }
    throw std::runtime_error("Scene file error: unknown texture channel '" +
                             value + "' in " + context + ".");
}

NormalMapConvention parse_normal_convention(const std::string &value,
                                            const std::string &context) {
    if (value == "opengl") {
        return NormalMapConvention::OpenGL;
    }
    if (value == "directx") {
        return NormalMapConvention::DirectX;
    }
    throw std::runtime_error(
        "Scene file error: unknown normal map convention '" + value +
        "' in " + context + ".");
}

class TextureIRParser {
  public:
    explicit TextureIRParser(SceneIR &ir) : m_ir(ir) {
    }

    void parse_named_textures(const json &root) {
        if (!root.contains("textures")) {
            return;
        }
        const json &textures = root["textures"];
        if (!textures.is_object()) {
            throw std::runtime_error(
                "Scene file error: 'textures' must be an object.");
        }

        for (auto it = textures.begin(); it != textures.end(); ++it) {
            const TextureIRId id = m_ir.textures.size();
            m_named.emplace(it.key(), id);
            m_ir.textures.push_back(
                TextureIRNode{it.key(), ConstantTextureIR{}});
        }
        for (auto it = textures.begin(); it != textures.end(); ++it) {
            const TextureIRId id = m_named.at(it.key());
            m_ir.textures[id].data =
                parse_texture_data(it.value(), "texture '" + it.key() + "'");
        }
    }

    TextureIRId parse_value(const json &value,
                            const std::string &context) {
        if (value.is_string()) {
            return lookup(value.get<std::string>(), context);
        }
        if (value.is_number()) {
            const double scalar = value.get<double>();
            return append(ConstantTextureIR{
                color(scalar, scalar, scalar)});
        }
        if (value.is_array()) {
            return append(
                ConstantTextureIR{read_vec3_value(value, context)});
        }
        if (value.is_object() && value.contains("ref")) {
            return lookup(value["ref"].get<std::string>(), context);
        }
        if (value.is_object()) {
            return append(parse_texture_data(value, context));
        }
        throw std::runtime_error("Scene file error: invalid texture value in " +
                                 context + ".");
    }

    void validate_graph() const {
        std::vector<int> state(m_ir.textures.size(), 0);
        for (TextureIRId id = 0; id < m_ir.textures.size(); ++id) {
            validate_node(id, state);
        }
    }

  private:
    TextureIRData parse_texture_data(const json &value,
                                     const std::string &context) {
        if (!value.is_object()) {
            const TextureIRId target = parse_value(value, context);
            return AliasTextureIR{target};
        }
        if (value.contains("ref")) {
            return AliasTextureIR{
                lookup(value["ref"].get<std::string>(), context)};
        }

        const std::string type = read_string(value, "type", context);
        if (type == "solid") {
            if (value.contains("color")) {
                return ConstantTextureIR{
                    read_vec3(value, "color", context)};
            }
            const json &data = require_key(value, "value", context);
            if (data.is_number()) {
                const double scalar = data.get<double>();
                return ConstantTextureIR{
                    color(scalar, scalar, scalar)};
            }
            return ConstantTextureIR{
                read_vec3_value(data, context + ".value")};
        }
        if (type == "checker") {
            const json &even =
                value.contains("even")
                    ? value["even"]
                    : require_key(value, "color1", context);
            const json &odd =
                value.contains("odd")
                    ? value["odd"]
                    : require_key(value, "color2", context);
            return CheckerTextureIR{
                parse_value(even, context + ".even"),
                parse_value(odd, context + ".odd")};
        }
        if (type == "noise") {
            return NoiseTextureIR{read_double_or(value, "scale", 1.0)};
        }
        if (type == "image") {
            ImageTextureIR image;
            image.path = read_string(value, "path", context);
            if (value.contains("color_space")) {
                const std::string color_space =
                    value["color_space"].get<std::string>();
                if (color_space == "srgb") {
                    image.color_space = TextureColorSpaceIR::SRGB;
                } else if (color_space == "linear") {
                    image.color_space = TextureColorSpaceIR::Linear;
                } else {
                    throw std::runtime_error(
                        "Scene file error: unknown color_space '" +
                        color_space + "' in " + context + ".");
                }
            }
            if (value.contains("channel")) {
                image.channel = parse_channel(
                    value["channel"].get<std::string>(),
                    context + ".channel");
                image.channel_explicit = true;
            }
            image.sampler.wrap_u = parse_wrap(
                value.value("wrap_u", std::string("repeat")),
                context + ".wrap_u");
            image.sampler.wrap_v = parse_wrap(
                value.value("wrap_v", std::string("repeat")),
                context + ".wrap_v");
            image.sampler.filter = parse_filter(
                value.value("filter", std::string("bilinear")),
                context + ".filter");
            return image;
        }
        if (type == "scale") {
            return ScaleTextureIR{
                parse_value(require_key(value, "input", context),
                            context + ".input"),
                read_double_or(value, "scale", 1.0)};
        }
        if (type == "multiply") {
            return MultiplyTextureIR{
                parse_value(require_key(value, "a", context),
                            context + ".a"),
                parse_value(require_key(value, "b", context),
                            context + ".b")};
        }
        if (type == "mix") {
            return MixTextureIR{
                parse_value(require_key(value, "a", context),
                            context + ".a"),
                parse_value(require_key(value, "b", context),
                            context + ".b"),
                parse_value(value.contains("factor") ? value["factor"]
                                                     : json(0.5),
                            context + ".factor")};
        }
        if (type == "color_ramp") {
            return ColorRampTextureIR{
                parse_value(require_key(value, "input", context),
                            context + ".input"),
                read_vec3(value, "low", context),
                read_vec3(value, "high", context),
                read_double_or(value, "min", 0.0),
                read_double_or(value, "max", 1.0)};
        }
        throw std::runtime_error("Scene file error: unknown texture type '" +
                                 type + "' in " + context + ".");
    }

    TextureIRId append(TextureIRData data) {
        const TextureIRId id = m_ir.textures.size();
        m_ir.textures.push_back(
            TextureIRNode{"#inline_" + std::to_string(id),
                          std::move(data)});
        return id;
    }

    TextureIRId lookup(const std::string &name,
                       const std::string &context) const {
        auto found = m_named.find(name);
        if (found == m_named.end()) {
            throw std::runtime_error("Scene file error: unknown texture '" +
                                     name + "' in " + context + ".");
        }
        return found->second;
    }

    std::vector<TextureIRId> dependencies(TextureIRId id) const {
        return std::visit(
            [](const auto &data) {
                using T = std::decay_t<decltype(data)>;
                if constexpr (std::is_same_v<T, AliasTextureIR>) {
                    return std::vector<TextureIRId>{data.target};
                } else if constexpr (std::is_same_v<T, CheckerTextureIR>) {
                    return std::vector<TextureIRId>{data.even, data.odd};
                } else if constexpr (std::is_same_v<T, ScaleTextureIR>) {
                    return std::vector<TextureIRId>{data.input};
                } else if constexpr (std::is_same_v<T,
                                                    MultiplyTextureIR>) {
                    return std::vector<TextureIRId>{data.a, data.b};
                } else if constexpr (std::is_same_v<T, MixTextureIR>) {
                    return std::vector<TextureIRId>{data.a, data.b,
                                                    data.factor};
                } else if constexpr (std::is_same_v<T,
                                                    ColorRampTextureIR>) {
                    return std::vector<TextureIRId>{data.input};
                } else {
                    return std::vector<TextureIRId>{};
                }
            },
            m_ir.textures[id].data);
    }

    void validate_node(TextureIRId id, std::vector<int> &state) const {
        if (id >= m_ir.textures.size()) {
            throw std::runtime_error(
                "Scene file error: invalid texture IR reference.");
        }
        if (state[id] == 2) {
            return;
        }
        if (state[id] == 1) {
            throw std::runtime_error(
                "Scene file error: texture cycle involving '" +
                m_ir.textures[id].name + "'.");
        }
        state[id] = 1;
        for (TextureIRId dependency : dependencies(id)) {
            validate_node(dependency, state);
        }
        state[id] = 2;
    }

    SceneIR &m_ir;
    std::unordered_map<std::string, TextureIRId> m_named;
};

MaterialIR parse_material(const std::string &name, const json &value,
                          TextureIRParser &textures) {
    if (!value.is_object()) {
        throw std::runtime_error("Scene file error: material '" + name +
                                 "' must be an object.");
    }
    const std::string context = "material '" + name + "'";
    const std::string type = read_string(value, "type", context);

    MaterialIR material;
    material.name = name;
    if (type == "lambertian") {
        const json &albedo =
            value.contains("texture")
                ? value["texture"]
                : (value.contains("albedo")
                       ? value["albedo"]
                       : require_key(value, "color", context));
        material.data =
            LambertianMaterialIR{textures.parse_value(albedo, context)};
        return material;
    }
    if (type == "metal") {
        material.data = MetalMaterialIR{
            read_vec3(value, "albedo", context),
            read_double_or(value, "fuzz", 0.0)};
        return material;
    }
    if (type == "dielectric") {
        material.data = DielectricMaterialIR{
            read_double_or(value, "ir",
                           read_double_or(value, "index_of_refraction",
                                          1.5))};
        return material;
    }
    if (type == "diffuse_light") {
        const json &emission =
            value.contains("texture")
                ? value["texture"]
                : (value.contains("emit")
                       ? value["emit"]
                       : require_key(value, "color", context));
        material.data = DiffuseLightMaterialIR{
            textures.parse_value(emission, context + ".emission")};
        return material;
    }
    if (type == "pbr" || type == "principled") {
        PrincipledMaterialIR principled;
        const json &base =
            value.contains("base_color")
                ? value["base_color"]
                : require_key(value, "albedo", context);
        principled.base_color =
            textures.parse_value(base, context + ".base_color");
        principled.roughness = textures.parse_value(
            value.contains("roughness") ? value["roughness"] : json(0.5),
            context + ".roughness");
        principled.metallic = textures.parse_value(
            value.contains("metallic") ? value["metallic"] : json(0.0),
            context + ".metallic");
        principled.emission_strength =
            read_double_or(value, "emission_strength", 1.0);

        if (value.contains("normal_map")) {
            const json &normal = value["normal_map"];
            if (!normal.is_object() || !normal.contains("texture")) {
                throw std::runtime_error(
                    "Scene file error: normal_map requires a texture in " +
                    context + ".");
            }
            principled.normal_map = textures.parse_value(
                normal["texture"], context + ".normal_map.texture");
            principled.normal_settings.strength =
                read_double_or(normal, "strength", 1.0);
            principled.normal_settings.convention =
                parse_normal_convention(
                    normal.value("convention", std::string("opengl")),
                    context + ".normal_map");
        } else if (value.contains("normal")) {
            principled.normal_map =
                textures.parse_value(value["normal"], context + ".normal");
        } else if (value.contains("normal_texture")) {
            principled.normal_map = textures.parse_value(
                value["normal_texture"], context + ".normal_texture");
        }
        if (value.contains("emission")) {
            principled.emission = textures.parse_value(
                value["emission"], context + ".emission");
        }
        if (value.contains("clearcoat")) {
            principled.clearcoat = textures.parse_value(
                value["clearcoat"], context + ".clearcoat");
        }
        if (value.contains("clearcoat_roughness")) {
            principled.clearcoat_roughness = textures.parse_value(
                value["clearcoat_roughness"],
                context + ".clearcoat_roughness");
        }
        material.data = principled;
        return material;
    }
    throw std::runtime_error("Scene file error: unknown material type '" +
                             type + "' for " + context + ".");
}

Quaternion parse_quaternion(const json &value, const std::string &context) {
    if (!value.is_array() || value.size() != 4) {
        throw std::runtime_error(
            "Scene file error: expected quaternion [x,y,z,w] in " +
            context + ".");
    }
    return Quaternion{value[0].get<double>(), value[1].get<double>(),
                      value[2].get<double>(), value[3].get<double>()};
}

Transform parse_transform(const json &value, const std::string &context) {
    if (!value.is_object()) {
        throw std::runtime_error("Scene file error: " + context +
                                 " must be an object.");
    }
    if (value.contains("matrix")) {
        const json &matrix = value["matrix"];
        if (!matrix.is_array() || matrix.size() != 16) {
            throw std::runtime_error(
                "Scene file error: transform.matrix must contain 16 numbers in " +
                context + ".");
        }
        double elements[16];
        for (std::size_t index = 0; index < 16; ++index) {
            elements[index] = matrix[index].get<double>();
        }
        return Transform(Matrix4::from_column_major(elements));
    }

    const vec3 translation =
        read_vec3_or(value, "translation", vec3(0, 0, 0), context);
    const vec3 scale = read_vec3_or(value, "scale", vec3(1, 1, 1), context);
    const Quaternion rotation =
        value.contains("rotation")
            ? parse_quaternion(value["rotation"], context + ".rotation")
            : Quaternion{};
    return Transform::from_trs(translation, rotation, scale);
}

class ObjectIRParser {
  public:
    ObjectIRParser(SceneIR &ir, TextureIRParser &textures)
        : m_ir(ir), m_textures(textures) {
    }

    ObjectIRId parse(const json &object, const std::string &context) {
        if (!object.is_object()) {
            throw std::runtime_error("Scene file error: " + context +
                                     " must be an object.");
        }
        const std::string type = read_string(object, "type", context);
        if (type == "sphere") {
            return append(context,
                          SphereObjectIR{
                              read_vec3(object, "center", context),
                              read_double_or(object, "radius", 1.0),
                              read_string(object, "material", context)});
        }
        if (type == "moving_sphere") {
            return append(
                context,
                MovingSphereObjectIR{
                    read_vec3(object, "center0", context),
                    read_vec3(object, "center1", context),
                    read_double_or(object, "time0", 0.0),
                    read_double_or(object, "time1", 1.0),
                    read_double_or(object, "radius", 1.0),
                    read_string(object, "material", context)});
        }
        if (type == "box") {
            return append(context,
                          BoxObjectIR{
                              read_vec3(object, "min", context),
                              read_vec3(object, "max", context),
                              read_string(object, "material", context)});
        }
        if (type == "xy_rect" || type == "xz_rect" || type == "yz_rect") {
            AxisRectObjectIR rectangle;
            rectangle.material = read_string(object, "material", context);
            if (type == "xy_rect") {
                rectangle.plane = AxisRectPlane::XY;
                rectangle.a0 = read_double_or(object, "x0", 0.0);
                rectangle.a1 = read_double_or(object, "x1", 1.0);
                rectangle.b0 = read_double_or(object, "y0", 0.0);
                rectangle.b1 = read_double_or(object, "y1", 1.0);
            } else if (type == "xz_rect") {
                rectangle.plane = AxisRectPlane::XZ;
                rectangle.a0 = read_double_or(object, "x0", 0.0);
                rectangle.a1 = read_double_or(object, "x1", 1.0);
                rectangle.b0 = read_double_or(object, "z0", 0.0);
                rectangle.b1 = read_double_or(object, "z1", 1.0);
            } else {
                rectangle.plane = AxisRectPlane::YZ;
                rectangle.a0 = read_double_or(object, "y0", 0.0);
                rectangle.a1 = read_double_or(object, "y1", 1.0);
                rectangle.b0 = read_double_or(object, "z0", 0.0);
                rectangle.b1 = read_double_or(object, "z1", 1.0);
            }
            rectangle.k = read_double_or(object, "k", 0.0);
            return append(context, rectangle);
        }
        if (type == "quad") {
            return append(context,
                          QuadObjectIR{
                              read_vec3(object, "Q", context),
                              read_vec3(object, "u", context),
                              read_vec3(object, "v", context),
                              read_string(object, "material", context)});
        }
        if (type == "triangle") {
            TriangleObjectIR triangle;
            if (object.contains("vertices")) {
                const json &positions = object["vertices"];
                if (!positions.is_array() || positions.size() != 3) {
                    throw std::runtime_error(
                        "Scene file error: triangle.vertices must contain 3 "
                        "points in " +
                        context + ".");
                }
                for (std::size_t index = 0; index < 3; ++index) {
                    triangle.positions[index] = read_vec3_value(
                        positions[index], context + ".vertices[" +
                                              std::to_string(index) + "]");
                }
            } else {
                triangle.positions = {read_vec3(object, "v0", context),
                                      read_vec3(object, "v1", context),
                                      read_vec3(object, "v2", context)};
            }
            triangle.has_normals = object.contains("n0") &&
                                   object.contains("n1") &&
                                   object.contains("n2");
            if (triangle.has_normals) {
                triangle.normals = {
                    unit_vector(read_vec3(object, "n0", context)),
                    unit_vector(read_vec3(object, "n1", context)),
                    unit_vector(read_vec3(object, "n2", context))};
            }
            triangle.has_uv0 = object.contains("uv0") &&
                               object.contains("uv1") &&
                               object.contains("uv2");
            if (triangle.has_uv0) {
                triangle.uv0 = {
                    read_vec2_value(object["uv0"], context + ".uv0"),
                    read_vec2_value(object["uv1"], context + ".uv1"),
                    read_vec2_value(object["uv2"], context + ".uv2")};
            }
            triangle.material = read_string(object, "material", context);
            return append(context, triangle);
        }
        if (type == "obj") {
            if (object.contains("implementation")) {
                throw std::runtime_error(
                    "Scene file error: obj.implementation was removed in " +
                    context + ".");
            }
            ObjObjectIR obj;
            obj.path = read_string(object, "path", context);
            obj.local_translation = read_vec3_or(
                object, "translate", vec3(0, 0, 0), context);
            obj.scale =
                read_vec3_or(object, "scale", vec3(1, 1, 1), context);
            obj.rotation_y = read_double_or(object, "rotation_y", 0.0);
            obj.has_position = object.contains("position");
            obj.position =
                read_vec3_or(object, "position", vec3(0, 0, 0), context);
            obj.auto_lift_to_ground =
                read_bool_or(object, "auto_lift_to_ground", false);
            obj.build_bvh = read_bool_or(object, "build_bvh", true);
            obj.use_vertex_normals =
                read_bool_or(object, "use_vertex_normals", true);
            obj.material = read_string(object, "material", context);
            return append(context, obj);
        }
        if (type == "model") {
            ModelObjectIR model;
            model.path = read_string(object, "path", context);
            model.scene_index = read_int_or(object, "scene", -1);
            if (object.contains("transform")) {
                model.transform = parse_transform(object["transform"],
                                                  context + ".transform");
            }
            if (object.contains("material_overrides")) {
                const json &overrides = object["material_overrides"];
                if (!overrides.is_object()) {
                    throw std::runtime_error(
                        "Scene file error: material_overrides must be an "
                        "object in " +
                        context + ".");
                }
                for (auto it = overrides.begin(); it != overrides.end(); ++it) {
                    model.material_overrides.emplace(
                        it.key(), it.value().get<std::string>());
                }
            }
            return append(context, model);
        }
        if (type == "translate") {
            const ObjectIRId child = parse(
                require_key(object, "object", context), context + ".object");
            const vec3 offset =
                object.contains("offset")
                    ? read_vec3(object, "offset", context)
                    : read_vec3_or(object, "translate", vec3(0, 0, 0),
                                   context);
            return append(context,
                          TransformObjectIR{child,
                                            Transform::translate(offset)});
        }
        if (type == "rotate_y") {
            const ObjectIRId child = parse(
                require_key(object, "object", context), context + ".object");
            return append(
                context,
                TransformObjectIR{
                    child, Transform::rotate_y(
                               read_double_or(object, "angle", 0.0))});
        }
        if (type == "transform") {
            const ObjectIRId child = parse(
                require_key(object, "object", context), context + ".object");
            const json &transform_json =
                object.contains("transform") ? object["transform"] : object;
            return append(context,
                          TransformObjectIR{
                              child, parse_transform(transform_json,
                                                     context + ".transform")});
        }
        if (type == "flip_face") {
            return append(
                context,
                FlipFaceObjectIR{
                    parse(require_key(object, "object", context),
                          context + ".object")});
        }
        if (type == "constant_medium") {
            const json &boundary =
                object.contains("boundary")
                    ? object["boundary"]
                    : require_key(object, "object", context);
            ConstantMediumObjectIR medium;
            medium.boundary = parse(boundary, context + ".boundary");
            medium.density = read_double_or(object, "density", 1.0);
            if (object.contains("texture")) {
                medium.texture = m_textures.parse_value(
                    object["texture"], context + ".texture");
            } else {
                medium.albedo = read_vec3_or(
                    object, "color", color(1, 1, 1), context);
            }
            return append(context, medium);
        }
        if (type == "list" || type == "accel") {
            const json &children = require_key(object, "objects", context);
            if (!children.is_array()) {
                throw std::runtime_error("Scene file error: " + context +
                                         ".objects must be an array.");
            }
            GroupObjectIR group;
            group.accelerate = type == "accel" ||
                               read_bool_or(object, "accelerate", false);
            group.time0 = read_double_or(object, "time0", 0.0);
            group.time1 = read_double_or(object, "time1", 1.0);
            group.children.reserve(children.size());
            for (std::size_t index = 0; index < children.size(); ++index) {
                group.children.push_back(parse(
                    children[index], context + ".objects[" +
                                         std::to_string(index) + "]"));
            }
            return append(context, std::move(group));
        }
        throw std::runtime_error("Scene file error: unknown object type '" +
                                 type + "' in " + context + ".");
    }

  private:
    template <typename T>
    ObjectIRId append(const std::string &context, T data) {
        const ObjectIRId id = m_ir.object_nodes.size();
        m_ir.object_nodes.push_back(
            ObjectIRNode{context, ObjectIRData{std::move(data)}});
        return id;
    }

    SceneIR &m_ir;
    TextureIRParser &m_textures;
};

LightIR parse_light(const json &value, const std::string &context) {
    if (!value.is_object()) {
        throw std::runtime_error("Scene file error: " + context +
                                 " must be an object.");
    }
    const std::string type = read_string(value, "type", context);
    LightIR light;
    light.context = context;
    if (type == "point") {
        light.data = PointLightIR{read_vec3(value, "position", context),
                                  read_vec3(value, "intensity", context)};
    } else if (type == "directional") {
        light.data =
            DirectionalLightIR{read_vec3(value, "direction", context),
                               read_vec3(value, "color", context)};
    } else if (type == "spot") {
        light.data = SpotLightIR{
            read_vec3(value, "position", context),
            read_vec3(value, "direction", context),
            read_double_or(value, "cutoff", 20.0),
            read_vec3(value, "intensity", context)};
    } else if (type == "quad") {
        light.data = QuadLightIR{read_vec3(value, "Q", context),
                                 read_vec3(value, "u", context),
                                 read_vec3(value, "v", context),
                                 read_vec3(value, "intensity", context)};
    } else if (type == "environment") {
        light.data =
            EnvironmentLightIR{read_string(value, "path", context)};
    } else {
        throw std::runtime_error("Scene file error: unknown light type '" +
                                 type + "' in " + context + ".");
    }
    return light;
}

} // namespace

SceneIR parse_scene_ir(const SceneDescription &description) {
    const json &root = description.root;
    if (!root.is_object()) {
        throw std::runtime_error("Scene file error: root must be an object.");
    }

    SceneIR ir;
    ir.source_path = description.source_path;
    if (root.contains("name")) {
        ir.name = root["name"].get<std::string>();
    }
    apply_camera_json(root, ir);
    apply_render_json(root, ir);
    ir.world_accel = read_bool_or(root, "world_accel", true);
    ir.time0 = read_double_or(root, "time0", 0.0);
    ir.time1 = read_double_or(root, "time1", 1.0);

    TextureIRParser texture_parser(ir);
    texture_parser.parse_named_textures(root);

    const json &materials = require_key(root, "materials", "root");
    if (!materials.is_object()) {
        throw std::runtime_error(
            "Scene file error: 'materials' must be an object.");
    }
    for (auto it = materials.begin(); it != materials.end(); ++it) {
        ir.materials.push_back(
            parse_material(it.key(), it.value(), texture_parser));
    }

    const json &objects = require_key(root, "objects", "root");
    if (!objects.is_array()) {
        throw std::runtime_error(
            "Scene file error: 'objects' must be an array.");
    }
    ObjectIRParser object_parser(ir, texture_parser);
    for (std::size_t i = 0; i < objects.size(); ++i) {
        ir.objects.push_back(object_parser.parse(
            objects[i], "objects[" + std::to_string(i) + "]"));
    }

    const bool has_explicit_lights =
        root.contains("lights") && root["lights"].is_array() &&
        !root["lights"].empty();
    ir.auto_emitters =
        read_bool_or(root, "auto_emitters", !has_explicit_lights);
    if (root.contains("lights")) {
        const json &lights = root["lights"];
        if (!lights.is_array()) {
            throw std::runtime_error(
                "Scene file error: 'lights' must be an array.");
        }
        for (std::size_t i = 0; i < lights.size(); ++i) {
            ir.lights.push_back(parse_light(
                lights[i], "lights[" + std::to_string(i) + "]"));
        }
    }

    texture_parser.validate_graph();
    return ir;
}
