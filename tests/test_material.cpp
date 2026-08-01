#include "material_programs.h"
#include "scene_ir.h"
#include "test_harness.h"

#include <stdexcept>

namespace {

ShaderEvalContext default_context() {
    ShaderEvalContext context;
    context.geometry_normal = vec3(0, 0, 1);
    context.shading_normal = vec3(0, 0, 1);
    context.frame.build_from_normal(vec3(0, 0, 1));
    context.wo = vec3(0, 0, 1);
    context.front_face = true;
    return context;
}

bool parse_throws(const std::string &source) {
    SceneDescription description;
    description.source_path = "test.json";
    description.root = nlohmann::json::parse(source);
    try {
        (void)parse_scene_ir(description);
    } catch (const std::runtime_error &) {
        return true;
    }
    return false;
}

class OversizedProgram final : public MaterialProgram {
  public:
    void evaluate(const ShaderEvalContext &, const MaterialParameterBlock &,
                  ShaderScratch &, MaterialOutput &) const override {
    }

    void validate(const MaterialParameterBlock &) const override {
    }

    std::size_t max_closures() const override {
        return BSDF::kMaxClosures + 1;
    }
};

} // namespace

TEST_CASE(lambertian_program_builds_a_fixed_bsdf) {
    const MaterialHandle material =
        make_lambertian_material(color(0.8, 0.4, 0.2));
    ShaderScratch scratch;
    MaterialOutput output;
    material->evaluate(default_context(), scratch, output);

    REQUIRE(output.bsdf.size() == 1);
    REQUIRE(!output.has_emission);
    REQUIRE_NEAR(output.bsdf.eval(vec3(0, 0, 1), vec3(0, 0, 1)).x(),
                 0.8 / pi, 1e-12);
}

TEST_CASE(material_metadata_is_available_without_shader_evaluation) {
    const color emission(4.0, 3.0, 2.0);
    const MaterialHandle material =
        make_diffuse_light_material(emission);
    REQUIRE(material->is_emissive());
    REQUIRE_NEAR(material->emission_estimate().x(), 4.0, 1e-12);

    ShaderScratch scratch;
    MaterialOutput output;
    material->evaluate(default_context(), scratch, output);
    REQUIRE(output.has_emission);
    REQUIRE_NEAR(output.emission.y(), 3.0, 1e-12);
}

TEST_CASE(material_factories_preserve_typed_host_descriptions) {
    const auto albedo =
        std::make_shared<SolidColorTexture>(color(0.2, 0.4, 0.6));
    const MaterialHandle lambertian = make_lambertian_material(albedo);
    REQUIRE(std::holds_alternative<LambertianMaterialDescription>(
        lambertian->description()));
    REQUIRE(std::get<LambertianMaterialDescription>(
                lambertian->description())
                .albedo == albedo);

    const MaterialHandle metal =
        make_metal_material(color(0.8, 0.7, 0.6), 2.0);
    const auto &description =
        std::get<MetalMaterialDescription>(metal->description());
    REQUIRE_NEAR(description.roughness, 1.0, 1e-12);
}

TEST_CASE(principled_emission_can_be_double_sided) {
    const TextureHandle white =
        std::make_shared<SolidColorTexture>(color(1, 1, 1));
    const TextureHandle black =
        std::make_shared<SolidColorTexture>(color(0, 0, 0));
    const MaterialHandle material = make_principled_material(
        white, white, black, nullptr, white, 2.0, nullptr, nullptr, {}, true);
    REQUIRE(material->is_emissive());
    REQUIRE(material->is_double_sided());

    ShaderEvalContext context = default_context();
    context.front_face = false;
    ShaderScratch scratch;
    MaterialOutput output;
    material->evaluate(context, scratch, output);
    REQUIRE(output.has_emission);
    REQUIRE_NEAR(output.emission.x(), 2.0, 1e-12);
}

TEST_CASE(material_program_closure_capacity_is_validated) {
    bool threw = false;
    try {
        (void)std::make_shared<const MaterialInstance>(
            std::make_shared<OversizedProgram>(),
            MaterialParameterBlock{});
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    REQUIRE(threw);
}

TEST_CASE(scene_ir_lowers_material_and_texture_types) {
    SceneDescription description;
    description.source_path = "test.json";
    description.root = nlohmann::json::parse(R"({
        "textures": {
            "albedo": {"type": "solid", "color": [0.2, 0.4, 0.6]}
        },
        "materials": {
            "mat": {"type": "lambertian", "albedo": "albedo"}
        },
        "objects": []
    })");

    const SceneIR ir = parse_scene_ir(description);
    REQUIRE(ir.textures.size() == 1);
    REQUIRE(ir.materials.size() == 1);
    REQUIRE(std::holds_alternative<ConstantTextureIR>(
        ir.textures[0].data));
    REQUIRE(std::holds_alternative<LambertianMaterialIR>(
        ir.materials[0].data));
}

TEST_CASE(scene_ir_rejects_unknown_texture_references) {
    REQUIRE(parse_throws(R"({
        "materials": {
            "mat": {"type": "lambertian", "albedo": "missing"}
        },
        "objects": []
    })"));
}

TEST_CASE(scene_ir_rejects_texture_cycles) {
    REQUIRE(parse_throws(R"({
        "textures": {
            "a": {"ref": "b"},
            "b": {"ref": "a"}
        },
        "materials": {
            "mat": {"type": "lambertian", "albedo": "a"}
        },
        "objects": []
    })"));
}

TEST_CASE(scene_ir_rejects_unknown_material_types) {
    REQUIRE(parse_throws(R"({
        "materials": {
            "mat": {"type": "unknown"}
        },
        "objects": []
    })"));
}
