#ifndef MATERIAL_H
#define MATERIAL_H

#include "shader_context.h"
#include "shading.h"
#include "texture.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

using MaterialParameterValue =
    std::variant<double, color, TextureHandle, bool, std::uint32_t>;

class MaterialParameterBlock {
  public:
    template <typename T> std::size_t add(T value) {
        m_values.emplace_back(std::move(value));
        return m_values.size() - 1;
    }

    template <typename T> const T &get(std::size_t index) const {
        return std::get<T>(m_values[index]);
    }

    std::size_t size() const;
    const MaterialParameterValue &operator[](std::size_t index) const;

  private:
    std::vector<MaterialParameterValue> m_values;
};

struct MaterialMetadata {
    bool emissive = false;
    bool double_sided = false;
    color emission_estimate{0, 0, 0};
};

struct LambertianMaterialDescription {
    TextureHandle albedo;
};

struct MetalMaterialDescription {
    color albedo{1, 1, 1};
    double roughness = 0.0;
};

struct DielectricMaterialDescription {
    double ior = 1.5;
};

struct DiffuseLightMaterialDescription {
    TextureHandle emission;
    bool double_sided = true;
};

struct PrincipledMaterialDescription {
    TextureHandle base_color;
    TextureHandle roughness;
    TextureHandle metallic;
    TextureHandle normal_map;
    TextureHandle emission;
    TextureHandle clearcoat;
    TextureHandle clearcoat_roughness;
    double emission_strength = 1.0;
    double normal_strength = 1.0;
    std::uint32_t normal_convention = 0;
    bool double_sided = false;
};

struct IsotropicMaterialDescription {
    TextureHandle albedo;
};

using MaterialDescription =
    std::variant<std::monostate, LambertianMaterialDescription,
                 MetalMaterialDescription, DielectricMaterialDescription,
                 DiffuseLightMaterialDescription,
                 PrincipledMaterialDescription,
                 IsotropicMaterialDescription>;

class MaterialProgram {
  public:
    virtual ~MaterialProgram() = default;

    virtual void evaluate(const ShaderEvalContext &context,
                          const MaterialParameterBlock &parameters,
                          ShaderScratch &scratch,
                          MaterialOutput &output) const = 0;
    virtual void validate(const MaterialParameterBlock &parameters) const = 0;
    virtual std::size_t max_closures() const = 0;
};

class MaterialInstance {
  public:
    MaterialInstance(std::shared_ptr<const MaterialProgram> program,
                     MaterialParameterBlock parameters,
                     MaterialMetadata metadata = {},
                     MaterialDescription description = {});

    void evaluate(const ShaderEvalContext &context, ShaderScratch &scratch,
                  MaterialOutput &output) const {
        scratch.reset();
        m_program->evaluate(context, m_parameters, scratch, output);
    }

    bool is_emissive() const;
    bool is_double_sided() const;
    const color &emission_estimate() const;
    const MaterialParameterBlock &parameters() const;
    const std::shared_ptr<const MaterialProgram> &program() const;
    const MaterialDescription &description() const;

  private:
    std::shared_ptr<const MaterialProgram> m_program;
    MaterialParameterBlock m_parameters;
    MaterialMetadata m_metadata;
    MaterialDescription m_description;
};

using MaterialHandle = std::shared_ptr<const MaterialInstance>;

#endif
