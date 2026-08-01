#include "material.h"

#include <stdexcept>
#include <utility>

std::size_t MaterialParameterBlock::size() const {
    return m_values.size();
}

const MaterialParameterValue &
MaterialParameterBlock::operator[](std::size_t index) const {
    return m_values.at(index);
}

MaterialInstance::MaterialInstance(
    std::shared_ptr<const MaterialProgram> program,
    MaterialParameterBlock parameters, MaterialMetadata metadata,
    MaterialDescription description)
    : m_program(std::move(program)), m_parameters(std::move(parameters)),
      m_metadata(metadata), m_description(std::move(description)) {
    if (!m_program) {
        throw std::invalid_argument("MaterialInstance requires a program");
    }
    if (m_program->max_closures() > BSDF::kMaxClosures) {
        throw std::invalid_argument(
            "MaterialProgram exceeds BSDF closure capacity");
    }
    m_program->validate(m_parameters);
}

bool MaterialInstance::is_emissive() const {
    return m_metadata.emissive;
}

bool MaterialInstance::is_double_sided() const {
    return m_metadata.double_sided;
}

const color &MaterialInstance::emission_estimate() const {
    return m_metadata.emission_estimate;
}

const MaterialParameterBlock &MaterialInstance::parameters() const {
    return m_parameters;
}

const std::shared_ptr<const MaterialProgram> &
MaterialInstance::program() const {
    return m_program;
}

const MaterialDescription &MaterialInstance::description() const {
    return m_description;
}
