#include "render_data/packed_bsdf.h"

#include "render_data/packed_bsdf_core.h"

PackedBSDFStatus sample_packed_bsdf(const PackedMaterialOutput &output,
                                    Float3 wo, RNG &rng,
                                    PackedBSDFSample &sample) {
    return packed_bsdf::sample_packed_bsdf_core(output, wo, rng, sample);
}

PackedBSDFStatus evaluate_packed_bsdf(const PackedMaterialOutput &output,
                                      Float3 wo, Float3 wi,
                                      Float3 &result) {
    return packed_bsdf::eval_packed_bsdf_core(output, wo, wi, result);
}

PackedBSDFStatus packed_bsdf_pdf(const PackedMaterialOutput &output,
                                 Float3 wo, Float3 wi, float &result) {
    return packed_bsdf::pdf_packed_bsdf_core(output, wo, wi, result);
}

bool packed_bsdf_is_phase(const PackedMaterialOutput &output) {
    return packed_bsdf::output_is_phase(output);
}

float packed_bsdf_abs_cos_theta(const PackedMaterialOutput &output,
                                Float3 direction) {
    return packed_bsdf::abs_cos_theta(output, direction);
}
