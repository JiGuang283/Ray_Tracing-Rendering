#ifndef PACKED_BSDF_H
#define PACKED_BSDF_H

#include "packed_types.h"
#include "rng.h"

PackedBSDFStatus sample_packed_bsdf(const PackedMaterialOutput &output,
                                    Float3 wo, RNG &rng,
                                    PackedBSDFSample &sample);

PackedBSDFStatus evaluate_packed_bsdf(const PackedMaterialOutput &output,
                                      Float3 wo, Float3 wi,
                                      Float3 &result);

PackedBSDFStatus packed_bsdf_pdf(const PackedMaterialOutput &output,
                                 Float3 wo, Float3 wi, float &result);

bool packed_bsdf_is_phase(const PackedMaterialOutput &output);
float packed_bsdf_abs_cos_theta(const PackedMaterialOutput &output,
                                Float3 direction);

#endif
