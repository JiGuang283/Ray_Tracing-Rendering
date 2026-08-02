#ifndef GLTF_IMPORTER_H
#define GLTF_IMPORTER_H

#include "model_asset.h"

#include <memory>
#include <string>

class ResourceRegistry;

std::shared_ptr<const ModelAsset>
load_gltf_model_asset(const std::string &filename,
                      const ModelImportOptions &options,
                      ResourceRegistry &resources, std::string &error);

#endif
