#ifndef OBJ_IMPORTER_H
#define OBJ_IMPORTER_H

#include "mesh_asset.h"

#include <memory>
#include <string>

struct ObjImportOptions {
    bool build_bvh = true;
    bool use_vertex_normals = true;
};

std::shared_ptr<const MeshAsset>
load_obj_mesh_asset(const std::string &filename,
                    const ObjImportOptions &options,
                    std::string &error);

#endif
