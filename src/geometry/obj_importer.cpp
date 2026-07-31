#include "obj_importer.h"

#include "tiny_obj_loader.h"

#include <cstdint>
#include <exception>
#include <iterator>
#include <utility>
#include <vector>

std::shared_ptr<const MeshAsset>
load_obj_mesh_asset(const std::string &filename,
                    const ObjImportOptions &options, std::string &error) {
    tinyobj::ObjReaderConfig config;
    config.triangulate = true;
    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(filename, config)) {
        error = reader.Error();
        return nullptr;
    }

    const tinyobj::attrib_t &attributes = reader.GetAttrib();
    std::vector<MeshVertex> vertices;
    std::vector<MeshTriangle> triangles;
    std::vector<MeshPrimitive> primitives;

    auto read_vertex = [&](const tinyobj::index_t &index, MeshVertex &vertex,
                           std::uint8_t &flags) -> bool {
        if (index.vertex_index < 0 ||
            attributes.vertices.size() <=
                static_cast<std::size_t>(index.vertex_index * 3 + 2)) {
            return false;
        }
        const std::size_t position =
            static_cast<std::size_t>(index.vertex_index) * 3;
        vertex.position = point3(attributes.vertices[position],
                                 attributes.vertices[position + 1],
                                 attributes.vertices[position + 2]);

        if (options.use_vertex_normals && index.normal_index >= 0 &&
            attributes.normals.size() >
                static_cast<std::size_t>(index.normal_index * 3 + 2)) {
            const std::size_t normal =
                static_cast<std::size_t>(index.normal_index) * 3;
            vertex.normal = unit_vector(vec3(attributes.normals[normal],
                                              attributes.normals[normal + 1],
                                              attributes.normals[normal + 2]));
            flags |= MESH_ATTRIBUTE_NORMAL;
        }
        if (index.texcoord_index >= 0 &&
            attributes.texcoords.size() >
                static_cast<std::size_t>(index.texcoord_index * 2 + 1)) {
            const std::size_t uv =
                static_cast<std::size_t>(index.texcoord_index) * 2;
            vertex.uv0 =
                vec2(attributes.texcoords[uv], attributes.texcoords[uv + 1]);
            flags |= MESH_ATTRIBUTE_UV0;
        }
        return true;
    };

    for (const tinyobj::shape_t &shape : reader.GetShapes()) {
        const std::uint32_t first_triangle =
            static_cast<std::uint32_t>(triangles.size());
        std::size_t index_offset = 0;
        for (unsigned char face_size : shape.mesh.num_face_vertices) {
            if (face_size != 3) {
                index_offset += face_size;
                continue;
            }

            MeshVertex face_vertices[3];
            std::uint8_t vertex_flags[3]{MESH_ATTRIBUTE_NONE,
                                         MESH_ATTRIBUTE_NONE,
                                         MESH_ATTRIBUTE_NONE};
            bool valid = true;
            for (std::size_t corner = 0; corner < 3; ++corner) {
                valid = read_vertex(shape.mesh.indices[index_offset + corner],
                                    face_vertices[corner],
                                    vertex_flags[corner]) &&
                        valid;
            }
            index_offset += 3;
            if (!valid) {
                continue;
            }

            const std::uint32_t first_vertex =
                static_cast<std::uint32_t>(vertices.size());
            vertices.insert(vertices.end(), std::begin(face_vertices),
                            std::end(face_vertices));
            MeshTriangle triangle;
            triangle.vertices[0] = first_vertex;
            triangle.vertices[1] = first_vertex + 1;
            triangle.vertices[2] = first_vertex + 2;
            triangle.primitive_index =
                static_cast<std::uint32_t>(primitives.size());
            triangle.material_slot = 0;
            if ((vertex_flags[0] & MESH_ATTRIBUTE_NORMAL) != 0 &&
                (vertex_flags[1] & MESH_ATTRIBUTE_NORMAL) != 0 &&
                (vertex_flags[2] & MESH_ATTRIBUTE_NORMAL) != 0) {
                triangle.attributes |= MESH_ATTRIBUTE_NORMAL;
            }
            if ((vertex_flags[0] & MESH_ATTRIBUTE_UV0) != 0 &&
                (vertex_flags[1] & MESH_ATTRIBUTE_UV0) != 0 &&
                (vertex_flags[2] & MESH_ATTRIBUTE_UV0) != 0) {
                triangle.attributes |= MESH_ATTRIBUTE_UV0;
            }
            triangles.push_back(triangle);
        }
        if (triangles.size() > first_triangle) {
            primitives.push_back(
                {shape.name, first_triangle,
                 static_cast<std::uint32_t>(triangles.size()) -
                     first_triangle,
                 0});
        }
    }

    if (triangles.empty()) {
        error = "no valid triangles found";
        return nullptr;
    }

    try {
        return std::make_shared<MeshAsset>(
            std::move(vertices), std::move(triangles),
            std::move(primitives), options.build_bvh);
    } catch (const std::exception &exception) {
        error = exception.what();
        return nullptr;
    }
}
