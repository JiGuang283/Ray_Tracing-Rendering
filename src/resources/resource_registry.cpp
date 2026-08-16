#include "resource_registry.h"

#include "gltf_importer.h"
#include "obj_importer.h"

#include <filesystem>
#include <iostream>
#include <stdexcept>

std::shared_ptr<const ImageAsset>
ResourceRegistry::load_image(const std::string &path) {
    const std::string normalized = normalize_path(path);
    auto found = m_images.find(normalized);
    if (found != m_images.end()) {
        return found->second;
    }

    std::string error;
    auto image = ImageAsset::load(normalized, error);
    if (!image) {
        if (m_strict_assets) {
            throw std::runtime_error("failed to load image resource '" +
                                     path + "': " + error);
        }
        std::cerr << "ERROR: Could not load image resource '" << path
                  << "': " << error << ". Using diagnostic texture.\n";
        image = ImageAsset::diagnostic();
    }
    m_images.emplace(normalized, image);
    return image;
}

std::shared_ptr<const ImageAsset> ResourceRegistry::load_image_from_memory(
    const std::string &resource_key, const std::uint8_t *data,
    std::size_t size) {
    auto found = m_images.find(resource_key);
    if (found != m_images.end()) {
        return found->second;
    }

    std::string error;
    auto image = ImageAsset::load_from_memory(data, size, error);
    if (!image) {
        if (m_strict_assets) {
            throw std::runtime_error(
                "failed to load embedded image resource '" + resource_key +
                "': " + error);
        }
        std::cerr << "ERROR: Could not load embedded image resource '"
                  << resource_key << "': " << error
                  << ". Using diagnostic texture.\n";
        image = ImageAsset::diagnostic();
    }
    m_images.emplace(resource_key, image);
    return image;
}

std::shared_ptr<const ModelAsset>
ResourceRegistry::load_model(const std::string &path,
                             const ModelImportOptions &options,
                             std::string &error) {
    error.clear();
    const std::string normalized = normalize_path(path);
    const std::string cache_key =
        normalized + "#bvh=" + (options.build_bvh ? "1" : "0") +
        ";normals=" + (options.generate_normals ? "1" : "0") +
        ";tangents=" + (options.generate_tangents ? "1" : "0");
    auto found = m_models.find(cache_key);
    if (found != m_models.end()) {
        return found->second;
    }

    auto model =
        load_gltf_model_asset(normalized, options, *this, error);
    if (model) {
        m_models.emplace(cache_key, model);
    }
    return model;
}

std::shared_ptr<const MeshAsset>
ResourceRegistry::load_obj(const std::string &path, bool build_bvh,
                           bool use_vertex_normals, std::string &error) {
    error.clear();
    const std::string normalized = normalize_path(path);
    const std::string cache_key =
        normalized + "#bvh=" + (build_bvh ? "1" : "0") +
        ";normals=" + (use_vertex_normals ? "1" : "0");
    auto found = m_meshes.find(cache_key);
    if (found != m_meshes.end()) {
        return found->second;
    }

    ObjImportOptions options;
    options.build_bvh = build_bvh;
    options.use_vertex_normals = use_vertex_normals;
    auto mesh = load_obj_mesh_asset(normalized, options, error);
    if (mesh) {
        m_meshes.emplace(cache_key, mesh);
    }
    return mesh;
}

std::size_t ResourceRegistry::image_count() const {
    return m_images.size();
}

std::size_t ResourceRegistry::model_count() const {
    return m_models.size();
}

std::size_t ResourceRegistry::mesh_count() const {
    return m_meshes.size();
}

std::string ResourceRegistry::normalize_path(const std::string &path) {
    std::error_code error;
    std::filesystem::path candidate(path);
    if (std::filesystem::exists(candidate, error)) {
        return std::filesystem::weakly_canonical(candidate, error).string();
    }
    error.clear();
    return std::filesystem::absolute(candidate, error)
        .lexically_normal()
        .string();
}
