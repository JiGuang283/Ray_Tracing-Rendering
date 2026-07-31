#!/usr/bin/env python3
"""Validate JSON scene catalog and scene files.

The validator is intentionally read-only. It checks the declarative scene data
that is now the source of truth, without regenerating or rewriting any JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = PROJECT_ROOT / "assets/scenes/catalog.json"

OBJECT_TYPES = {
    "accel",
    "box",
    "constant_medium",
    "flip_face",
    "list",
    "moving_sphere",
    "model",
    "obj",
    "quad",
    "rotate_y",
    "sphere",
    "translate",
    "transform",
    "triangle",
    "xy_rect",
    "xz_rect",
    "yz_rect",
}
MATERIAL_TYPES = {
    "lambertian",
    "metal",
    "dielectric",
    "diffuse_light",
    "pbr",
    "principled",
}
TEXTURE_TYPES = {
    "solid",
    "checker",
    "noise",
    "image",
    "scale",
    "multiply",
    "mix",
    "color_ramp",
}
LIGHT_TYPES = {"directional", "environment", "point", "quad", "spot"}
TONE_MAPPING_MODES = {"linear", "reinhard", "aces", "ACES"}
COLOR_SPACES = {"srgb", "linear"}
TEXTURE_CHANNELS = {"rgb", "r", "g", "b", "a"}
WRAP_MODES = {"repeat", "clamp", "mirror"}
FILTER_MODES = {"nearest", "bilinear"}
NORMAL_MAP_CONVENTIONS = {"opengl", "directx"}


class Reporter:
    def __init__(self) -> None:
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def error(self, context: str, message: str) -> None:
        self.errors.append(f"{context}: {message}")

    def warn(self, context: str, message: str) -> None:
        self.warnings.append(f"{context}: {message}")


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def require_object(value: Any, context: str, reporter: Reporter) -> bool:
    if not isinstance(value, dict):
        reporter.error(context, "expected object")
        return False
    return True


def require_array(value: Any, length: int, context: str, reporter: Reporter) -> bool:
    if not isinstance(value, list) or len(value) != length:
        reporter.error(context, f"expected array of length {length}")
        return False
    if not all(is_number(item) for item in value):
        reporter.error(context, "expected numeric array")
        return False
    return True


def require_number(object_: Dict[str, Any], key: str, context: str,
                   reporter: Reporter) -> None:
    if key not in object_:
        reporter.error(context, f"missing '{key}'")
    elif not is_number(object_[key]):
        reporter.error(f"{context}.{key}", "expected number")


def require_vec(object_: Dict[str, Any], key: str, length: int, context: str,
                reporter: Reporter) -> None:
    if key not in object_:
        reporter.error(context, f"missing '{key}'")
    else:
        require_array(object_[key], length, f"{context}.{key}", reporter)


def validate_texture_value(value: Any, textures: Set[str], context: str,
                           reporter: Reporter) -> None:
    if isinstance(value, str):
        if value not in textures:
            reporter.error(context, f"unknown texture reference '{value}'")
        return
    if is_number(value):
        return
    if isinstance(value, list):
        require_array(value, 3, context, reporter)
        return
    if isinstance(value, dict):
        validate_texture_object(value, textures, context, reporter)
        return
    reporter.error(context, "invalid texture value")


def validate_texture_object(texture: Dict[str, Any], textures: Set[str],
                            context: str, reporter: Reporter) -> None:
    if "ref" in texture:
        ref = texture["ref"]
        if not isinstance(ref, str):
            reporter.error(f"{context}.ref", "expected string")
        elif ref not in textures:
            reporter.error(f"{context}.ref", f"unknown texture reference '{ref}'")
        return

    texture_type = texture.get("type")
    if texture_type not in TEXTURE_TYPES:
        reporter.error(context, f"unknown texture type '{texture_type}'")
        return

    if texture_type == "solid":
        if "color" in texture:
            require_array(texture["color"], 3, f"{context}.color", reporter)
        elif "value" in texture:
            validate_texture_value(texture["value"], textures,
                                   f"{context}.value", reporter)
        else:
            reporter.error(context, "solid texture needs 'color' or 'value'")
    elif texture_type == "checker":
        even_key = "even" if "even" in texture else "color1"
        odd_key = "odd" if "odd" in texture else "color2"
        if even_key not in texture:
            reporter.error(context, "checker texture missing even/color1")
        else:
            validate_texture_value(texture[even_key], textures,
                                   f"{context}.{even_key}", reporter)
        if odd_key not in texture:
            reporter.error(context, "checker texture missing odd/color2")
        else:
            validate_texture_value(texture[odd_key], textures,
                                   f"{context}.{odd_key}", reporter)
    elif texture_type == "noise":
        if "scale" in texture and not is_number(texture["scale"]):
            reporter.error(f"{context}.scale", "expected number")
    elif texture_type == "image":
        path = texture.get("path")
        if not isinstance(path, str):
            reporter.error(context, "image texture missing string 'path'")
        elif not resolve_asset(path).exists():
            reporter.warn(context, f"image texture path does not exist: {path}")
        color_space = texture.get("color_space")
        if color_space is None:
            reporter.warn(
                context,
                "image texture omits color_space; runtime will infer it "
                "from the material input")
        elif not isinstance(color_space, str):
            reporter.error(
                f"{context}.color_space", "expected string")
        elif color_space not in COLOR_SPACES:
            reporter.error(
                f"{context}.color_space",
                f"unknown color space '{color_space}'")
        channel = texture.get("channel")
        if channel is not None and not isinstance(channel, str):
            reporter.error(f"{context}.channel", "expected string")
        elif channel is not None and channel not in TEXTURE_CHANNELS:
            reporter.error(
                f"{context}.channel",
                f"unknown texture channel '{channel}'")
        for field in ("wrap_u", "wrap_v"):
            if field in texture and not isinstance(texture[field], str):
                reporter.error(f"{context}.{field}", "expected string")
            elif field in texture and texture[field] not in WRAP_MODES:
                reporter.error(
                    f"{context}.{field}",
                    f"unknown wrap mode '{texture[field]}'")
        if "filter" in texture and not isinstance(texture["filter"], str):
            reporter.error(f"{context}.filter", "expected string")
        elif ("filter" in texture and
              texture["filter"] not in FILTER_MODES):
            reporter.error(
                f"{context}.filter",
                f"unknown filter mode '{texture['filter']}'")
    elif texture_type == "scale":
        if "input" not in texture:
            reporter.error(context, "scale texture missing 'input'")
        else:
            validate_texture_value(texture["input"], textures,
                                   f"{context}.input", reporter)
        if "scale" in texture and not is_number(texture["scale"]):
            reporter.error(f"{context}.scale", "expected number")
    elif texture_type == "multiply":
        for field in ("a", "b"):
            if field not in texture:
                reporter.error(context, f"multiply texture missing '{field}'")
            else:
                validate_texture_value(texture[field], textures,
                                       f"{context}.{field}", reporter)
    elif texture_type == "mix":
        for field in ("a", "b"):
            if field not in texture:
                reporter.error(context, f"mix texture missing '{field}'")
            else:
                validate_texture_value(texture[field], textures,
                                       f"{context}.{field}", reporter)
        if "factor" in texture:
            validate_texture_value(texture["factor"], textures,
                                   f"{context}.factor", reporter)
    elif texture_type == "color_ramp":
        if "input" not in texture:
            reporter.error(context, "color_ramp texture missing 'input'")
        else:
            validate_texture_value(texture["input"], textures,
                                   f"{context}.input", reporter)
        for field in ("low", "high"):
            if field not in texture:
                reporter.error(context, f"color_ramp texture missing '{field}'")
            else:
                require_array(texture[field], 3, f"{context}.{field}",
                              reporter)
        for field in ("min", "max"):
            if field in texture and not is_number(texture[field]):
                reporter.error(f"{context}.{field}", "expected number")


def texture_references(texture: Dict[str, Any]) -> Set[str]:
    if isinstance(texture.get("ref"), str):
        return {texture["ref"]}

    texture_type = texture.get("type")
    fields: tuple[str, ...] = ()
    if texture_type == "checker":
        fields = ("even", "odd", "color1", "color2")
    elif texture_type == "scale":
        fields = ("input",)
    elif texture_type == "multiply":
        fields = ("a", "b")
    elif texture_type == "mix":
        fields = ("a", "b", "factor")
    elif texture_type == "color_ramp":
        fields = ("input",)

    references: Set[str] = set()
    for field in fields:
        value = texture.get(field)
        if isinstance(value, str):
            references.add(value)
        elif isinstance(value, dict):
            references.update(texture_references(value))
    return references


def validate_texture_cycles(textures: Dict[str, Any], context: str,
                            reporter: Reporter) -> None:
    state: Dict[str, int] = {}

    def visit(name: str) -> None:
        if state.get(name) == 2:
            return
        if state.get(name) == 1:
            reporter.error(
                f"{context}.{name}",
                f"texture reference cycle involving '{name}'")
            return
        state[name] = 1
        texture = textures.get(name)
        if isinstance(texture, dict):
            for dependency in texture_references(texture):
                if dependency in textures:
                    visit(dependency)
        state[name] = 2

    for texture_name in textures:
        visit(texture_name)


def validate_material(material: Dict[str, Any], textures: Set[str], context: str,
                      reporter: Reporter) -> None:
    material_type = material.get("type")
    if material_type not in MATERIAL_TYPES:
        reporter.error(context, f"unknown material type '{material_type}'")
        return

    if material_type == "lambertian":
        field = next((key for key in ("texture", "albedo", "color")
                      if key in material), None)
        if field is None:
            reporter.error(context, "lambertian material needs texture/albedo/color")
        else:
            validate_texture_value(material[field], textures,
                                   f"{context}.{field}", reporter)
    elif material_type == "metal":
        require_vec(material, "albedo", 3, context, reporter)
        if "fuzz" in material and not is_number(material["fuzz"]):
            reporter.error(f"{context}.fuzz", "expected number")
    elif material_type == "dielectric":
        for key in ("ir", "index_of_refraction"):
            if key in material and not is_number(material[key]):
                reporter.error(f"{context}.{key}", "expected number")
    elif material_type == "diffuse_light":
        field = next((key for key in ("texture", "emit", "color")
                      if key in material), None)
        if field is None:
            reporter.error(context, "diffuse_light needs texture/emit/color")
        else:
            validate_texture_value(material[field], textures,
                                   f"{context}.{field}", reporter)
    elif material_type in {"pbr", "principled"}:
        base_field = "base_color" if "base_color" in material else "albedo"
        if base_field not in material:
            reporter.error(context,
                           f"{material_type} material missing base_color/albedo")
        else:
            validate_texture_value(material[base_field], textures,
                                   f"{context}.{base_field}", reporter)
        for field in ("roughness", "metallic", "normal", "emission",
                      "clearcoat", "clearcoat_roughness"):
            if field in material:
                validate_texture_value(material[field], textures,
                                       f"{context}.{field}", reporter)
        if "normal_texture" in material:
            validate_texture_value(
                material["normal_texture"], textures,
                f"{context}.normal_texture", reporter)
            reporter.warn(
                f"{context}.normal_texture",
                "legacy field; use normal_map.texture")
        if "normal_map" in material:
            normal_map = material["normal_map"]
            if not isinstance(normal_map, dict):
                reporter.error(f"{context}.normal_map", "expected object")
            elif "texture" not in normal_map:
                reporter.error(
                    f"{context}.normal_map", "missing 'texture'")
            else:
                validate_texture_value(
                    normal_map["texture"], textures,
                    f"{context}.normal_map.texture", reporter)
                convention = normal_map.get("convention", "opengl")
                if not isinstance(convention, str):
                    reporter.error(
                        f"{context}.normal_map.convention",
                        "expected string")
                elif convention not in NORMAL_MAP_CONVENTIONS:
                    reporter.error(
                        f"{context}.normal_map.convention",
                        f"unknown convention '{convention}'")
                if ("strength" in normal_map and
                        not is_number(normal_map["strength"])):
                    reporter.error(
                        f"{context}.normal_map.strength",
                        "expected number")
                elif normal_map.get("strength", 1.0) < 0.0:
                    reporter.error(
                        f"{context}.normal_map.strength",
                        "must be non-negative")
        if "emission_strength" in material and not is_number(
                material["emission_strength"]):
            reporter.error(f"{context}.emission_strength", "expected number")


def validate_material_ref(object_: Dict[str, Any], materials: Set[str],
                          context: str, reporter: Reporter) -> None:
    material = object_.get("material")
    if not isinstance(material, str):
        reporter.error(context, "missing string 'material'")
    elif material not in materials:
        reporter.error(context, f"unknown material reference '{material}'")


def validate_object(object_: Any, materials: Set[str], textures: Set[str],
                    context: str, reporter: Reporter) -> None:
    if not require_object(object_, context, reporter):
        return

    object_type = object_.get("type")
    if object_type not in OBJECT_TYPES:
        reporter.error(context, f"unknown object type '{object_type}'")
        return

    if object_type in {"sphere", "moving_sphere"}:
        center_key = "center" if object_type == "sphere" else "center0"
        require_vec(object_, center_key, 3, context, reporter)
        if object_type == "moving_sphere":
            require_vec(object_, "center1", 3, context, reporter)
            require_number(object_, "time0", context, reporter)
            require_number(object_, "time1", context, reporter)
        require_number(object_, "radius", context, reporter)
        validate_material_ref(object_, materials, context, reporter)
    elif object_type == "box":
        require_vec(object_, "min", 3, context, reporter)
        require_vec(object_, "max", 3, context, reporter)
        validate_material_ref(object_, materials, context, reporter)
    elif object_type in {"xy_rect", "xz_rect", "yz_rect"}:
        required = {
            "xy_rect": ("x0", "x1", "y0", "y1", "k"),
            "xz_rect": ("x0", "x1", "z0", "z1", "k"),
            "yz_rect": ("y0", "y1", "z0", "z1", "k"),
        }[object_type]
        for key in required:
            require_number(object_, key, context, reporter)
        validate_material_ref(object_, materials, context, reporter)
    elif object_type == "triangle":
        if "vertices" in object_:
            vertices = object_["vertices"]
            if not isinstance(vertices, list) or len(vertices) != 3:
                reporter.error(f"{context}.vertices",
                               "expected three vertex arrays")
            else:
                for index, vertex in enumerate(vertices):
                    require_array(vertex, 3,
                                  f"{context}.vertices[{index}]", reporter)
        else:
            for key in ("v0", "v1", "v2"):
                require_vec(object_, key, 3, context, reporter)
        for key in ("n0", "n1", "n2"):
            if key in object_:
                require_array(object_[key], 3, f"{context}.{key}", reporter)
        validate_material_ref(object_, materials, context, reporter)
    elif object_type == "obj":
        validate_material_ref(object_, materials, context, reporter)
        if "implementation" in object_:
            reporter.error(f"{context}.implementation",
                           "field was removed; MeshAsset is the only OBJ path")
        path = object_.get("path")
        if not isinstance(path, str):
            reporter.error(context, "obj missing string 'path'")
        elif not resolve_asset(path).exists():
            reporter.error(context, f"OBJ path does not exist: {path}")
        for key in ("translate", "scale", "position"):
            if key in object_:
                require_array(object_[key], 3, f"{context}.{key}", reporter)
    elif object_type == "model":
        path = object_.get("path")
        if not isinstance(path, str):
            reporter.error(context, "model missing string 'path'")
        elif not resolve_asset(path).exists():
            reporter.error(context, f"model path does not exist: {path}")
        elif Path(path).suffix.lower() not in {".gltf", ".glb"}:
            reporter.error(f"{context}.path", "expected .gltf or .glb model")
        if "scene" in object_ and (not isinstance(object_["scene"], int) or
                                   isinstance(object_["scene"], bool)):
            reporter.error(f"{context}.scene", "expected integer")
        transform = object_.get("transform")
        if transform is not None:
            validate_transform(transform, f"{context}.transform", reporter)
        overrides = object_.get("material_overrides", {})
        if not isinstance(overrides, dict):
            reporter.error(f"{context}.material_overrides", "expected object")
        else:
            for source, target in overrides.items():
                if not isinstance(target, str):
                    reporter.error(
                        f"{context}.material_overrides.{source}",
                        "expected material name")
                elif target not in materials:
                    reporter.error(
                        f"{context}.material_overrides.{source}",
                        f"unknown material reference '{target}'")
    elif object_type == "transform":
        if "object" not in object_:
            reporter.error(context, "missing nested 'object'")
        else:
            validate_object(object_["object"], materials, textures,
                            f"{context}.object", reporter)
        validate_transform(object_.get("transform"),
                           f"{context}.transform", reporter)
    elif object_type in {"translate", "rotate_y", "flip_face"}:
        if object_type == "translate":
            require_vec(object_, "offset", 3, context, reporter)
        if object_type == "rotate_y":
            require_number(object_, "angle", context, reporter)
        if "object" not in object_:
            reporter.error(context, "missing nested 'object'")
        else:
            validate_object(object_["object"], materials, textures,
                            f"{context}.object", reporter)
    elif object_type == "constant_medium":
        require_number(object_, "density", context, reporter)
        if "texture" in object_:
            validate_texture_value(
                object_["texture"], textures, f"{context}.texture", reporter)
        elif "color" in object_:
            require_array(object_["color"], 3, f"{context}.color", reporter)
        else:
            reporter.error(
                context, "constant_medium needs 'color' or 'texture'")
        nested_key = "boundary" if "boundary" in object_ else "object"
        if nested_key not in object_:
            reporter.error(
                context, "constant_medium missing 'boundary'/'object'")
        else:
            validate_object(object_[nested_key], materials, textures,
                            f"{context}.{nested_key}", reporter)
    elif object_type in {"list", "accel"}:
        validate_object_array(object_.get("objects"), materials, textures,
                              f"{context}.objects", reporter)


def validate_transform(value: Any, context: str,
                       reporter: Reporter) -> None:
    if not require_object(value, context, reporter):
        return
    if "matrix" in value:
        require_array(value["matrix"], 16, f"{context}.matrix", reporter)
        return
    for field in ("translation", "scale"):
        if field in value:
            require_array(value[field], 3, f"{context}.{field}", reporter)
    if "rotation" in value:
        require_array(value["rotation"], 4, f"{context}.rotation", reporter)


def validate_object_array(objects: Any, materials: Set[str],
                          textures: Set[str], context: str,
                          reporter: Reporter) -> None:
    if not isinstance(objects, list):
        reporter.error(context, "expected object array")
        return
    for index, object_ in enumerate(objects):
        validate_object(object_, materials, textures,
                        f"{context}[{index}]", reporter)


def validate_light(light: Dict[str, Any], context: str,
                   reporter: Reporter) -> None:
    light_type = light.get("type")
    if light_type not in LIGHT_TYPES:
        reporter.error(context, f"unknown light type '{light_type}'")
        return
    if light_type == "point":
        require_vec(light, "position", 3, context, reporter)
        require_vec(light, "intensity", 3, context, reporter)
    elif light_type == "directional":
        require_vec(light, "direction", 3, context, reporter)
        require_vec(light, "color", 3, context, reporter)
    elif light_type == "spot":
        require_vec(light, "position", 3, context, reporter)
        require_vec(light, "direction", 3, context, reporter)
        require_vec(light, "intensity", 3, context, reporter)
        require_number(light, "cutoff", context, reporter)
    elif light_type == "quad":
        for key in ("Q", "u", "v", "intensity"):
            require_vec(light, key, 3, context, reporter)
    elif light_type == "environment":
        path = light.get("path")
        if not isinstance(path, str):
            reporter.error(context, "environment light missing string 'path'")
        elif not resolve_asset(path).exists():
            reporter.warn(context, f"environment map path does not exist: {path}")


def validate_color_pipeline(value: Dict[str, Any], context: str,
                            reporter: Reporter) -> None:
    for key in ("exposure", "gamma"):
        if key in value:
            require_number(value, key, context, reporter)
    tone_mapping = value.get("tone_mapping")
    if tone_mapping is not None:
        if not isinstance(tone_mapping, str):
            reporter.error(f"{context}.tone_mapping", "expected string")
        elif tone_mapping not in TONE_MAPPING_MODES:
            reporter.error(f"{context}.tone_mapping",
                           f"unknown tone mapping '{tone_mapping}'")


def resolve_asset(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def validate_scene(path: Path, reporter: Reporter) -> None:
    context = str(path.relative_to(PROJECT_ROOT))
    try:
        scene = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - report JSON parser context
        reporter.error(context, f"failed to parse JSON: {exc}")
        return

    if not require_object(scene, context, reporter):
        return

    if "auto_emitters" in scene and not isinstance(scene["auto_emitters"], bool):
        reporter.error(f"{context}.auto_emitters", "expected boolean")

    camera = scene.get("camera")
    if not isinstance(camera, dict):
        reporter.error(context, "missing object 'camera'")
    else:
        for key in ("lookfrom", "lookat", "vup"):
            require_vec(camera, key, 3, f"{context}.camera", reporter)
        for key in ("vfov", "aperture", "focus_dist", "aspect_ratio"):
            require_number(camera, key, f"{context}.camera", reporter)

    render = scene.get("render")
    if not isinstance(render, dict):
        reporter.error(context, "missing object 'render'")
    else:
        for key in ("width", "spp"):
            require_number(render, key, f"{context}.render", reporter)
        require_vec(render, "background", 3, f"{context}.render", reporter)
        validate_color_pipeline(render, f"{context}.render", reporter)
        if "color_pipeline" in render:
            pipeline = render["color_pipeline"]
            if not isinstance(pipeline, dict):
                reporter.error(f"{context}.render.color_pipeline",
                               "expected object")
            else:
                validate_color_pipeline(
                    pipeline, f"{context}.render.color_pipeline", reporter)

    textures_json = scene.get("textures", {})
    if textures_json is None:
        textures_json = {}
    if not isinstance(textures_json, dict):
        reporter.error(f"{context}.textures", "expected object")
        textures_json = {}
    texture_names = set(textures_json)
    for name, texture in textures_json.items():
        if not isinstance(texture, dict):
            reporter.error(f"{context}.textures.{name}", "expected object")
            continue
        validate_texture_object(texture, texture_names,
                                f"{context}.textures.{name}", reporter)
    validate_texture_cycles(textures_json, f"{context}.textures", reporter)

    materials_json = scene.get("materials", {})
    if materials_json is None:
        materials_json = {}
    if not isinstance(materials_json, dict):
        reporter.error(f"{context}.materials", "expected object")
        materials_json = {}
    material_names = set(materials_json)
    for name, material in materials_json.items():
        if not isinstance(material, dict):
            reporter.error(f"{context}.materials.{name}", "expected object")
            continue
        validate_material(material, texture_names,
                          f"{context}.materials.{name}", reporter)

    validate_object_array(scene.get("objects"), material_names, texture_names,
                          f"{context}.objects", reporter)

    lights = scene.get("lights", [])
    if not isinstance(lights, list):
        reporter.error(f"{context}.lights", "expected array")
    else:
        for index, light in enumerate(lights):
            if not isinstance(light, dict):
                reporter.error(f"{context}.lights[{index}]", "expected object")
                continue
            validate_light(light, f"{context}.lights[{index}]", reporter)


def validate_catalog(catalog_path: Path, reporter: Reporter) -> List[Path]:
    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - report JSON parser context
        reporter.error(str(catalog_path), f"failed to parse JSON: {exc}")
        return []

    if not isinstance(catalog, dict):
        reporter.error(str(catalog_path), "expected object")
        return []
    if catalog.get("format") != "scene_catalog_v1":
        reporter.error(str(catalog_path), "format must be scene_catalog_v1")

    scenes = catalog.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        reporter.error(str(catalog_path), "missing non-empty scenes array")
        return []

    default_scene_id = catalog.get("default_scene_id")
    if not isinstance(default_scene_id, int) or isinstance(default_scene_id, bool):
        reporter.error(str(catalog_path), "default_scene_id must be an integer")

    seen_ids: Set[int] = set()
    scene_paths: List[Path] = []
    for index, item in enumerate(scenes):
        context = f"{catalog_path.relative_to(PROJECT_ROOT)}.scenes[{index}]"
        if not isinstance(item, dict):
            reporter.error(context, "expected object")
            continue
        scene_id = item.get("id")
        if not isinstance(scene_id, int) or isinstance(scene_id, bool):
            reporter.error(context, "id must be an integer")
        elif scene_id in seen_ids:
            reporter.error(context, f"duplicate scene id {scene_id}")
        else:
            seen_ids.add(scene_id)
        if not isinstance(item.get("name"), str):
            reporter.error(context, "name must be a string")
        path_value = item.get("path")
        if not isinstance(path_value, str):
            reporter.error(context, "path must be a string")
            continue
        scene_path = resolve_asset(path_value)
        if not scene_path.exists():
            reporter.error(context, f"scene path does not exist: {path_value}")
            continue
        scene_paths.append(scene_path)

    if isinstance(default_scene_id, int) and default_scene_id not in seen_ids:
        reporter.error(str(catalog_path), "default_scene_id is not in scenes")

    catalog_scene_files = {path.resolve() for path in scene_paths}
    all_scene_files = {path.resolve() for path in
                       (PROJECT_ROOT / "assets/scenes").glob("scene_*.json")}
    for extra in sorted(all_scene_files - catalog_scene_files):
        reporter.warn(str(catalog_path),
                      "scene file is not listed in catalog: " +
                      str(extra.relative_to(PROJECT_ROOT)))

    return scene_paths


def print_report(reporter: Reporter, quiet: bool) -> None:
    if not quiet:
        for warning in reporter.warnings:
            print(f"WARNING: {warning}", file=sys.stderr)
    for error in reporter.errors:
        print(f"ERROR: {error}", file=sys.stderr)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--strict-assets", action="store_true",
                        help="treat missing texture/HDR assets as errors")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress warnings")
    args = parser.parse_args(list(argv) if argv is not None else None)

    reporter = Reporter()
    scene_paths = validate_catalog(args.catalog, reporter)
    for scene_path in scene_paths:
        validate_scene(scene_path, reporter)

    if args.strict_assets and reporter.warnings:
        reporter.errors.extend(reporter.warnings)
        reporter.warnings = []

    print_report(reporter, args.quiet)
    if reporter.errors:
        print(f"Scene validation failed with {len(reporter.errors)} error(s).")
        return 1

    print(f"Validated {len(scene_paths)} scene(s); "
          f"{len(reporter.warnings)} warning(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
