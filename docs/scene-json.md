# JSON 场景系统

当前项目的运行时场景来源是 `assets/scenes/catalog.json` 和 catalog
引用的 `assets/scenes/scene_*.json` 文件。旧的 C++ 内置场景、场景注册表、
procedural generator、legacy mesh 路径和旧 pointer BVH 都已经从默认运行路径中
移除。

## 入口规则

- `./build/CGAssignment4 23 4`：
  - 根据 scene id 查 `assets/scenes/catalog.json`。
  - 加载 catalog 中对应的 JSON 场景文件。
- `./build/CGAssignment4 --scene-file PATH --integrator 4`：
  - 直接加载指定 JSON。
  - 不允许同时传 positional scene id。
- catalog 中的 `default_scene_id` 用于未知 scene id 的 fallback。

## Scene 文件结构

顶层字段：

```json
{
  "name": "scene_name",
  "world_accel": true,
  "camera": {},
  "render": {},
  "textures": {},
  "materials": {},
  "objects": [],
  "lights": []
}
```

`camera` 支持：

- `lookfrom`
- `lookat`
- `vup`
- `vfov`
- `aperture`
- `focus_dist`
- `aspect_ratio`

`render` 支持：

- `width`
- `spp`
- `background`

## 支持的资源类型

Texture：

- `solid`
- `checker`
- `noise`
- `image`

Material：

- `lambertian`
- `metal`
- `dielectric`
- `diffuse_light`
- `pbr`

Object：

- `sphere`
- `moving_sphere`
- `box`
- `xy_rect`
- `xz_rect`
- `yz_rect`
- `quad`
- `triangle`
- `obj`
- `translate`
- `rotate_y`
- `flip_face`
- `constant_medium`
- `list`
- `accel`

Light：

- `point`
- `directional`
- `spot`
- `quad`
- `environment`

## 已移除路径

- `obj.implementation`
  - 已移除。OBJ 统一使用 `FlatMesh`。
- `random_scene_generator`
  - 已移除。scene 1/6 已展开为显式 JSON。
- `final_scene_generator`
  - 已移除。scene 9/22 已展开为显式 JSON。
- `SceneRegistry`
  - 已移除。`select_scene(id)` 直接读取 JSON catalog。
- `bvh_node`
  - 已移除。顶层加速结构使用 `LinearBVH`。
- `mesh`
  - 已移除。OBJ 使用 `FlatMesh`。
- `material::scatter`
  - 已移除。renderer 使用 `sample/eval/pdf/emitted`。

## 验证命令

构建：

```bash
cmake -S . -B build
cmake --build build -j2
```

校验所有 JSON 场景：

```bash
python3 tools/validate_scenes.py
```

当前缺失的 `earthmap.jpg` 和若干 HDR 环境贴图会以 warning 形式报告；运行时会使用
现有 fallback 行为继续渲染。

低成本加载/渲染冒烟：

```bash
python3 tools/smoke_catalog.py --width 32 --spp 1 --threads 1
```

代表场景：

```bash
./build/CGAssignment4 23 4 --bench --width 300 --spp 8 --runs 1 --seed 123 --threads 1
./build/CGAssignment4 --scene-file assets/scenes/scene_023_mis_comparison_scene.json --integrator 4 --bench --width 300 --spp 8 --runs 1 --seed 123 --threads 1
```
