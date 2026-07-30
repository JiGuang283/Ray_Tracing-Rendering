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

运行时加载分三步：

- `SceneDescription`：外部 JSON 描述和来源路径。
- `SceneIR`：强类型 camera/render/resource/object/light 描述；资源仍保留原
  JSON payload 以兼容现有 schema。
- `SceneConfig`：由 scene builder 构建出的运行时 camera、world、lights、render preset。

这条边界用于避免文件 IO、JSON schema 和运行时对象构建继续混在一个层里。

## Scene 文件结构

顶层字段：

```json
{
  "name": "scene_name",
  "world_accel": true,
  "auto_emitters": true,
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
- `exposure`
- `gamma`
- `tone_mapping`：`linear`、`reinhard`、`aces`
- `color_pipeline`：可选嵌套对象，支持同样的 `exposure`、`gamma`、
  `tone_mapping` 字段

## 支持的资源类型

Texture：

- `solid`
- `checker`
- `noise`
- `image`
- `scale`
- `multiply`
- `mix`
- `color_ramp`

Material：

- `lambertian`
- `metal`
- `dielectric`
- `diffuse_light`
- `principled`
- `pbr`

`pbr` 是兼容旧场景的别名，内部映射到 `principled`。新场景推荐使用
`principled`：

```json
{
  "type": "principled",
  "base_color": [0.8, 0.2, 0.1],
  "roughness": 0.4,
  "metallic": 0.0,
  "normal": {"type": "image", "path": "normal.png"},
  "clearcoat": 0.25,
  "clearcoat_roughness": 0.08,
  "emission": [1.0, 0.8, 0.4],
  "emission_strength": 2.0
}
```

材质在运行时作为 surface shader：它读取 texture 参数并生成 `ShadingResult`。
`ShadingResult` 中包含 BSDF closure 和 emission；integrator 只通过 BSDF 的
`sample/eval/pdf` 与材质交互，不直接读取具体材质类型。

Texture 可以作为轻量 shader graph 节点嵌套使用，例如：

```json
{
  "type": "mix",
  "a": [0.1, 0.1, 0.1],
  "b": {"type": "noise", "scale": 3.0},
  "factor": 0.35
}
```

`auto_emitters` 控制是否把 emissive geometry 自动加入 light sampler。默认规则：
没有显式 `lights` 时开启，有显式 `lights` 时关闭，以避免旧场景重复采样同一盏灯。
当前自动 emitter 支持 `xy_rect`、`xz_rect`、`yz_rect`、`quad`、`triangle`、
`sphere`、`obj` mesh，并支持 `translate` / `rotate_y` / `flip_face` / `list` /
`accel` 包裹。`flip_face` 会翻转矩形、三角形和 mesh emitter 的发光朝向；
flip sphere 不生成 inward sphere emitter。

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
  - 已移除。renderer 调用 `Material::shade(...)` 获取 `ShadingResult`，再只通过
    `BSDF::sample/eval/pdf` 完成散射计算。

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
