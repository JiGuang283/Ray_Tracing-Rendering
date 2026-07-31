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
- `SceneIR`：camera/render 配置以及强类型 Texture、Material、Object、Light
  variant；递归 object 会降低为 ID 引用图。
- `SceneConfig`：由 scene builder 构建出的运行时 camera、world、lights、render preset。

Texture IR 会把内联节点降成 ID 图并检查未知引用和循环引用。构建阶段再根据
`Color`、`Scalar`、`Normal` 输入语义生成不可变纹理和材质实例。

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

`image` 节点支持：

```json
{
  "type": "image",
  "path": "textures/base_color.png",
  "color_space": "srgb",
  "channel": "rgb",
  "wrap_u": "repeat",
  "wrap_v": "repeat",
  "filter": "bilinear"
}
```

- `color_space`：`srgb` 或 `linear`
- `channel`：`rgb`、`r`、`g`、`b`、`a`
- `wrap_u` / `wrap_v`：`repeat`、`clamp`、`mirror`
- `filter`：`nearest`、`bilinear`

旧文件省略 `color_space` 时仍可加载：颜色输入按 sRGB，标量和法线输入按
linear；validator 会提示补全。图像缺失时运行时只报告一次，并使用共享的
洋红/黑诊断纹理。

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
  "normal_map": {
    "texture": {
      "type": "image",
      "path": "normal.png",
      "color_space": "linear"
    },
    "convention": "opengl",
    "strength": 1.0
  },
  "clearcoat": 0.25,
  "clearcoat_roughness": 0.08,
  "emission": [1.0, 0.8, 0.4],
  "emission_strength": 2.0
}
```

材质会编译为不可变 `MaterialInstance`。共享的 `MaterialProgram` 读取索引参数
块并生成 `MaterialOutput`；integrator 只读取 emission 和 BSDF 的
`sample/eval/pdf`，不依赖具体材质类型。`normal` 与 `normal_texture` 仍可兼容，
新场景使用 `normal_map`。

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
`sphere`、`obj` 和 `model` mesh，并支持 `transform` / `translate` /
`rotate_y` / `flip_face` / `list` / `accel` 包裹。Mesh emitter 引用共享
`MeshInstance`，在采样点插值 UV/顶点色并求值 emission 纹理，不复制一份几何。
`flip_face` 会翻转矩形、三角形和 mesh emitter 的发光朝向；flip sphere 不生成
inward sphere emitter。

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
- `model`
- `transform`
- `translate`
- `rotate_y`
- `flip_face`
- `constant_medium`
- `list`
- `accel`

`model` 用于加载 glTF 2.0 `.gltf` / `.glb`：

```json
{
  "type": "model",
  "path": "assets/models/ToyCar/ToyCar.glb",
  "scene": 0,
  "transform": {
    "translation": [0.0, 0.5, 0.0],
    "rotation": [0.0, 0.0, 0.0, 1.0],
    "scale": [30.0, 30.0, 30.0]
  },
  "material_overrides": {
    "Glass": "glass_override"
  }
}
```

- `scene` 可省略，默认使用 glTF 的 default scene。
- `transform.rotation` 是 `[x, y, z, w]` quaternion；也可使用 16 元素
  column-major `matrix`。
- `material_overrides` 按 glTF material name 绑定场景中的 JSON 材质。
- mesh、image 和 model 都由 `ResourceRegistry` 按规范化路径缓存；多个 node 或
  多个 model object 共享不可变几何与图像资源。
- 导入器支持 triangle/triangle strip/triangle fan、interleaved/normalized/sparse
  accessor、POSITION/NORMAL/TANGENT/TEXCOORD_0/COLOR_0、节点层级、多 primitive、
  多材质、嵌入或外部图像、metallic-roughness、normal、emissive、clearcoat 和
  `KHR_texture_transform`。缺失法线和切线会分别生成，切线使用 MikkTSpace。
- 当前不支持 skin、morph animation、Draco、meshopt、GPU instancing、多个 UV set、
  alpha masking/blending，以及 transmission/sheen/volume 等尚未进入 BSDF 的扩展。
  对会改变画面语义的扩展会给出 warning 或明确加载错误，不静默伪装为完整支持。

`transform` 是通用仿射包装，可作用于任意 object：

```json
{
  "type": "transform",
  "transform": {
    "translation": [1.0, 0.0, 0.0],
    "scale": [2.0, 1.0, 0.5]
  },
  "object": {"type": "sphere", "center": [0, 0, 0],
             "radius": 1, "material": "paint"}
}
```

Light：

- `point`
- `directional`
- `spot`
- `quad`
- `environment`

## 已移除路径

- `obj.implementation`
  - 已移除。OBJ 统一导入为共享 `MeshAsset`，由 `MeshInstance` 绑定材质和变换。
- `random_scene_generator`
  - 已移除。scene 1/6 已展开为显式 JSON。
- `final_scene_generator`
  - 已移除。scene 9/22 已展开为显式 JSON。
- `SceneRegistry`
  - 已移除。`select_scene(id)` 直接读取 JSON catalog。
- `bvh_node`
  - 已移除。顶层加速结构使用 `LinearBVH`。
- `mesh`
  - 旧多态 mesh 已移除。OBJ/glTF 使用 `MeshAsset + MeshInstance`。
- `material::scatter`
  - 已移除。renderer 调用 `MaterialProgram::evaluate(...)` 获取
    `MaterialOutput`，再只通过 `BSDF::sample/eval/pdf` 完成散射计算。

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

当前缺失的 `earthmap.jpg` 和若干 HDR 环境贴图会以 warning 形式报告；运行时会
使用缓存的诊断图像继续渲染。

低成本加载/渲染冒烟：

```bash
python3 tools/smoke_catalog.py --width 32 --spp 1 --threads 1
```

代表场景：

```bash
./build/CGAssignment4 23 4 --bench --width 300 --spp 8 --runs 1 --seed 123 --threads 1
./build/CGAssignment4 --scene-file assets/scenes/scene_023_mis_comparison_scene.json --integrator 4 --bench --width 300 --spp 8 --runs 1 --seed 123 --threads 1
./build/CGAssignment4 63 4 --bench --width 640 --spp 32 --runs 1 --seed 123
```
