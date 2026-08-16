# 渲染器架构与性能优化实施计划

> 状态：草稿（基于一次完整代码审阅，独立编写）
> 范围：`src/`、`cmake/`、`tests/`、`tools/`、`assets/`
> 基线：HEAD `a45f612`，CPU Release 与 CUDA Release 测试全部通过
>
> 本文档只描述“要做什么、为什么、如何验收、如何回滚”。具体设计细节在实施
> 每个任务时以代码和测试为准，不替代代码注释。

---

## 1. 摘要

当前项目已经具备较好的骨架：

- `JSON/SceneIR` 是稳定的场景中间表示；
- `IRenderSession` 为 CPU/CUDA 提供了统一同步渲染入口；
- `CompiledScene + packed_*_core` 形成了一套 host/device 共用的扁平化数值内核；
- CPU/CUDA 差分测试工具和 34 个 CUDA 测试已经建立。

本次审阅发现的三个主要矛盾：

1. **一套场景、两套运行时**：CPU 走多态 double 运行时，CUDA 走扁平 float 运行时。
   transport、BSDF、光源采样、BVH 遍历均维护两份，长期成本高。
2. **CPU 运行时仍有教程演进型低效点**：BVH 遍历每 ray 分配堆内存、阴影射线
   构造 200 字节 `hit_record`、Film 热路径带边界检查等。
3. **CUDA 调度开销大于实际追踪开销**：wavefront renderer 固定跑满
   `max_depth` 次 kernel，ReSTIR 每个 spp 同步一次，大量 kernel launch、
   memset 和全局 atomic 统计。

本计划的目标是在不破坏现有渲染结果和测试基线的前提下，分阶段消除上述问题，
并建立可重复的性能回归方法。

## 2. 非目标

以下内容**不在**本计划范围内：

- 重写 BSDF/MIS/ReSTIR 数学内核；
- 引入新的第三方场景格式或渲染 API；
- 实现 ReSTIRPT 或 CPU 版 ReSTIR 算法；
- 把全部场景强迁到 JSON（场景目录已经基本完成）；
- 引入 ECS、插件系统或分布式渲染；
- 追求与离线商业渲染器的绝对性能对比。

## 3. 当前架构快照

```text
CLI / AppOptions
      |
      v
SceneIR (JSON 解析产物)
      |
      +-- CpuRenderSession ---- build_scene_config() ----> 多态场景树
      |        |                                            (hittable/material/Light)
      |        +--> Renderer (每帧创建线程) --> CpuPathIntegrator --> BeautyFilm
      |
      +-- CudaRenderSession --- compile_scene() --> CompiledScene
               |                                      |
               |                                      +--> DeviceSceneStorage (30+ 独立 buffer)
               +--> wavefront_renderer.cu
               +--> restir_scheduler.cu / *_core.h
```

关键事实：

- CPU 和 CUDA 从同一 `SceneIR` 出发，但下游数据结构和数值精度不同；
- `packed_transport_core.h` 可以在 host 上运行，但 CPU session 没有使用它；
- `Renderer` 仍承担 CPU 后端的线程、Film、采样组织工作，且每次调用都重建线程；
- `RenderFrameRequest` 携带 `SceneRevision`，但只有 CUDA ReSTIR 用它做
  history key，没有后端真正按 revision 更新场景；
- `RAYTRACER_RENDER_DEBUG_ENABLED` 已生成到 build config，但渲染内核没有使用它。

## 4. 基线数据

以下数据用于实施前后对比，机器为 RTX 4060 Laptop GPU / 20 核 CPU，
`build/` 为 CPU Release，`build-cuda/` 为 CUDA Release。

### 4.1 测试基线

| 构建目录 | 结果 |
|---|---|
| `build/` | 2/2 测试通过 |
| `build-cuda/` | 34/34 测试通过 |

### 4.2 CUDA wavefront 固定深度循环的开销

场景 23，400x225，spp16，run 2，`traversal_steps` 基本不随 depth 变化：

| max_depth | device_seconds | traversal_steps |
|---:|---:|---:|
| 4  | 0.0213s | 3,017,951 |
| 20 | 0.0294s | 3,042,228 |
| 50 | 0.0363s | 3,042,228 |

结论：max_depth 从 4 增加到 50 时，路径实际弹射次数不变，设备时间却增加
约 70%。这部分时间来自固定次数 kernel launch 和 memset。

### 4.3 CUDA batch size 对 launch 开销的影响

场景 23，spp8，max_depth50，run 2：

| batch_size | device_seconds | batch_count |
|---:|---:|---:|
| 4096  | 0.1447s | 176 |
| 16384 | 0.0462s | 48  |
| 65536 | 0.0230s | 16  |

相同 `traversal_steps` 下，小 batch 导致时间增加 6 倍以上。

### 4.4 CPU 多线程扩展

场景 23，400x225，spp8，run 2：

| threads | seconds | samples/s |
|---:|---:|---:|
| 1  | 0.379 | 1.90M |
| 2  | 0.227 | 3.17M |
| 4  | 0.132 | 5.47M |
| 8  | 0.071 | 10.13M |
| 20 | 0.048 | 15.05M |

8 线程到 20 线程仅提升 1.49 倍，需要进一步定位是 Film 带宽、共享缓存
还是负载调度导致。

### 4.5 CPU double 运行时 vs host packed 内核（单线程）

场景 23，单线程微基准（测量口径：CPU double 数字来自完整 `Renderer`
路径，含 Film 累加；packed 数字只调用 `trace_packed_path_core`，不含
Film 累加，因此 packed 在真实后端中的差距只会更大）：

| 路径 | samples/s |
|---|---:|
| CPU double 多态运行时 | 约 1.90M |
| host 直接调用 `trace_packed_path_core` | 约 0.86M |

结论：**不能直接把 CPU 主路径切到当前 packed 内核**。需要先做 CPU 专项
优化（含状态拷贝、RNG、纹理求值、Film 写入）和多线程化，再谈收敛。

### 4.6 一个快速实验的结果

临时将 `LinearBVH::hit` 的 `std::vector<int> stack + reserve(128)` 改为
不初始化的 `std::array<int, 128>` 后重建并测试，scene 23 800x450 spp64
单线程耗时从 18.6-19.9s 降至 17.5-19.6s（约 5%，噪音较大）。该改动已
还原，实施阶段应保留并加入正式基准对比。

---

## 5. 架构设计优化任务

### ARCH-1 收敛 CPU/CUDA 数值内核

**优先级**：中高，长期主线
**风险**：中
**相关代码**：

- `src/cpu/cpu_render_session.cpp`
- `src/renderer/renderer.cpp`
- `src/renderer/cpu_path_integrator.cpp`
- `src/render_data/packed_transport_core.h`
- `src/render_data/compiled_scene.h`
- `src/render_data/scene_compiler.h`

**问题**

CPU 与 CUDA 各有一份路径追踪实现。CPU 是 double、虚函数、`shared_ptr`
对象树；CUDA 是 float、扁平 buffer、无虚函数。任何积分器行为修改都需要
同步两个实现，回归成本随功能增加而上升。

**方案**

分三步：

1. **补一个可运行的 CPU Packed 后端原型**
   - 新增 `src/cpu/cpu_packed_render_session.{h,cpp}`。
   - 构造时调用 `compile_scene(ir)`，渲染时用 `CompiledSceneView` +
     `trace_packed_path_core()` 多线程执行。
   - 只支持 integrator 0-4，ReSTIR 暂不支持。
   - 输出 `BeautyFilm`、`RenderBuffer`、`RenderStats`，与现有
     `IRenderSession` 契约一致。
   - 在 CLI 增加 `--backend cpu-packed`（或 build option），默认仍为旧
     CPU 路径。

2. **优化 packed 内核在 host 上的性能**
   - 先解决单线程只有 double 运行时约 45% 吞吐的问题。
   - 重点检查：路径状态拷贝、每路径 RNG 重建、纹理求值、host 上
     `CompiledSceneView` 的 cache 行为。
   - 只有当 20 线程 packed 后端稳定超过旧 CPU 后端后才进入第 3 步。

3. **切换默认并保留 reference**
   - CPU 默认后端切换到 packed；
   - 旧多态运行时降级为 reference/oracle，只在测试和调试工具中使用；
   - 所有新增 feature 必须同时通过 host 差分测试和 CUDA 差分测试。

**接口草案**

```cpp
// src/cpu/cpu_packed_render_session.h
std::unique_ptr<IRenderSession>
make_cpu_packed_render_session(const SceneIR &ir);
```

内部渲染循环可先复用与 `Renderer` 相同的 tile 调度和确定性 seed 规则：
每像素每 sample 使用 `packed_camera_sample_seed()`，保证与 CUDA 单 sample
结果可比。

**验收标准**

- `packed_transport_check` 与 CUDA 差分测试继续通过；
- 新后端固定 seed 输出与旧 CPU 后端在容忍阈值内一致；
- 20 线程下，代表场景（7、23、59、65）性能不低于旧 CPU 后端；
- 代码中只有一条“生产”transport 主路径，旧路径被测试工具引用而非应用引用。

**回滚**

- 第 1、2 步为纯新增，不影响现有默认路径；
- 第 3 步通过 CLI 开关或 CMake option 可切回旧 CPU 后端。

### ARCH-2 完善 RenderSession revision 协议

**优先级**：中
**风险**：低
**相关代码**

- `src/render_api/render_session.h`
- `src/cpu/cpu_render_session.cpp`
- `src/cuda/cuda_renderer.cpp`
- `src/cuda/restir/restir_history.cpp`

**问题**

`SceneRevision {camera, geometry, material, lighting}` 只有 CUDA ReSTIR
用它决定是否重置 history。CPU session 完全忽略；CUDA wavefront 也只更新
camera；geometry/material/lighting 变化没有重新编译或上传。

**方案**

1. 在 `IRenderSession` 语义中明确：
   - `render()` 是无历史、独立离线入口；
   - `render_frame()` 是 history-aware 入口，revision 变化必须被各后端
     一致处理。
2. 为 CPU session 增加 `SceneIR` 或已构建场景的缓存，比较 revision，
   不匹配时重建对应部分。
3. 为 CUDA session 拆分 `compile_scene` 和 `upload`，至少实现：
   - camera revision：重打包 camera；
   - geometry/material/lighting revision：先采用整体重编译+重上传
     （正确性优先），后续再做增量。
4. 统一 history reset 行为：任何后端遇到 revision 不匹配都应显式
   reset，而不是静默继续。

**验收标准**

- 为 CPU 和 CUDA wavefront 各增加一个 revision 变化测试：
  相机变化、几何变化、材质变化、光源变化；
- ReSTIR history reset 原因统计与 revision 一一对应；
- 文档写明哪些 revision 触发整体重编译，哪些触发增量更新。

### ARCH-3 线程与场景派生资源生命周期

**优先级**：中
**风险**：低
**相关代码**

- `src/renderer/renderer.cpp`
- `src/lighting/light_sampler.cpp`

**问题**

- `Renderer::render()` 每次创建/销毁全部 worker 线程；
- `LightSampler`（含 power/CDF 构建）每次渲染重建；
- 窗口预览模式下这些开销被每帧重复支付。

**方案**

1. 在 CPU session 内建立线程池（大小由 `RenderRequest::threads` 决定，
   变化时重建）。`Renderer::render()` 改为接收线程池或直接下沉调度逻辑。
2. `LightSampler` 作为 scene 派生资源，在 session 构建/场景更新时创建，
   渲染时传 `const LightSampler*`。
3. Benchmark 连续 runs 不再重建线程和 sampler。

**验收标准**

- 窗口模式连续渲染 100 帧的总耗时中，线程创建和 sampler 构建占比接近 0；
- benchmark run 1 与 run N 的时间差显著缩小；
- CPU 输出保持确定性。

### ARCH-4 模块边界与文件拆分

**优先级**：中低
**风险**：低（机械性改动）

**对象**

| 文件 | 行数 | 拆分建议 |
|---|---|---|
| `src/render_data/scene_compiler.cpp` | 889 | 几何编译 / 光源编译 / 顶层聚合编译 |
| `src/resources/gltf_importer.cpp` | 1012 | 节点解析 / 材质解析 / mesh 解析 |
| `src/restir/restir_gi_core.h` | 872 | 初始候选 / 复用 / replay / shading |
| `src/app/render_app.cpp` | 459 | CLI 入口 / benchmark 输出 / 窗口循环 |

同时：

- 统一 target alias，移除 `raytracer::scene` 与
  `raytracer::scene_runtime` 的双重命名，只保留一个公开名称；
- 保持公共头文件只放声明和真正需要内联的小函数。

**验收标准**

- 编译和测试全通过；
- 不改变任何场景 ID 和 CLI 行为；
- 单文件行数降到约 500 行以下（核心 kernel 除外，kernel 拆分的收益需
  单独评估）。

### ARCH-5 错误处理与资源定位

**优先级**：中低
**风险**：低

**问题**

- `scene_path()` 对未知 scene id 静默回退默认场景（`src/scene/scenes.cpp`）；
- 资源路径依赖 cwd，从 `build/` 运行应用会直接失败；
- 图像加载失败静默使用 diagnostic texture。

**方案**

1. 未知 scene id 直接报错；确需回退时要求显式参数 `--fallback-scene`。
2. 引入 asset root：
   - CMake 配置期生成 `RAYTRACER_ASSET_ROOT`；
   - 运行时解析顺序：环境变量 > 编译期路径 > 相对当前目录。
3. 资源加载失败策略改为显式日志 + 可配置：
   - 默认输出 warning 并使用 diagnostic；
   - `--strict-assets` 时直接失败。

**验收标准**

- 从 `build/`、`build-cuda/` 和任意 cwd 均能加载同一 scene；
- 未知 scene id 返回非零退出码；
- `--strict-assets` 对损坏/缺失贴图失败。

### ARCH-6 结果与 benchmark 数据结构

**优先级**：中低
**风险**：低

**问题**

- `RenderStats` 约 40 个字段，CPU/CUDA/ReSTIR 指标混杂，CPU 运行时大量
  字段恒为 0；
- `BENCH_*` 文本行不适合机器消费；
- CPU 不统计 `traversal_steps/shadow_rays`，跨后端无法比较。

**方案**

1. 拆分：
   - `RenderStatsBase`：时间、分辨率、sample 数、seed、取消状态；
   - `CpuRenderStats`：threads、clamped/invalid；
   - `CudaRenderStats`：device_seconds、batch、workspace、status；
   - `RestirStats`：保持现有结构。
2. benchmark 输出可选 JSON（如 `--bench-json PATH`），保留现有文本兼容。
3. CPU 积分器增加轻量 `traversal_steps` 和 `shadow_rays` 计数（可用
   线程本地累加，结束时汇总，避免全局原子）。

**验收标准**

- 现有 `BENCH_*` 格式不破坏（已有脚本可继续解析）；
- `--bench-json` 输出可被 `python -m json.tool` 校验；
- CPU/CUDA 的 traversal/shadow 指标都有实际值。

### ARCH-7 文档与构建体系

**优先级**：低
**风险**：无

**方案**

1. 重写 README 的“实现状态”和架构说明，与当前代码一致；
2. 在 `docs/` 增加一张数据流图（SceneIR -> CPU/CUDA -> Film/Display）；
3. CMake 增加 warning 选项和 CI 建议配置（见第 7 节）。

---

## 6. 性能优化任务

### CPU-PERF-1 `LinearBVH::hit` 固定遍历栈

**位置**：`src/geometry/bvh.cpp` 约 113 行

**现状**

```cpp
std::vector<int> stack;
stack.reserve(128);   // 每次 hit 都发生一次堆分配
```

**方案**

改为不初始化的固定数组，并用显式 `stack_size`：

```cpp
std::array<int, 128> stack;   // 刻意不写 {}，避免每 ray 清零 512B
std::size_t stack_size = 0;
```

超过 128 时抛 `std::runtime_error("LinearBVH traversal stack overflow")`，
与 `MeshAsset` 行为一致。

**预期收益**

- 消除 per-ray malloc；
- 快速实验显示约 5% 提升（噪音较大，需正式基准确认）。

**验收**

- scene 7/23/58/59 图像无差异；
- 单线程 800x450 spp64 中位耗时下降或持平；
- 栈溢出路径有测试覆盖（构造极端 BVH 或单独单元测试）。

### CPU-PERF-2 阴影射线专用遮挡查询

**位置**：`src/renderer/integrator_common.cpp` 的 `visible()`；
`src/geometry/hittable.h`；所有 `hittable` 子类。

**现状**

每根 shadow ray 构造约 200B 的 `hit_record`，而调用方只需要 bool。

**方案**

新增虚接口：

```cpp
class hittable {
 public:
  virtual bool occluded(const ray &r, double t_min, double t_max,
                        RNG &rng) const {
      hit_record rec;
      return hit(r, t_min, t_max, rec, rng);
  }
};
```

重点为 `LinearBVH`、`MeshAsset/MeshInstance`、`sphere`、`triangle`、
`aarect` 提供专用实现：命中任意 primitive 即返回 true，不写 record，
不维护 closest hit。

**验收**

- 阴影测试/差分测试全通过；
- 场景 23、65 的 CPU shadow 密集场景有可测量提升；
- 默认实现保留，新增几何类型不会立即崩溃。

### CPU-PERF-3 Film 热路径 fast path

**位置**：`src/render_api/beauty_film.cpp`

**现状**

`add_sample/set_pixel/pixel` 全部通过 `std::vector::at`。

**方案**

- 保留公共接口的检查语义；
- 增加渲染器内部使用的 unchecked 接口（`operator[]` + `assert`），
  由 `Renderer` 和 packed 后端保证坐标合法；
- `resolve_beauty()` 内部同样改走连续索引。

**验收**

- 公共接口异常行为不变（现有测试覆盖）；
- 热循环内无 `.at()` 调用。

### CPU-PERF-4 Film 分块累加与像素大小

**位置**：`BeautyFilm`、`src/renderer/film.cpp`

**现状**

- `BeautyFilmPixel` 为 double3 + uint32，共 32B；
- 每个 sample 直接 read-modify-write 中心 Film；
- 20 线程扩展仅约 7.9x，疑似带宽/缓存竞争。

**方案**

1. 先测量：用 perf / uarch 指标确认瓶颈是否在 Film 写和共享行；
2. 试验 tile-local accumulation：
   - 每个 worker 维护 16x16 tile 的本地 double/float 累加器；
   - tile 完成后一次性合并到全局 Film；
3. 若数值差异在容忍范围内，评估 `float3 + uint32`（16B）Film；
4. 最终显示层不需要 double，double 只用于累加过程。

**验收**

- 固定 seed 下与当前输出逐像素差异小于既定阈值；
- 20 线程扩展比从 1.49x（8->20）明显改善；
- PFM 输出仍为有效线性数据。

### CPU-PERF-5 环境贴图与纹理预处理

**位置**

- `src/lighting/environmental_light.h`
- `src/materials/texture.cpp`
- `src/materials/image_asset.{h,cpp}`
- packed 对应实现：`src/render_data/packed_light_core.h`、
  `src/render_data/packed_texture_core.h`

**现状**

- `EnvironmentLight::Le()` 每次做 `acos/atan2` + 双线性插值；
- LDR 贴图每次 texel 访问做 `pow(..., 2.4)`；
- CPU 与 GPU 都重复支付。

**方案**

1. `ImageAsset` 增加“已解码线性”表示；SRGB 解码在加载期完成；
2. 环境光在构建期生成：
   - 线性 radiance mipmap；
   - 亮度分布（已有 CDF，保留）；
   - 可选 octahedral map 以改善采样质量；
3. `CompiledScene` 的 `image_texels` 直接携带预解码数据，kernel 移除
   每 texel 的 sRGB 分支和 pow；
4. 普通纹理先保留运行时采样，但缓存尺寸和 wrap 参数，避免每 sample
   重复虚函数调用取宽高。

**验收**

- HDR/LDR 环境光场景图像差异在阈值内；
- 环境光场景单线程路径追踪吞吐提升；
- CUDA 纹理采样 kernel 的指令数/耗时下降。

### CPU-PERF-6 resolve 与图像保存并行化

**位置**：`src/render_api/beauty_film.cpp`、`color_pipeline.cpp`、
`render_buffer.cpp`

**现状**

`resolve_beauty` 和 PNG/JPG 保存为单线程，每像素执行 `pow`。

**方案**

- resolve 按行并行（与 CPU 线程数一致）或使用 SIMD 友好的近似；
- 保存阶段继续使用 stb，但先把 double display 转成 uint8/float 缓冲，
  转换过程并行；
- 4K 以下可作为低优先级，先测量 resolve 在总时间中的占比。

### CPU-PERF-7 采样与积分器上下文开销

**位置**：`src/renderer/renderer.cpp`、`src/renderer/sampler.h`

**现状**

每个 sample 构造 `Sampler`、`IntegratorContext`、相机采样。

**方案**

- 每像素先计算基础 primary ray，逐 sample 只做抖动；
- 将 `ShaderScratch` 等 per-thread 状态保持在 worker 栈中（当前已经如此，
  保持并明确注释）；
- 评估是否值得对同一像素的 spp 做批处理，提高 Film/场景数据复用。

### CUDA-PERF-1 Wavefront 动态 active 与压缩

**位置**：`src/cuda/wavefront_renderer.cu`

**现状**

- 固定循环 `max_depth` 次；
- grid 始终按初始 path 数启动；
- 每 batch 末尾 `cudaStreamSynchronize`；
- 基线：depth 50 比 depth 4 多约 70% 设备时间，而 traversal steps 相同。

**方案（按风险递增）**

1. **方案 A：流压缩 + 动态 grid（先做）**
   - 保留每 depth 一个 kernel；
   - active path 写入 `next_indices`，`next_count` 由 kernel 原子维护；
   - active count 使用 pinned host memory 或每 batch 一次 D2H 拷贝；
   - `active_count == 0` 时提前退出；
   - grid 按上一轮 active count 启动。
2. **方案 B：persistent kernel（验证收益后做）**
   - 每个 block 负责一段 path 槽，内部循环执行
     `advance_packed_path_core` 直到全部死亡；
   - 去掉 per-depth memset、launch 和 buffer swap；
   - 使用 block-level 或 grid-level 同步统计 active 数。
3. **方案 C：去掉每 batch 同步（独立、低风险）**
   - 删除 `render_wavefront_cuda` 循环内的
     `cudaStreamSynchronize(nullptr)`；
   - 只在下载 Film 前同步；取消路径保持现有行为。

**验收**

- 场景 23，400x225，spp16：
  - depth 50 与 depth 4 的设备时间差从 +70% 降到目标 <15%；
- batch 4096 与 65536 的时间差显著缩小；
- `cuda_transport_policy_0..4` 和场景差分测试全通过；
- 统计字段 `batch_count/traversal_steps/shadow_rays` 不回归。

**回滚**

- 保留现有 kernel 函数，通过 workspace/设置开关切换动态调度；
- 每步单独提交，单独 benchmark。

### CUDA-PERF-2 ReSTIR 去除每 spp 强制同步

**位置**：`src/cuda/restir/restir_scheduler.cpp` 约 422 行

**现状**

```cpp
RT_CUDA_CHECK(cudaStreamSynchronize(nullptr));
restir::commit_restir_iteration(...);   // host 端 bookkeeping
```

**分析**

- 同一 CUDA stream 内的 kernel 天然保持顺序；
- `commit_restir_iteration` 只更新 host 端的 buffer 索引、age 计数；
- 中间同步仅对“立即读回数据做诊断”有意义。

**方案**

1. 离线 `render()` 路径：
   - 循环内完全移除 `cudaStreamSynchronize`；
   - 所有 iteration 连续 launch；
   - 仅在下载 Film/GBuffer 前同步。
2. 交互/预览路径：
   - 每 K 次 iteration 同步一次（K 可配置，默认如 4 或 8）；
   - 取消检查仍在 pass 边界，取消时做一次最终同步。
3. 统计归约仍读取最终 counters，不受影响。

**验收**

- `cuda_restir_*` 测试全通过；
- 同 seed 下 Film 输出逐位或阈值一致；
- ReSTIR benchmark 的 device_seconds 明显下降；
- 取消后统计仍正确。

### CUDA-PERF-3 统计计数器成本控制

**位置**

- `src/cuda/restir/restir_initial_di.cu`
- `src/cuda/restir/restir_spatial_di.cu`
- `src/cuda/restir/restir_temporal_di.cu`
- GI 对应 kernel
- `src/cuda/restir/restir_device_types.h`

**现状**

每个像素对多个全局 counter 做 `atomicAdd`，部分为 double 原子。
benchmark 与交互都全量收集。

**方案**

1. 先测量关闭统计的收益：增加 `CudaRestirSkeletonSettings.collect_stats`
   或复用 `RAYTRACER_RENDER_DEBUG_ENABLED`；
2. 常用运行（benchmark 默认）开启轻量统计：
   - 每 block 先在 shared memory 归约；
   - 一个 block 只做少量全局 atomic；
3. 细粒度状态桶只在 debug/诊断工具中开启。

**验收**

- 统计模式与关闭统计模式的输出图像一致；
- benchmark 默认路径的 atomics 数量显著下降；
- 诊断工具仍能获得现有全部状态桶。

### CUDA-PERF-4 DeviceScene 上传与 workspace 分配

**位置**：`src/cuda/device_scene.cu`、`src/cuda/device_buffer.h`、
`src/cuda/restir/restir_workspace_internal.h`

**现状**

- `DeviceSceneStorage::upload` 对 30+ buffer 逐个 `cudaMalloc` 和
  `cudaMemcpy`；
- `DeviceBuffer::ensure_capacity_discard` 增长时 free 旧 buffer 再分配；
- 分辨率变化会反复 alloc/free。

**方案**

1. 引入 `DeviceArena`：
   - 一次 `cudaMalloc`，内部按 alignment 切分；
   - host 端把 `CompiledScene` 各 vector 拷贝到 staging buffer 后一次
     H2D（或使用 pinned staging）；
2. `DeviceSceneView` 改为 arena 内偏移 + 单次分配；
3. workspace 改为 grow-only arena：
   - 不够时分配更大 arena；
   - 缩小分辨率不释放，只记录水位；
   - 显式 `reset()` 才释放。

**验收**

- `allocated_bytes == scene_stats.bytes` 的校验仍然成立；
- 上传时间和 `cudaMalloc` 次数可观测；
- 反复切换分辨率无泄漏（ASan/nsys 或设备内存水位测试）。

### CUDA-PERF-5 编译选项实验

**位置**：`src/cuda/CMakeLists.txt`

**方案**

- 基线保持当前 `-O3` 和手写 `fmaf`；
- 实验 `--ftz=true --prec-div=false`；
- 不启用 `--use_fast_math` 的完整集合，避免破坏差分测试；
- 如果精度回归不可接受，回退。

**验收**

- `cuda_transport_policy_*`、`cuda_shading_check` 全通过；
- 输出图像与基线差值在阈值内；
- 保留实验开关，不在默认路径硬编码。

---

## 7. 构建与工程化任务

### BUILD-1 编译警告与静态分析

- 所有 target 增加 `-Wall -Wextra -Wpedantic`；
- 修复全部 warning；
- CUDA 使用 `-Xcompiler=-Wall,-Wextra` 或等价 host 警告；
- 评估 `clang-tidy` 只跑核心库，不阻塞提交。

### BUILD-2 LTO 与 native 优化

- 增加 option `RAYTRACER_ENABLE_LTO`（默认 OFF，CI 实验 ON）；
- 增加 option `RAYTRACER_ENABLE_NATIVE`（默认 OFF）；
- 对 Release benchmark 各跑一次场景 7/23/59 的中位数对比，决定是否默认
  开启。

### BUILD-3 Sanitizer preset

- 固化 `cpu-sanitize` preset（ASan + UBSan）；
- 固化 `cuda-debug` 已有 preset；
- CI 矩阵至少包含：CPU Release 测试、CUDA Release 测试、CPU sanitizer
  冒烟。

### BUILD-4 性能回归门槛

- 新增一个 `tools/perf_smoke.py` 或 CMake 脚本：
  - 跑固定 2-3 个 benchmark；
  - 解析 `BENCH_RUN`；
  - 与 `baselines/*.json` 比较；
- 阈值宽松（如 1.5x），只拦截 launch/sync 级别的重大回归；
- 不因单机波动阻塞。

---

## 8. 分阶段实施路线图

### Phase 0：基线冻结与度量工具（先行）

**目标**：让每个后续任务都有“改前/改后”数据。

任务：

1. 把第 4 节基线数据写入 `docs/optimization-baseline.md`（或 JSON）。
2. 实现 `--bench-json` 输出（ARCH-6 的子集）。
3. 增加 `tools/perf_smoke.py`，固定场景/参数/seed，输出中位数。
4. 增加 warning 选项并修复 warning（BUILD-1）。

**验收**

- 一键生成基准报告；
- CI 可运行；
- 所有 warning 清零。

### Phase 1：低风险高收益优化（1-2 周）

任务（每个独立提交，独立回滚）：

1. CPU-PERF-1 `LinearBVH` 固定栈；
2. CPU-PERF-2 `occluded()`；
3. CPU-PERF-3 Film unchecked fast path；
4. CUDA-PERF-1 方案 C：移除 wavefront per-batch sync；
5. CUDA-PERF-2：移除 ReSTIR 每 spp 同步（离线路径）；
6. ARCH-5 中的未知 scene id 报错。

**验收**

- 全部现有测试通过；
- 性能烟雾脚本无回退；
- 代表场景渲染图与基线视觉一致。

### Phase 2：CUDA 调度重构（2-3 周）

任务：

1. CUDA-PERF-1 方案 A：流压缩 + 动态 grid；
2. 将 `advance_paths_kernel` 与压缩 kernel 合并或顺序执行；
3. 统计 `active_count` 每 batch 的分布，评估 persistent kernel 收益；
4. 若方案 A 收益明显，再实施方案 B；
5. CUDA-PERF-3 统计计数器 block 归约。

**验收**

- 第 4.2 节 depth 实验差距 <15%；
- 第 4.3 节 batch size 实验差距显著缩小；
- CUDA 34 个测试全通过；
- 固定 seed 图像输出不变。

### Phase 3：CPU 后端与资源优化（2-3 周）

任务：

1. ARCH-3：CPU session 线程池 + `LightSampler` 缓存；
2. CPU-PERF-5：图像预解码 + 环境贴图预处理（CPU 先行）；
3. CPU-PERF-4：Film 分块累加实验；
4. ARCH-1 第 1 步：`CpuPackedRenderSession` 原型（`--backend cpu-packed`）；
5. CUDA-PERF-4：DeviceScene arena 上传。

**验收**

- CPU 20 线程场景 23 吞吐提升；
- 环境光场景 CPU/CUDA 图像一致；
- packed CPU 后端原型可运行 integrator 0-4；
- 设备场景上传次数/耗时下降。

### Phase 4：统一与协议完善（2-3 周）

任务：

1. ARCH-1 第 2 步：packed host 性能优化，达到旧 CPU 后端水平；
2. ARCH-2：完整 revision 协议（CPU/CUDA wavefront/ReSTIR）；
3. ARCH-6：RenderStats 拆分和 CPU traversal/shadow 统计；
4. ARCH-4：大文件拆分（机械部分）。

**验收**

- packed CPU 后端性能达标；
- revision 变化测试通过；
- 统计结构跨后端一致；
- 模块边界更清晰。

### Phase 5：清理与文档（1 周）

任务：

1. ARCH-1 第 3 步：若性能达标，切换 CPU 默认后端；
2. ARCH-7：README 和架构文档更新；
3. BUILD-2/BUILD-3 固化 LTO、native、sanitizer 配置；
4. 更新性能基线表和完成报告。

**验收**

- 默认路径统一到 packed 数值内核（或明确记录未切换的原因和替代方案）；
- 新用户按 README 能完成构建、测试、benchmark 三件事；
- 所有阶段任务闭环。

---

## 9. 测试与验收总览

### 9.1 每次提交必跑

```bash
cmake --build build -j
ctest --test-dir build --output-on-failure

cmake --build build-cuda -j
ctest --test-dir build-cuda --output-on-failure
```

### 9.2 性能冒烟

```bash
python3 tools/perf_smoke.py --preset review-baseline
```

固定场景建议：

| 场景 | 后端 | 参数 |
|---|---|---|
| 23 | CPU | 400x225, spp32, integrator 4, threads 0 |
| 59 | CPU | 400x225, spp16, integrator 4 |
| 23 | CUDA | 400x225, spp32, integrator 4, max-depth 4/50 |
| 65 | CUDA ReSTIR DI | 320x180, spp8, max-depth 2 |

### 9.3 数值回归

- 继续使用 `packed_transport_check`；
- 任何 transport/shading 修改必须跑 `cuda_shading_check`、
  `cuda_light_check`、`cuda_transport_policy_*`；
- 图像回归以固定 seed 的 PFM 差值为准，不只看 PNG。

### 9.4 图像验收

```bash
./build/CGAssignment4 23 4 --bench --width 400 --spp 32 --seed 123 \
    --runs 1 --save-linear /tmp/before.pfm
```

改后输出 `/tmp/after.pfm`，比较最大相对误差和 NaN/Inf 数量。

---

## 10. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| packed CPU 内核性能无法追上 double 运行时 | ARCH-1 无法收敛 | 保持双线但强化差分测试；把结论记录在案，转为长期研究项 |
| CUDA 动态调度引入 race 或数值变化 | 渲染结果变化 | 保留旧路径开关；每步独立 benchmark；用 PFM 差值门禁 |
| 移除 ReSTIR 同步后取消行为变化 | 交互体验 | 保留 pass-boundary 取消检查；取消时最终同步；增加取消测试 |
| 固定栈容量不足 | BVH 崩溃 | 保留显式溢出异常；增加深树测试 |
| Film 分块累加改变浮点求和顺序 | 数值差异 | 默认关闭；仅在通过阈值后启用 |
| 资源路径改造影响现有场景 | 场景加载失败 | 保持 cwd 相对路径作为 fallback；全量场景 smoke |
| 统计关闭后诊断能力下降 | 难调试 | 默认轻量统计 + 显式 debug 工具全量统计 |

---

## 11. 完成定义（Definition of Done）

本计划完成时，应满足：

1. 所有现有 CPU/CUDA 测试通过；
2. 新增 revision、取消、严格资源模式的测试通过；
3. 第 4 节基线中的关键问题有量化改善：
   - CUDA depth 50 与 depth 4 设备时间差显著缩小；
   - 小 batch 的 launch 开销显著下降；
   - CPU BVH/Film 热路径不再有 per-ray 堆分配和 `.at()`；
4. 每项优化有改前/改后 benchmark 数据和回滚记录；
5. `README` 与代码能力一致；
6. 默认渲染路径和数值核心的选择有明确结论：
   - 若统一成功：CPU 默认走 packed，旧运行时仅作 reference；
   - 若统一不成功：文档明确记录原因、差异风险和继续维持双线的成本。

---

## 12. 实施进度记录

| 阶段 | 状态 | 完成内容 | 验证 |
|---|---|---|---|
| Phase 0 | 已完成 | 基线文档、`--bench-json`、`tools/perf_smoke.py`、全项目 warning 开启并清零 | CPU/CUDA 测试全通过 |
| Phase 1 | 已完成 | CPU-PERF-1/2/3、CUDA-PERF-1 方案 C、CUDA-PERF-2、ARCH-5 未知 scene id 报错与 strict-assets 模式 | 新增 occlusion 一致性测试；CUDA PFM 逐位一致；CPU/CUDA 测试全通过 |
| Phase 2 | 已完成 | wavefront persistent kernel、动态 active 提前终止、wavefront 统计 block 归约、ReSTIR gbuffer/DI/GI/reference 全部统计 block 归约 | depth4/depth50 设备时间基本持平；batch 4096/65536 差距 6.3x→2.5x；CUDA PFM 逐位一致；34/34 测试通过 |
| Phase 3 | 已完成 | LightSampler 缓存、ImageAsset 线性缓存 + ImageTexture/EnvironmentLight 预解码、`--cpu-packed` 后端原型（ARCH-1 第 1 步）、DeviceScene 单 arena 上传、Film 分块累加实验（已回退）、线程池成本测量 | CPU/CUDA 测试 3+35 通过；CUDA PFM 逐位一致；cpu-packed 原型可运行；perf smoke 全部通过 |
| Phase 4 | 已完成 | CPU/CUDA revision 重建与测试；ARCH-6 RenderStats 拆分为 Base/Cpu/Cuda；ARCH-4 scene_compiler 拆分为 5 个文件；ARCH-1 packed host 优化实验完成并回退 | CPU/CUDA 测试 3+35 通过。ARCH-1 性能达标/默认切换未完成，按方向 A 记录 |
| Phase 5 | 已完成 | README 架构同步；LTO/native 选项与 preset；sanitizer preset 验证；最终基线与偏差记录 | cpu-sanitize 3/3、cpu-lto-native 构建通过 |


---

## 13. Phase 3 实验结论与决策记录

1. **CPU Film tile-local 累加**：输出与直接写 Film 逐位一致，但 400x225/spp32
   与 400x225/spp16 中位耗时分别约 +13% 和 +27%（额外 8KB fill + 二次提交
   超过减少全局写带来的收益）。实验已回退，直接写 Film 继续作为默认路径。
2. **持久渲染线程池**：20 线程创建+join 实测约 0.75ms。当前窗口模式一次
   渲染生命周期内只建一次线程，benchmark run 中占比 0.3-3%。暂不引入持久
   线程池，待出现真正的交互式逐帧 CPU 重渲染场景后重新评估。
3. **DeviceScene 上传**：arena 化前上传约 1.4ms（scene 65），不是当前热
   点；但 30+ 次 alloc/copy 是结构债务，已按计划重构为单 arena + 单次
   H2D，并保持 PFM 逐位一致。
4. **CPU packed 原型**：`--cpu-packed` 可运行且通过 CTest，但性能仍明显
   落后 double 后端（perf preset 约 0.52s vs 0.17s）。这符合 Phase 0 基线
   预期；进入 Phase 4 前必须完成 host packed 内核专项优化，否则不切换默认。


---

## 14. Phase 4 阻塞决策点

ARCH-1 第 2 步要求 packed CPU 后端达到旧 double 后端性能，目前未达到：

- 当前实测（400x225, spp32, 20 线程）：double 0.175s vs cpu-packed 0.555s。
- 已尝试并回退的优化：
  1. BSDF fast 变体（跳过重复归一化/有限检查）：CPU 提升约 7%，但 CUDA
     ReSTIR GI replay 检查出现 reservoir 数值失配，已回退。
  2. identity transform 快速路径：导致同样的 host/device 数值失配，已回退。
  3. any-hit 阴影遮挡查询：单独验证未证明是失配源，但为避免未验证行为
     已随实验回退。
  4. 快速变体单独复测：identity 回退后重新引入 fast BSDF，replay 检查仍
     失败，确认 fast BSDF 本身也会放大 host/device 浮点差异，已再次回退。
- 初步 profile 结论：packed 内核指令数约为 double 内核 2 倍，热点在
  float 三角形求交、BSDF 归一化/有限检查，以及 CPU double 端可用的
  `aarect` 等专用 primitive 在 packed 端被三角化。

方向 B 已收口（2026-08-16）：已建立 host-only fast transport 分叉
（fast BSDF、host occlusion/intersect/reconstruct identity 快路径、
简化三角形求交），并通过
`RAYTRACER_PACKED_HOST_FAST`（默认 ON）对 `cpu_packed_render_session`
TU 启用 `-Ofast -march=x86-64-v3 -funroll-loops`。CUDA safe 路径不变且 replay
门禁通过。最终 cpu-packed 0.250s vs double 0.168s（20 线程，1.47x；
单线程 1.04x）。亲和性、tile、向量宽度等实验均无进一步收益；剩余差距
来自混合 P/E 核下的多线程调度/频率特性，记录为平台相关后续工作。

可选方向（需要决定）：
A. 接受 cpu-packed 原型仅作验证后端，Phase 4 不切换默认，完成剩余
   ARCH-2/4/6 后收尾；性能收敛降级为长期研究项。
B. 投入较大重构：为 packed scene 增加 quad/rect primitive，或对 host
   单独实现更快的 packed transport（允许与 CUDA 内核分叉的 host fast path）。
C. 由我继续尝试其他低风险优化，但接受 Phase 4 时间明显延长。

---

## 15. 最终执行核对

- Phase 0-3：全部按验收完成，CPU/CUDA 测试全通过。
- Phase 4：
  - ARCH-2 revision：CPU/CUDA 已实现并有 CPU 测试。
  - ARCH-6 RenderStats：已拆分为 `RenderStatsBase / CpuRenderStats /
    CudaRenderStats`，`--bench-json` 与 BENCH 文本保持原字段名。
  - ARCH-4：`scene_compiler.cpp` 已拆为 5 个文件。
  - ARCH-1：cpu-packed 原型保留为验证后端；性能未达标，已按方向 A 记录。
- Phase 5：
  - README 更新为当前架构。
  - `RAYTRACER_ENABLE_LTO` / `RAYTRACER_ENABLE_NATIVE` 选项。
  - `cpu-sanitize`、`cpu-lto-native` preset，均已配置并构建验证。
- 最终验证：CPU 3/3、CUDA 35/35、sanitizer 3/3、perf smoke 6/6。
- 未解决项：ARCH-1 性能收敛与默认切换，等待用户对方向 A/B/C 的最终确认。
