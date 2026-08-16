# CUDA 后端进一步优化实施计划

> 状态：草稿
> 前置状态：Phase 0-5 架构/性能计划已完成并提交；CUDA 35/35 测试通过，
> ReSTIR replay 数值门禁通过。
> 本文档只覆盖 CUDA 后端后续优化，不涉及 CPU packed 后端。

## 1. 摘要

当前 CUDA 后端已经从“调度开销远大于追踪开销”修复为“调度开销可控、单
kernel 效率成为主要矛盾”。下一步的核心方向是：

1. **减少 kernel launch 数量**：wavefront 每个 sample 仍 launch 一次；
   ReSTIR 每个 spp 仍 launch 8-12 个 stage。
2. **减少路径状态访存**：当前每 bounce 将 112B `PackedPathState` 从
   global 读回并写回，是稳定的带宽浪费。
3. **改善 ReSTIR stage 编排**：用 CUDA Graph 和 stage 融合替代显式
   launch 序列。
4. **压缩场景数据**：BVH 节点、纹理、增量上传仍以未压缩/全量为主。

## 2. 当前基线

机器：RTX 4060 Laptop（8GB），driver 允许运行但性能计数器未授权
（`ncu` 返回 `ERR_NVGPUCTRPERM`；`nsys` 可用作 timeline 分析）。

### 2.1 Wavefront

场景 23，400x225，spp8：

| 配置 | device 中位 |
|---|---:|
| max_depth 4 | 0.0199s |
| max_depth 50 | 0.0227s |
| batch 4096 | 0.0296s |
| batch 65536 | 0.0120s |

小图基线（64x36，spp1）：

- `batch_count = 1`
- `wavefront_advance_launches = 75`（即 75 个 block-depth 迭代）
- `active_path_steps = 4256`
- `traversal_steps = 4768`
- `shadow_rays = 1756`

### 2.2 ReSTIR DI

场景 65，320x180，spp8：

- device 中位约 0.0141s；
- 每个 iteration 约 8-12 个 kernel（gbuffer、initial、temporal、spatial、
  shading、fallback、reference）。

## 3. 目标与非目标

### 目标

- 将 wavefront 的 launch 数降低 8-16 倍；
- 将 ReSTIR 单 iteration 的 host launch 开销降到接近单次 graph exec；
- 保持所有现有数值门禁：CUDA 35/35、replay 门禁、PFM 逐位/阈值一致；
- 每一步可独立 benchmark 和回滚。

### 非目标

- 不修改 ReSTIR 数学/偏差校正语义；
- 不引入 OptiX/硬件 BVH；
- 不做多 GPU；
- 不追求 CPU packed 后端性能。

## 4. Wavefront 优化任务

### W1：单 kernel 处理多个 sample

**位置**：`src/cuda/wavefront_renderer.cu`、`wavefront_renderer.h`

**现状**

外层 `for sample`，内层 `for batch`，每个 sample/batch 调用一次
`render_paths_kernel`。全帧 4K + spp64 时约有 2048 次 launch。

**方案**

- 新增 `CudaRenderSettings::samples_per_launch`，默认 8，自动 clamp 到
  `samples_per_pixel`；
- 外层只按 pixel batch 循环，每个 path slot 连续处理
  `samples_per_launch` 个 sample；
- sample 内部使用独立 RNG seed，film 写入保持串行，因此**不需要 atomic**；
- 每个 path slot 处理完一组 sample 后可以复用或直接重置。

伪代码：

```cpp
for (pixel_offset = 0; pixel_offset < pixel_count;
     pixel_offset += batch_size) {
    launch_render_paths_batch(
        scene, settings,
        pixel_offset, count,
        sample_begin, sample_end);  // 一次 launch
}
```

**验收**

- `batch_count == ceil(pixel_count / batch_size)`，不再乘以 spp；
- 64x36、spp64 的 wall/device 时间显著下降；
- 与单 sample 版本的 PFM 输出**逐位一致**；
- `cuda_transport_policy_0..4` 全通过。

**回滚**

- 保留 `samples_per_launch = 1` 的旧路径开关。

### W2：固定 persistent grid + 全局 work stealing

**位置**：`wavefront_renderer.cu`

**现状**

grid 随 batch 变化，block 之间不能交换路径，先结束的 block 会空闲。

**方案**

- 启动固定 grid，例如
  `SM_count * blocks_per_sm`（可由 occupancy API 计算）；
- 设备端维护 `next_pixel_block` 原子计数器；
- 每个 block 循环获取 32/64 个 pixel 的 chunk；
- 每个 chunk 内部完成 W1 的 sample 批次；
- 小 batch 不再依赖 host 选择 grid 大小。

**验收**

- batch 4096 与 65536 的设备时间差进一步缩小，目标 <1.5x；
- 大场景无 tail idle 恶化；
- PFM 逐位一致。

### W3：路径状态减少 global 往返

**位置**：`packed_types.h`、`render_paths_kernel`

**现状**

每 bounce：

```cpp
PackedPathState state = states[path_index];  // 112B load
advance(...);
states[path_index] = state;                  // 112B store
```

**方案**

阶段 A：将 active 路径的 `PackedPathState` 放入 shared memory 数组
（`block_size * sizeof(PackedPathState)`），路径死亡才写回 global。shared
占用约 14KB/block，需要评估 occupancy。

阶段 B：若 A 的 occupancy 损失过大，改为 SoA：
`ray[] / throughput[] / radiance[] / rng_state[]` 分数组存放，只访问
实际变化的字段。

**验收**

- cachegrind/nsys 显示 global load/store 明显下降；
- 128 block size 下 occupancy 不低于当前 50% 太多；
- 固定 seed 输出逐位一致。

### W4：block size 与 launch 配置自动调优

**位置**：`wavefront_renderer.cu`

**方案**

- 增加 `--cuda-block-size` CLI 或运行时 heuristic；
- 首次渲染可选 `autotune = true`，对 128/192/256/512 各跑 1 spp 小图，
  选择最优 block size；
- 缓存到 `CudaRenderWorkspace`。

**验收**

- 不改变输出；
- 代表场景有可复现的最优 block size 选择。

### W5：场景描述符与 `__restrict__`

**位置**：`compiled_scene.h`、所有 kernel 签名

**方案**

- kernel 只接收一个 `const DeviceSceneView *`；
- 数组访问指针加 `const T *__restrict__`；
- 将 camera/background/count 等热标量放入 `__constant__`。

**验收**

- 编译与测试通过；
- 无明显性能回退。

## 5. ReSTIR 优化任务

### R1：CUDA Graph 捕获 iteration 模板

**位置**：`restir_scheduler.cpp`、新增 `restir_graph.{h,cpp}`

**现状**

每个 spp 在 host 上显式 launch 8-12 个 kernel，参数几乎不变，只有
iteration/seed/buffer 索引变化。

**方案**

1. 以“一个完整 DI/GI iteration”为单位 capture graph；
2. 可变参数放入 device parameter buffer：
   - iteration
   - seed
   - gbuffer/reservoir/film buffer 指针
   - history 相关指针
3. 每次 exec 前 host 只做一次小 H2D 更新参数 buffer；
4. 离线渲染多个 spp 时循环 exec graph；
5. 分辨率/设置变化时重建 graph。

**验收**

- ReSTIR DI 的 host-side total time 明显下降；
- `cuda_restir_*` 测试与 replay 门禁全通过；
- 取消语义保持：pass 边界或 iteration 边界可取消。

**风险**

- graph 更新接口在不同 CUDA 版本差异较大；
- 初始实现先支持固定拓扑 + 参数更新，不支持动态拓扑。

### R2：融合小 stage

**方案**

- 融合 `gbuffer + initial DI candidates`；
- 融合 `fallback shading + reference shading`；
- 评估融合 `initial DI shading` 与最终 DI shading。

**验收**

- 单 iteration kernel 数下降；
- 各统计桶与未融合版本一致；
- PFM 逐位一致。

### R3：空间复用 tiling

**位置**：`restir_spatial_di.cu`、`restir_spatial_gi.cu`

**方案**

- 以 8x8 像素 tile 为单位，warp 协作预取邻域 reservoir；
- 邻居顺序按 Morton/tile order 重排；
- 先实现 shared memory 预取，再评估重排收益。

**验收**

- spatial reuse 吞吐提升；
- `cuda_restir_spatial_*_check` 和 statistics 测试通过。

### R4：统计等级

**位置**：`restir_device_types.h`、所有 ReSTIR kernels

**方案**

- `CudaRestirSkeletonSettings.collect_stats`：
  - `None`：不更新 counters；
  - `Summary`：sample/visibility/timing 核心指标；
  - `Full`：当前全量桶。

**验收**

- `None/Summary` 下输出与 `Full` 一致；
- 诊断工具仍使用 Full。

### R5：GI replay/final gather 的 wavefront 化

**位置**：`restir_initial_gi.cu`、`restir_gi_core.h`

**方案**

- 将 replay 候选从“每 pixel kernel 内循环”抽取为候选列表；
- 用类似 `render_paths_kernel` 的 persistent 架构推进；
- 仅对 `final_gather` 或高候选数配置启用。

**验收**

- replay 统计字段一致；
- 高 final gather 负载场景性能提升。

## 6. 场景数据与内存任务

### S1：BVH 节点压缩

**位置**：`packed_types.h`、`packed_bvh.cpp`、`flat_intersector_core.h`

**方案**

- 将 `PackedBVHNode` 从 32B 压缩到 24B 或 16B；
- min/max 使用父节点范围 16-bit 量化；
- leaf/internal 继续复用 `first/meta`。

**验收**

- 大 mesh 场景 device scene bytes 下降；
- 遍历时间不回退；
- replay 与场景差分门禁通过。

### S2：纹理 mipmap 与压缩

**位置**：`resource_compiler.cpp`、`packed_texture_core.h`

**方案**

- 编译期为 CUDA 生成 mip chain；
- 远距离采样使用高 mip，减少 4-tap 带宽；
- 评估 half/BC6H 压缩。

**验收**

- 纹理密集场景显存和带宽下降；
- `cuda_shading_check` 通过。

### S3：增量场景上传

**位置**：`device_scene.cu`、`cuda_renderer.cpp`

**方案**

- 按 buffer 维护 dirty bit；
- revision 变化时只上传变化 buffer；
- arena 内支持替换等长 buffer，长度变化时重建 arena。

**验收**

- `render_frame` revision 更新耗时下降；
- 与全量上传输出一致。

### S4：双流 overlap

**方案**

- 渲染使用 stream A；
- 场景上传/参数更新使用 stream B；
- film D2H 与下一帧渲染 overlap。

**验收**

- 交互模式总帧时间下降；
- 取消与同步语义保持。

## 7. 实施进度

- Phase A / W1：已完成。`samples_per_launch` 默认 8，PFM 与旧逐 sample
  launch 逐位一致；CUDA 35/35 通过。
  - 64x36 spp64 CLI 验证：`--cuda-samples-per-launch 16` 时 batch_count=4。
  - depth4/depth50 device 中位 0.0173s / 0.0197s（优化前 0.0199/0.0227）。
  - 新增 CLI：`--cuda-samples-per-launch N`。

## 8. 工具与观测

- `nsys profile` 生成 timeline，验证 launch 数量与 gap；
- `ncu` 需要授权性能计数器；无法授权时使用 nsys 的 kernel 时间与
  cachegrind 近似；
- 每个阶段使用 `--bench-json` 输出记录中位数。

## 8. 实施路线

### Phase A：wavefront 调度再优化（P0）

1. W1 多样本单 kernel；
2. W4 block size autotune；
3. 重跑 PFM 门禁和 batch size 基准。

**验收**：batch_count 不随 spp 增长；64x36 spp64 显著提速；PFM 逐位一致。

### Phase B：ReSTIR stage 融合与统计（P0/P1）

1. R2 融合小 stage；
2. R4 统计等级；
3. S4 双流（可选）。

**验收**：单 iteration kernel 数下降；测试全绿；统计字段一致。

### Phase C：CUDA Graph（P1）

1. R1 graph 捕获与参数更新；
2. 取消/重建路径。

**验收**：host launch 开销显著下降；replay 门禁通过。

### Phase D：wavefront 全局调度与状态（P1）

1. W2 fixed persistent grid + work stealing；
2. W3 状态 shared/SoA 实验。

**验收**：小 batch 与大 batch 差距 <1.5x；全局访存下降。

### Phase E：数据压缩与增量上传（P2）

1. S1 BVH 压缩；
2. S2 mipmap；
3. S3 增量上传。

**验收**：显存/带宽下降；输出一致。

## 9. 全局验收

- `ctest --test-dir build-cuda` 全绿；
- ReSTIR replay 门禁通过；
- 每个任务有改前/改后 `--bench-json` 数据；
- PFM 逐位一致或记录阈值；
- 文档更新 CUDA 架构说明。

## 10. 风险

| 风险 | 缓解 |
|---|---|
| W1 破坏 sample 顺序/数值一致性 | 每个 slot 内 sample 顺序与旧实现一致；PFM bit compare |
| CUDA Graph 版本/接口差异 | 限定固定拓扑参数更新；保留显式 launch 回退 |
| W3 shared 状态降低 occupancy | 以 occupancy 和实际耗时为验收，不强行切换 |
| BVH 量化改变遍历结果 | replay + 差分门禁；默认关闭开关 |
| 多 stream 同步复杂 | 每个阶段独立提交；取消时最终同步 |
