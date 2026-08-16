# 渲染器优化计划执行报告

> 状态：实施完成（按方向 A 收口，偏差已记录）
> 基线：`a45f612`；当前工作区为全部 Phase 0-5 修改。

## 1. 完成情况

| 阶段 | 结果 |
|---|---|
| Phase 0 度量与基线 | 完成 |
| Phase 1 CPU/CUDA 低风险优化 | 完成 |
| Phase 2 CUDA wavefront/ReSTIR 调度与统计 | 完成 |
| Phase 3 CPU 资源/预解码/CPU packed 原型 | 完成 |
| Phase 4 revision、统计拆分、文件拆分 | 完成 |
| Phase 4 ARCH-1 packed 性能收敛 | **未达标（记录偏差）** |
| Phase 5 文档与构建固化 | 完成 |

## 2. 关键结果

- CUDA depth4/depth50 设备时间基本持平（固定深度空跑消除）。
- batch 4096/65536 性能差距从约 6.3x 降至约 2.5x。
- CPU 阴影查询、Film fast path、BVH 固定栈等优化落地。
- 环境贴图/图像 sRGB 预解码完成。
- `RenderStats` 拆为 Base/Cpu/Cuda，BENCH/JSON 兼容。
- `scene_compiler.cpp` 拆为 5 个文件。
- 新增 `--cpu-packed` 验证后端、`--bench-json`、`tools/perf_smoke.py`。
- 新增 `cpu-sanitize`、`cpu-lto-native` preset。

## 3. 最终测试证据

- CPU Release：3/3。
- CUDA Release：35/35。
- CPU ASan+UBSan：3/3（方向 B host fast 路径后复测通过）。
- LTO+native 构建：通过。
- 编译警告：CPU/CUDA 均 0。
- `perf_smoke.py review-baseline + cpu_packed_scene23`：6/6。

## 4. 偏差与原因

ARCH-1 要求 `cpu-packed` 达到旧 double 后端性能并切换默认。初始实测
（400x225, spp32, 20 线程）double 约 0.17s，cpu-packed 约 0.55s。
方向 B 实施后：double 0.168s，cpu-packed 0.250s（单线程 1.04x，多线程
1.47x）；未切换默认，但验证后端性能已提升约 55%。

已尝试并排除/保留的优化：

- fast BSDF：单独复测仍导致 CUDA ReSTIR GI replay host/device 失配。
- identity-transform 快速路径：同样导致 replay 失配。
- any-hit occlusion：未证明失配源，但随实验回退。

因此默认 CPU 后端保持 double 多态运行时；`--cpu-packed` 作为验证后端保留。

cachegrind 补充数据（64x36, spp2, 单线程）：packed D refs 13.9M / writes
6.8M，double 11.1M / 5.6M。方向 B 已将单线程差距降至约 4%；剩余多线程
差距与更高的总内存引用及混合 P/E 核上的 AVX2 调度特性相关。
