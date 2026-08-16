# 优化基线数据

> 采集时间：2026-08-15
> 机器：20 核 CPU / NVIDIA GeForce RTX 4060 Laptop GPU
> 源码基线：`a45f612`（HEAD）
> 构建：`build/`（CPU Release）、`build-cuda/`（CUDA Release）

## 1. 测试基线

| 构建目录 | 结果 |
|---|---|
| `build/` | 2/2 测试通过 |
| `build-cuda/` | 34/34 测试通过 |

## 2. CUDA wavefront：max_depth 对设备时间的影响

场景 23，400x225，spp16，seed 123，run 2：

| max_depth | device_seconds | traversal_steps |
|---:|---:|---:|
| 4  | 0.0213 | 3,017,951 |
| 20 | 0.0294 | 3,042,228 |
| 50 | 0.0363 | 3,042,228 |

固定深度循环导致 depth 50 比 depth 4 多约 70% 设备时间，而实际
traversal 量基本不变。

## 3. CUDA batch size 对 launch 开销的影响

场景 23，spp8，max_depth50，seed 123，run 2：

| batch_size | device_seconds | batch_count |
|---:|---:|---:|
| 4096  | 0.1447 | 176 |
| 16384 | 0.0462 | 48  |
| 65536 | 0.0230 | 16  |

相同 traversal_steps 下，小 batch 时间增加 6 倍以上。

## 4. CPU 多线程扩展

场景 23，400x225，spp8，seed 123，run 2：

| threads | seconds | samples/s |
|---:|---:|---:|
| 1  | 0.379 | 1.90M |
| 2  | 0.227 | 3.17M |
| 4  | 0.132 | 5.47M |
| 8  | 0.071 | 10.13M |
| 20 | 0.048 | 15.05M |

8 线程到 20 线程仅提升 1.49 倍。

## 5. CPU double 运行时 vs host packed 内核（单线程）

场景 23，单线程微基准。CPU double 数字来自完整 `Renderer` 路径（含
Film 累加）；packed 数字只调用 `trace_packed_path_core`（不含 Film），
真实后端中 packed 差距只会更大。

| 路径 | samples/s |
|---|---:|
| CPU double 多态运行时 | 约 1.90M |
| host 直接调用 `trace_packed_path_core` | 约 0.86M |

## 6. LinearBVH 固定栈快速实验

将 `LinearBVH::hit` 的 `std::vector<int> stack + reserve(128)` 临时改为
不初始化的 `std::array<int, 128>` 后，场景 23 800x450 spp64 单线程：

| 版本 | seconds（3 次） |
|---|---:|
| vector + reserve | 18.95 / 19.87 / 18.64 |
| array（不初始化） | 17.49 / 18.00 / 19.55 |

约 5% 提升（噪音较大）。实验代码已还原，实施阶段应重新测量并保留。
