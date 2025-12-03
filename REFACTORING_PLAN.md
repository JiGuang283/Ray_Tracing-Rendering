# 光线追踪渲染器代码重构方案

## 代码库概况分析

### 1. 项目结构总览

当前项目基于《Ray Tracing in One Weekend》和《The Next Week》教程实现，包含以下模块：

| 模块 | 文件 | 职责 |
|------|------|------|
| 基础数学库 | `vec3.h`, `ray.h` | 向量、射线的定义与操作 |
| 工具库 | `rtweekend.h` | 常量、随机数生成、通用工具函数 |
| 几何体系统 | `sphere.h`, `moving_sphere.h`, `aarect.h`, `box.h` | 可渲染几何体 |
| 加速结构 | `aabb.h`, `bvh.h` | 包围盒与BVH加速结构 |
| 材质系统 | `material.h`, `texture.h`, `perlin.h` | 材质与纹理 |
| 体积渲染 | `constant_medium.h` | 烟雾等体积效果 |
| 核心抽象 | `hittable.h`, `hittable_list.h` | 可命中对象接口 |
| 相机 | `camera.h` | 相机与射线生成 |
| 图像加载 | `rtw_stb_image.h`, `stb_image.h` | 纹理图像加载 |
| 窗口系统 | `WindowsApp.h`, `WindowsApp.cpp` | SDL2窗口显示 |
| 主程序 | `main.cpp` | 场景定义、渲染循环 |

### 2. 发现的主要问题

#### 2.1 头文件循环依赖与包含顺序
- `vec3.h` 包含了 `rtweekend.h`，而其他文件又依赖 `vec3.h`，存在隐式依赖
- 多个头文件重复 `using` 声明（如 `hittable_list.h` 再次声明 `using std::shared_ptr`）

#### 2.2 命名与代码风格不一致
- Include Guard 命名有拼写错误：`moving_sphere.h` 使用了 `MOVING_SHPERE_H`（少了一个 `E`）
- 函数参数命名不一致：`outpot_box` vs `output_box`（拼写错误）
- 成员变量命名混乱：`mat_ptr`, `mp`, `ptr` 表示同一概念

#### 2.3 const 正确性问题
- `ray` 类的成员变量 `orig`, `dir`, `tm` 是 public 的，应为 private
- 多处函数参数可以添加 `const` 修饰
- 部分函数返回值缺少 `const`

#### 2.4 现代 C++ 特性缺失
- 缺少 `noexcept` 标记
- 部分地方未使用 `override` 关键字
- 未使用 `nullptr`（虽然大部分已使用）
- 类的析构函数缺少 `virtual` 或者应使用 `= default`

#### 2.5 代码卫生问题
- 大量被注释掉的"临时代码"残留（尤其在 `main.cpp`）
- 魔术数字散落各处（如 `0.001`, `0.0001`, `1e-8`）
- `main.cpp` 中使用全局变量存储渲染参数
- `switch(9)` 硬编码场景选择
- 头文件中直接实现函数（非 inline）
- `perlin.h` 中使用裸指针 `new/delete`

#### 2.6 潜在 Bug
- `vec3::near_zero()` 中存在逻辑错误：`fabs(e[0] < s)` 应为 `fabs(e[0]) < s`
- `image_texture` 析构函数使用 `delete data` 应为 `stbi_image_free(data)`

---

## 重构计划

> **原则**：每个 Phase 完成后，代码必须可编译且能正确渲染图像。

---

### Phase 1: 紧急 Bug 修复

**目标**：修复可能导致运行时错误或渲染结果不正确的 Bug。

#### Step 1.1: 修复 `vec3::near_zero()` 逻辑错误

**文件**：`src/vec3.h`

**问题**：`fabs(e[0] < s)` 括号位置错误，`e[0] < s` 返回 bool，`fabs(bool)` 无意义。

**修改**：
```cpp
// 修改前
bool near_zero() const {
    const auto s = 1e-8;
    return (fabs(e[0] < s)) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
}

// 修改后
bool near_zero() const {
    constexpr auto s = 1e-8;
    return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
}
```

**验证点**：编译通过，渲染 Cornell Box 场景，确认散射方向在法线附近时不会出现异常。

---

#### Step 1.2: 修复 `image_texture` 内存释放方式

**文件**：`src/texture.h`

**问题**：`stbi_load` 分配的内存应使用 `stbi_image_free` 释放，而非 `delete`。

**修改**：
```cpp
// 修改前
~image_texture() {
    delete data;
}

// 修改后
~image_texture() {
    if (data) {
        stbi_image_free(data);
    }
}
```

**验证点**：渲染 `earth()` 场景，程序退出时无内存错误（可用 Valgrind 检测）。

---

### Phase 2: 头文件卫生（Include Guards & Typos）

**目标**：统一头文件保护宏命名，修复拼写错误。

#### Step 2.1: 修复 `moving_sphere.h` 的 Include Guard

**文件**：`src/moving_sphere.h`

**修改**：
```cpp
// 修改前
#ifndef MOVING_SHPERE_H
#define MOVING_SPHERE_H

// 修改后
#ifndef MOVING_SPHERE_H
#define MOVING_SPHERE_H
```

**验证点**：编译通过。

---

#### Step 2.2: 修复函数参数拼写错误 `outpot_box` → `output_box`

**文件**：`src/hittable.h`, `src/constant_medium.h`

**涉及位置**：
- `hittable.h`：`bounding_box` 虚函数声明，`rotate_y::bounding_box` 函数
- `constant_medium.h`：`constant_medium::bounding_box` 函数

**修改示例**（`hittable.h`）：
```cpp
// 修改前
virtual bool bounding_box(double time0, double time1,
                          aabb &outpot_box) const = 0;

// 修改后
virtual bool bounding_box(double time0, double time1,
                          aabb &output_box) const = 0;
```

**验证点**：编译通过，渲染任意场景正常。

---

### Phase 3: 基础数学库清理（`vec3.h`, `ray.h`）

**目标**：提升代码质量，增加 const 正确性，使用现代 C++ 特性。

#### Step 3.1: 清理 `vec3.h`

**修改内容**：

1. **将成员变量设为 private**（虽然这会破坏一些外部访问，但可通过 `operator[]` 访问）。
   
   > **注意**：由于 `bvh.h` 中 `box_compare` 直接访问 `v.e[axis]`，暂时保留 public。可选择添加 `double operator[](int i) const` 后再改为 private。

2. **使用 `constexpr` 替代 `const` 用于编译期常量**

3. **参数传递优化**：`unit_vector(vec3 v)` 改为 `unit_vector(const vec3& v)`

4. **移除 `using std::sqrt;`**：在函数内使用 `std::sqrt`

**具体修改**：
```cpp
// 1. 将魔术数字提取为命名常量
constexpr double kNearZeroThreshold = 1e-8;

bool near_zero() const {
    return (std::fabs(e[0]) < kNearZeroThreshold) 
        && (std::fabs(e[1]) < kNearZeroThreshold) 
        && (std::fabs(e[2]) < kNearZeroThreshold);
}

// 2. 优化 unit_vector 参数传递
inline vec3 unit_vector(const vec3& v) {
    return v / v.length();
}
```

**验证点**：编译通过，渲染 `random_scene()` 场景正常。

---

#### Step 3.2: 清理 `ray.h`

**修改内容**：

1. **成员变量设为 private**
2. **使用 `= default` 默认构造函数**
3. **添加 `noexcept` 标记**

```cpp
class ray {
public:
    ray() = default;
    
    ray(const point3& origin, const vec3& direction, double time = 0.0) noexcept
        : orig(origin), dir(direction), tm(time) {}

    const point3& origin() const noexcept { return orig; }
    const vec3& direction() const noexcept { return dir; }
    double time() const noexcept { return tm; }

    point3 at(double t) const noexcept {
        return orig + t * dir;
    }

private:
    point3 orig;
    vec3 dir;
    double tm = 0.0;
};
```

**验证点**：编译通过，渲染任意场景正常。

---

### Phase 4: 工具库清理（`rtweekend.h`）

**目标**：整理常量定义，清理代码。

#### Step 4.1: 使用 `constexpr` 定义常量

```cpp
// 修改前
const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

// 修改后
constexpr double infinity = std::numeric_limits<double>::infinity();
constexpr double pi = 3.14159265358979323846;
```

#### Step 4.2: 移除多余的分号

```cpp
// 修改前（random_double 函数末尾）
return distribution(generator);
;  // <- 多余的分号

// 修改后
return distribution(generator);
```

#### Step 4.3: 添加 `noexcept` 标记

```cpp
inline constexpr double degrees_to_radians(double degrees) noexcept {
    return degrees * pi / 180.0;
}

inline double clamp(double x, double min, double max) noexcept {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}
```

**验证点**：编译通过。

---

### Phase 5: 核心抽象层清理（`hittable.h`, `hittable_list.h`, `aabb.h`）

#### Step 5.1: 为 `hittable` 基类添加虚析构函数

**文件**：`src/hittable.h`

```cpp
class hittable {
public:
    virtual ~hittable() = default;
    
    virtual bool hit(const ray &r, double t_min, double t_max,
                     hit_record &rec) const = 0;
    virtual bool bounding_box(double time0, double time1,
                              aabb &output_box) const = 0;
};
```

**验证点**：编译通过，确保多态析构正确。

---

#### Step 5.2: 清理 `hittable_list.h` 中冗余的 `using` 声明

**文件**：`src/hittable_list.h`

```cpp
// 删除以下行（已在 rtweekend.h 中声明）
using std::make_shared;
using std::shared_ptr;
```

**验证点**：编译通过。

---

#### Step 5.3: 将 `aabb.h` 中 `surrounding_box` 改为 `inline`

**文件**：`src/aabb.h`

**问题**：非模板函数在头文件中定义但未标记 `inline`，可能导致链接时重定义错误。

```cpp
// 修改前
aabb surrounding_box(aabb box0, aabb box1) { ... }

// 修改后
inline aabb surrounding_box(const aabb& box0, const aabb& box1) {
    point3 small(
        std::fmin(box0.min().x(), box1.min().x()),
        std::fmin(box0.min().y(), box1.min().y()),
        std::fmin(box0.min().z(), box1.min().z())
    );
    point3 big(
        std::fmax(box0.max().x(), box1.max().x()),
        std::fmax(box0.max().y(), box1.max().y()),
        std::fmax(box0.max().z(), box1.max().z())
    );
    return aabb(small, big);
}
```

**验证点**：编译通过。

---

#### Step 5.4: 为 `aabb` 类添加默认初始化

```cpp
class aabb {
public:
    aabb() : minimum(point3(0,0,0)), maximum(point3(0,0,0)) {}
    // ...
};
```

**验证点**：编译通过。

---

### Phase 6: 几何体模块清理

#### Step 6.1: 清理 `sphere.h`

**修改内容**：
1. 删除空的默认构造函数（或使用 `= default` 并初始化成员）
2. 将成员变量设为 private，提供 getter 方法
3. 添加 `override` 关键字

```cpp
class sphere : public hittable {
public:
    sphere(const point3& cen, double r, shared_ptr<material> m)
        : center(cen), radius(r), mat_ptr(std::move(m)) {}

    bool hit(const ray &r, double t_min, double t_max,
             hit_record &rec) const override;

    bool bounding_box(double time0, double time1,
                      aabb &output_box) const override;

private:
    point3 center;
    double radius;
    shared_ptr<material> mat_ptr;

    static void get_sphere_uv(const point3 &p, double &u, double &v) {
        // ...
    }
};
```

**验证点**：编译通过，渲染球体场景正常。

---

#### Step 6.2: 清理 `aarect.h` 中的魔术数字

**文件**：`src/aarect.h`

```cpp
// 提取常量
namespace {
    constexpr double kAABBPadding = 0.0001;
}

// 使用
output_box = aabb(
    point3(x0, y0, k - kAABBPadding), 
    point3(x1, y1, k + kAABBPadding)
);
```

**验证点**：编译通过，渲染 Cornell Box 正常。

---

#### Step 6.3: 统一 `aarect.h` 中函数参数命名

**问题**：`xz_rect::hit` 和 `yz_rect::hit` 使用 `t0`, `t1` 参数名，而 `xy_rect::hit` 使用 `t_min`, `t_max`。

**修改**：统一为 `t_min`, `t_max`。

---

### Phase 7: 材质系统清理

#### Step 7.1: 为 `material` 基类添加虚析构函数

**文件**：`src/material.h`

```cpp
class material {
public:
    virtual ~material() = default;
    
    virtual color emitted(double u, double v, const point3 &p) const {
        return color(0, 0, 0);
    }

    virtual bool scatter(const ray &r_in, const hit_record &rec,
                         color &attenuation, ray &scattered) const = 0;
};
```

**验证点**：编译通过。

---

#### Step 7.2: 清理 `texture.h` 中的 `image_texture`

**修改内容**：
1. 禁止拷贝（资源管理类）
2. 考虑使用智能指针管理 `data`

```cpp
class image_texture : public texture {
public:
    // 禁止拷贝
    image_texture(const image_texture&) = delete;
    image_texture& operator=(const image_texture&) = delete;
    
    // 允许移动（可选，如需要）
    image_texture(image_texture&&) = default;
    image_texture& operator=(image_texture&&) = default;

    // ... 其他代码
    
private:
    static constexpr int bytes_per_pixel = 3;
    unsigned char* data = nullptr;
    int width = 0;
    int height = 0;
    int bytes_per_scanline = 0;
};
```

**验证点**：编译通过，渲染 earth 场景正常。

---

#### Step 7.3: 为 `texture` 基类添加虚析构函数

```cpp
class texture {
public:
    virtual ~texture() = default;
    virtual color value(double u, double v, const point3 &p) const = 0;
};
```

---

### Phase 8: Perlin 噪声模块现代化

**目标**：将裸指针改为 `std::vector` 或 `std::array`。

#### Step 8.1: 使用 `std::vector` 替代裸指针

**文件**：`src/perlin.h`

```cpp
class perlin {
public:
    perlin() {
        ranvec.resize(point_count);
        for (int i = 0; i < point_count; ++i) {
            ranvec[i] = unit_vector(vec3::random(-1, 1));
        }

        perm_x = perlin_generate_perm();
        perm_y = perlin_generate_perm();
        perm_z = perlin_generate_perm();
    }

    // 删除析构函数（不再需要手动释放）
    // ~perlin() { ... }

    // ...

private:
    static constexpr int point_count = 256;
    std::vector<vec3> ranvec;
    std::vector<int> perm_x;
    std::vector<int> perm_y;
    std::vector<int> perm_z;

    static std::vector<int> perlin_generate_perm() {
        std::vector<int> p(point_count);
        for (int i = 0; i < point_count; i++) {
            p[i] = i;
        }
        permute(p.data(), point_count);
        return p;
    }
    
    // ...
};
```

**验证点**：渲染 `two_perlin_spheres()` 场景正常。

---

### Phase 9: 相机模块清理

**文件**：`src/camera.h`

#### Step 9.1: 改进构造函数参数

```cpp
class camera {
public:
    camera(
        const point3& lookfrom,
        const point3& lookat,
        const vec3& vup,
        double vfov,               // 垂直视场角（度）
        double aspect_ratio,
        double aperture,
        double focus_dist,
        double time0 = 0.0,
        double time1 = 0.0
    ) { /* ... */ }

    ray get_ray(double s, double t) const;

private:
    point3 origin;
    point3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    double lens_radius;
    double time0;
    double time1;
};
```

**验证点**：编译通过，渲染任意场景正常。

---

### Phase 10: 主程序清理（`main.cpp`）

**目标**：清理临时代码，提取配置结构，消除全局变量。

#### Step 10.1: 删除所有注释掉的代码

**修改**：删除 `main.cpp` 中所有被注释的代码块，包括：
- 旧的 `hit_sphere` 函数
- 被注释的世界对象和材质
- 被注释的渲染循环变体

**验证点**：编译通过。

---

#### Step 10.2: 创建场景配置结构体

```cpp
struct SceneConfig {
    shared_ptr<hittable> world;
    color background;
    point3 lookfrom;
    point3 lookat;
    vec3 vup{0, 1, 0};
    double vfov = 40.0;
    double aperture = 0.0;
    double focus_dist = 10.0;
    double aspect_ratio = 16.0 / 9.0;
    int image_width = 1280;
    int samples_per_pixel = 400;
};

SceneConfig select_scene(int scene_id) {
    SceneConfig config;
    
    switch (scene_id) {
    case 1:
        config.world = random_scene();
        config.background = color(0.70, 0.80, 1.00);
        config.lookfrom = point3(13, 2, 3);
        config.lookat = point3(0, 0, 0);
        config.vfov = 20.0;
        config.aperture = 0.1;
        break;
    // ... 其他场景
    }
    
    return config;
}
```

**验证点**：编译通过，渲染任意场景正常。

---

#### Step 10.3: 提取魔术数字为命名常量

```cpp
namespace RenderConfig {
    constexpr int kMaxDepth = 50;
    constexpr double kTMin = 0.001;  // 避免自相交
}
```

使用位置：
- `ray_color` 函数中的递归深度检查
- `hit` 调用中的 `t_min` 参数

**验证点**：编译通过。

---

#### Step 10.4: 将 `switch(9)` 改为可配置的场景选择

```cpp
int main(int argc, char* args[]) {
    // 从命令行参数或配置获取场景 ID
    int scene_id = 9;  // 默认场景
    if (argc > 1) {
        scene_id = std::atoi(args[1]);
    }
    
    SceneConfig config = select_scene(scene_id);
    // ...
}
```

**验证点**：编译通过，可通过命令行参数选择场景。

---

### Phase 11: BVH 模块清理

**文件**：`src/bvh.h`

#### Step 11.1: 使用 lambda 替代独立的比较函数

```cpp
bvh_node::bvh_node(const std::vector<shared_ptr<hittable>>& src_objects,
                   size_t start, size_t end, double time0, double time1) {
    auto objects = src_objects;
    
    int axis = random_int(0, 2);
    
    auto comparator = [axis](const shared_ptr<hittable>& a, 
                             const shared_ptr<hittable>& b) {
        aabb box_a, box_b;
        if (!a->bounding_box(0, 0, box_a) || !b->bounding_box(0, 0, box_b)) {
            std::cerr << "No bounding box in bvh_node constructor.\n";
        }
        return box_a.min()[axis] < box_b.min()[axis];
    };
    
    // ... 使用 comparator
}
```

**验证点**：编译通过，渲染加速正常。

---

#### Step 11.2: 删除未使用的默认构造函数声明

```cpp
// 删除
bvh_node();
```

---

### Phase 12: 窗口应用模块清理

**文件**：`src/WindowsApp.h`, `src/WindowsApp.cpp`

#### Step 12.1: 改进代码风格一致性

- 使用 `nullptr` 替代 `NULL`（如果存在）
- 成员变量命名统一使用 `m_` 前缀（已遵循）
- 添加 `noexcept` 标记

```cpp
bool shouldWindowClose() const noexcept { return m_quit; }
int getMouseMotionDeltaX() const noexcept { return m_mouse_delta_x; }
// ...
```

**验证点**：编译通过，窗口功能正常。

---

### Phase 13: 头文件依赖优化

**目标**：减少不必要的头文件包含，使用前向声明。

#### Step 13.1: 在 `material.h` 中使用前向声明

目前 `material.h` 包含了 `hittable.h`，但只需要 `hit_record` 的定义。

由于 `hit_record` 是一个 struct 且需要完整定义，这里保持现状。但可以重新组织：

**建议**：将 `hit_record` 移至独立的 `hit_record.h` 文件（可选的未来优化）。

---

### Phase 14: 统一代码风格

#### Step 14.1: 构造函数初始化列表格式

采用一致的风格：
```cpp
// 统一风格
ClassName::ClassName(Type1 param1, Type2 param2)
    : member1(param1)
    , member2(param2) 
{}
```

#### Step 14.2: 空构造函数处理

```cpp
// 修改前
sphere() {}

// 修改后（如果确实需要默认构造）
sphere() = default;

// 或者删除（如果不需要）
```

---

## 完整修改清单（按执行顺序）

| 步骤 | 文件 | 修改类型 | 风险级别 |
|------|------|----------|----------|
| 1.1 | vec3.h | Bug 修复 | 高 |
| 1.2 | texture.h | Bug 修复 | 中 |
| 2.1 | moving_sphere.h | 拼写修复 | 低 |
| 2.2 | hittable.h, constant_medium.h | 拼写修复 | 低 |
| 3.1 | vec3.h | 代码质量 | 低 |
| 3.2 | ray.h | 代码质量 | 低 |
| 4.1-4.3 | rtweekend.h | 代码质量 | 低 |
| 5.1 | hittable.h | 析构函数 | 中 |
| 5.2 | hittable_list.h | 清理 | 低 |
| 5.3 | aabb.h | inline 修复 | 中 |
| 6.1-6.3 | sphere.h, aarect.h | 代码质量 | 低 |
| 7.1-7.3 | material.h, texture.h | 析构函数 | 中 |
| 8.1 | perlin.h | 现代化 | 中 |
| 9.1 | camera.h | 代码质量 | 低 |
| 10.1-10.4 | main.cpp | 清理重构 | 中 |
| 11.1-11.2 | bvh.h | 代码质量 | 低 |
| 12.1 | WindowsApp.h/cpp | 代码质量 | 低 |

---

## 推荐的提交策略

每完成一个 Phase 后进行一次 Git 提交，提交信息格式建议：

```
refactor(module): 简要描述

- 具体修改点 1
- 具体修改点 2
```

例如：
```
fix(vec3): 修复 near_zero() 函数逻辑错误

- 修正 fabs() 括号位置
- 使用 constexpr 定义阈值常量
```

---

## 附录：C++11/14/17 现代特性使用检查表

| 特性 | 当前状态 | 建议 |
|------|----------|------|
| `auto` | ✅ 已使用 | 继续保持 |
| `nullptr` | ✅ 已使用 | 继续保持 |
| `override` | ⚠️ 部分使用 | 全面添加 |
| `final` | ❌ 未使用 | 可选添加 |
| `constexpr` | ⚠️ 部分使用 | 扩大使用 |
| `noexcept` | ❌ 未使用 | 建议添加 |
| 范围 for | ✅ 已使用 | 继续保持 |
| `std::move` | ⚠️ 部分使用 | 适当优化 |
| 智能指针 | ✅ 已使用 | 继续保持 |
| `= default` | ❌ 未使用 | 建议使用 |
| `= delete` | ❌ 未使用 | 建议使用 |
| 统一初始化 | ⚠️ 部分使用 | 可选统一 |

---

## 结语

本方案将重构工作分解为 14 个 Phase，每个 Phase 都是独立可编译的原子操作。建议按顺序执行，特别是 **Phase 1 的 Bug 修复应当优先完成**。

在执行每个步骤后，请务必：
1. 编译项目确保无错误和警告
2. 运行渲染器验证输出图像正确
3. 提交 Git 记录变更

祝重构顺利！
