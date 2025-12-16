# 光线追踪渲染器

## 改进

1. 实现多线程
2. 积分器改进
3. 材质改进
4. 光源改进
5. 实现外部模型加载（未实现）
6. 渲染管线优化及GUI（未实现）

## 1. 实现多线程

### 1.1 实现方式

渲染时通过基于图块（Tile-based）的动态任务分配实现了多线程渲染。

核心思想是将整个图像分割成许多小的矩形区域（Tile），然后启动多个线程，让每个线程不断地从一个共享的任务队列中“抢”任务来做，直到所有图块都渲染完毕。

这种动态调度的方式具有优势。因为渲染场景中不同部分的计算量差异很大（例如天空可能很快，而玻璃球或焦散区域很慢）。如果静态分配（例如线程 A 负责上半屏，线程 B 负责下半屏），可能会出现线程 A 早就干完了，在等线程 B 的情况。而这种“抢任务”的模式能保证所有 CPU 核心始终处于忙碌状态，直到整个图像渲染完成。

### 1.2 性能

对于标准的 cornell_box ，令 samples_per_pixel = 400，kMaxDepth = 50 渲染耗时为18.98s。
（原来单线程的代码现在找不到了，之后补上对比结果）

## 2. 积分器优化

### 2.1 实现轮盘赌终止策略

使用俄罗斯轮盘赌的方法来提前终止光线，其思路如下：
根据其贡献值（吞吐量）随机决定是否终止光线。如果决定继续追踪，就增加这条光线的“权重（强度）”，以补偿那些被提前终止的光线所损失的能量。

#### 算法流程：
假设当前光线的累积吞吐量（Throughput，即光线携带的能量/颜色权重）为 $L_o$，我们设定一个继续追踪的概率为 $P$（$0 < P < 1$）。

1.  生成一个 $[0, 1]$ 之间的随机数 $\xi$。
2.  **判定：**
    *   **如果 $\xi > P$（概率为 $1-P$）：** 光线不幸“死亡”，终止追踪，返回结果 **0**。
    *   **如果 $\xi \leq P$（概率为 $P$）：** 光线“存活”，继续追踪下一跳。**但是**，为了守恒能量，必须将当前的光线强度除以 $P$。
        *   新强度 $L_{new} = L_o / P$

#### 最佳实践：基于吞吐量（Throughput）的动态 P

最常用的策略是根据当前光线的“贡献度”来决定 $P$。如果一条光线经过几次反弹后，打到了黑色的墙上，它的吞吐量变得很低（对最终画面贡献小），我们就应该加大它死亡的概率。

通常使用当前光线颜色的最大分量作为概率：

$$ P = \text{Clamp}(\max(Throughput.r, Throughput.g, Throughput.b), \text{min\_prob}, \text{max\_prob}) $$

或者更简单的形式（不加 Clamp，但需防止除零）：
$$ P = \max(Throughput.r, Throughput.g, Throughput.b) $$

#### 代码实现

```cpp
if (depth >= m_rr_start_depth) {
    double p_survive =
        std::max({throughput.x(), throughput.y(), throughput.z()});
    p_survive = clamp(p_survive, 0.005, 0.95);

    if (random_double() > p_survive) {
        break;
    }
    throughput /= p_survive;
}
```

#### 效果

实现轮盘赌的策略后，可以在不大幅度影响渲染结果质量加快渲染。

在标准的 cornell_box 场景中进行测试。

![原积分器](images/scene07_integrator0_1765799984.png)
###### 无轮盘赌终止

![轮盘赌](images/scene07_integrator1_1765799945.png)
###### 使用轮盘赌终止

可以看到两者图片从视觉上基本没有差别

### 2.2 实现pbr材质（BSDF采样）

为支持后续的pbr微表面模型材质，需要改进现有积分器支持BSDF采样。对于原来的光线追踪，其只定义了漫反射，金属，绝缘体三种材质以及简单的反射折射行为。而pbr材质基于微表面模型，可以更好地描述光线与材质的互动，从而更好的描述光线的反射折射方向。

### 2.3 实现直接光照采样

在传统的光线追踪中，需要光线直接打到光源上才会计算光照并贡献颜色。在光源较小的情况，光线会很难击中光源，导致画面中有较多的噪点。而直接光照采样则在光线打到物体上时，直接对光源进行采样并计算贡献值。

在相同参数设置下（samples_per_pixel = 400，MaxDepth = 50，使用轮盘赌终止策略），其结果如下：

![原积分器](images/scene07_integrator1_1765801791.png)
###### 原积分器结果，渲染时间：10.3s

![直接光照采样](images/scene21_integrator3_1765801855.png)
###### 直接光照采样，渲染时间：19.1s

可以看到尽管直接光照采样的渲染时间较长，但其噪点明显减少，且保持了光线追踪的全局光照渲染效果，如盒子的侧面有墙反射过来的颜色。

### 2.4 实现MIS

直接光源采样有一个缺点，当物体的材质较为光滑，其会镜面反射大部分的光，而光源较大时，在光源上只有一小部分会贡献高光。但普通的大型光源如面光源是随机采样，因此很难采样到贡献高光的部分。进而无法正确反射。

![alt text](images/image.png)

为解决这一问题，引入了多重重要性采样。

多重重要性采样在光线击中物体时即执行直接光源采样，同时也继续执行光线追踪来判断最终是否击中光源。（直接光照采样中不关注最后是否击中光源）。之后将两者的贡献进行分别加权并求和。

![alt text](images/image-1.png)

这样就可以解决直接光照采样在光滑表面的问题。

#### 与MIS的对比

| 图 1（直接光照采样） | 图 2（MIS） |
|-----|-----|
| ![](images/scene23_integrator3_1765847116.png) | ![](images/scene23_integrator4_1765847185.png) |
###### 图1渲染时间：1.1s，图2渲染时间：1.3s



| 细节位置 | NEE (直接光照采样) | MIS (多重重要性采样) |
| :---: | :---: | :---: |
| **小球顶部** | ![NEE小球顶部](images/image-3.png) | ![MIS小球顶部](images/image-2.png) |
| **小球左侧** | ![NEE小球左侧](images/image-4.png) | ![MIS小球左侧](images/image-5.png) |

可以看到直接光照采样中，左侧的光滑小球表面没有正确反射顶部的面光源，同时在中间小球上的像也较为模糊。

## 3. 材质优化

### 3.1 新接口

由于要实现蒙特卡洛积分，以及pbr材质，因此需要设计材质的新接口，使其在材质采样时可以获取pdf等数值。
```cpp
struct BSDFSample {
    vec3 wi; // 采样生成的入射方向
    color f; // BSDF 吞吐量, 存 f (BSDF值)，让积分器自己乘 cos 和除
             // pdf(delta材质除外)。
    double pdf;                   // 采样该方向的概率密度
    bool is_specular;             // 是否是镜面反射（Delta分布）
    bool is_transmission = false; // 区分透射
};

class material {
  public:
    virtual ~material() = default;

    // 原emitted函数（保留旧接口）
    virtual color emitted(double u, double v, const point3 &p) const {
        return color(0, 0, 0);
    }

    // 新 emitted 接口
    virtual color emitted(const hit_record &rec, const vec3 &wo) const {
        return color(0, 0, 0);
    }

    // 判断材质是否含有完美镜面（Delta 分布）成分
    virtual bool is_specular() const {
        return false;
    }

    // 采样 BSDF 生成入射方向 (Importance Sampling)
    virtual bool sample(const hit_record &rec, const vec3 &wo,
                        BSDFSample &sampled) const {
        return false;
    }

    // 给定两个方向，计算反射比率（BSDF值）
    virtual color eval(const hit_record &rec, const vec3 &wo,
                       const vec3 &wi) const {
        return color(0, 0, 0);
    }

    // 计算pdf
    virtual double pdf(const hit_record &rec, const vec3 &wo,
                       const vec3 &wi) const {
        return 0.0;
    }

    // Deprecated scatter
    virtual bool scatter(const ray &r_in, const hit_record &rec, color &albedo,
                         ray &scattered, double &pdf_val) const {
        return false;
    }

    // 保留旧接口
    virtual bool scatter(const ray &r_in, const hit_record &rec,
                         color &attenuation, ray &scattered) const {
        return false;
    }
};
```

### 3.2 PBR材质

使用 Cook-Torrance 微表面模型来描述PBR材质，以得到近似物理正确的结果。

![alt text](images/20230611105830.png)

Cook-Torrance 假设物体表面不是绝对光滑的，而是由无数个微小的、像镜子一样的 **微平面（Microfacets）** 组成的。

Cook-Torrance 是一个 **BRDF（双向反射分布函数）** 的高光（Specular）部分。一个完整的 PBR 材质通常包含漫反射和高光反射两部分：

$$ f_r = k_d f_{diffuse} + k_s f_{specular} $$

其中 **$f_{specular}$** 就是 Cook-Torrance 的核心公式：

$$ f_{specular} = \frac{D \cdot F \cdot G}{4 (\omega_o \cdot n) (\omega_i \cdot n)} $$

**公式中的变量含义：**
*   $\omega_i$：光照方向（Light vector, $l$）
*   $\omega_o$：观察方向（View vector, $v$）
*   $n$：宏观表面法线
*   $D, F, G$：三个核心物理项，法线分布函数，菲涅尔方程与几何函数。
*   分母 $4 (\omega_o \cdot n) (\omega_i \cdot n)$：用于校正从微观几何体转换到宏观表面时的数学变换因子。

直接通过这些系数来定义PBR材质较为困难且不符合直觉，通常使用粗糙度和金属度来进行描述。

### 3.3 场景展示

![alt text](images/scene37_integrator4_1765855502.png)
###### 这是一个展示了粗糙度和金属度变化的场景。从左到右粗糙度逐渐增大，从上到下金属度逐渐增大

可以看到，左上角的小球高光清晰锐利，接近塑料的质感；左下角的小球高光相同，但高光颜色为红色，质感更接近金属；而从左向右粗糙度逐渐增大，高光逐渐扩散，变得模糊。

![alt text](images/scene13_integrator2_1765866884.png)
![alt text](images/scene14_integrator2_1765867576.png)
###### PBR材质展示，展示了不同粗糙度和颜色的金属和非金属，可以看到结果与现实中的材质非常接近

## 4. 光源改进

### 4.1 新接口

```cpp
struct LightSample {
    color Li;    // 到达着色点的辐射亮度 (Radiance)，让积分器自己乘 cos 和除 pdf
    vec3 wi;     // 入射方向 (指向光源)
    double pdf;  // 采样概率密度 (通常是关于立体角的)
    double dist; // 到光源的距离 (用于阴影射线遮挡测试)
    bool is_delta; // 是否是 Delta 光源 (点光源/平行光)
};

class Light {
  public:
    virtual ~Light() = default;

    // 统一采样接口
    // u: [0,1)^2 的随机数，用于面光源采样
    virtual LightSample sample(const point3 &p, const vec2 &u) const = 0;

    // 给定方向的 PDF (用于 MIS)
    virtual double pdf(const point3 &origin, const vec3 &direction) const {
        return 0.0;
    }

    // 是否是 Delta 光源
    virtual bool is_delta() const {
        return false;
    }

    // 是否是无限远环境光源
    virtual bool is_infinite() const {
        return false;
    }

    // BSDF 命中光源时获取辐射度
    virtual color Le(const ray &r) const {
        return color(0, 0, 0);
    }

    // 功率加权选择
    virtual color power() const {
        return color(0, 0, 0);
    }
};
```

### 4.2 平行光

平行光实现较为简单，只要将距离设为无限，并返回恒定值即可。

![alt text](images/scene17_integrator4_1765869234.png)
###### 平行光场景

可以看到，不同位置的物体阴影大小和方向相同，且为硬阴影，说明平行光实现正确。

### 4.3 点光源

![alt text](images/scene15_integrator3_1765869519.png)
可以看到小球上反射的点光源高光，且小球阴影与相对点光源的位置有关，且为硬阴影。

注意到图片上有明亮的噪点。这是因为光线偶然选择了概率非常小的路径，而蒙特卡洛积分会按概率补偿其强度，这就导致会返回非常大的数值。解决方法是在积分器中截断过高的贡献值。尽管理论上会引入偏差，但对最终渲染结果影响较小，可以接受。

![alt text](images/scene15_integrator3_1765869548.png)
###### 应用截断后的结果，可以看到亮点消失

### 4.4 面光源

面光源所生成的阴影应该为软阴影，其阴影的亮度与阴影区域能看到的光源面积有关。

![alt text](images/scene20_integrator4_1765870743.png)

![alt text](images/scene38_integrator4_1765870688.png)
###### 图中悬空的红色小球离面光源更近，因而阴影的边缘更模糊

可以看到，面光源的阴影为软阴影，且阴影的颜色与到光源的距离有关。

### 4.5 环境光贴图

实现了支持hdr格式的环境光贴图，实现了CDF，即优先向更亮的地方发射光线。同时支持两种常见的映射模式 Light Probe (Angular Map) 和 Equirectangular (经纬度映射)。


![alt text](images/image-6.png)
![alt text](images/scene24_integrator4_1765851013.png)
###### 场景1（室内）

![alt text](images/image-7.png)
![alt text](images/scene25_integrator4_1765851051.png)
###### 场景2（室外）

![alt text](images/image-9.png)
![alt text](images/scene26_integrator4_1765851063.png)
###### 场景3（rnl）

![alt text](images/image-8.png)
![alt text](images/scene27_integrator4_1765851071.png)
###### 场景4（stpeters）

从小球的反射可以看出环境光实现正常。

## 5.综合展示

### 5.1 场景1

![alt text](images/scene30_integrator4_1765872018.png)
###### 在环境光中展示不同材质

### 5.2 场景2

![alt text](images/scene31_integrator4_1765874019.png)
###### 改进后的康奈尔盒，可以清晰地看到软阴影、全局光照、玻璃折射

### 5.3 场景3

![alt text](images/scene39_integrator4_1765875042.png)
###### 用一个较光滑的非金属材质作为底座，可以看到环境光多次反射的效果
