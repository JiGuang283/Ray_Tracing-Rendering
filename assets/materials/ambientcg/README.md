# ambientCG Material Library

These are selected 1K JPG PBR materials from
[ambientCG](https://ambientcg.com/). All included assets are distributed
under the Creative Commons CC0 1.0 license.

| Asset | Source | Local maps | Archive SHA-256 |
| --- | --- | --- | --- |
| Asphalt011 | https://ambientcg.com/view?id=Asphalt011 | Color, Roughness, NormalGL, Opacity, Displacement, MaterialX | `d2abbe44a54dc0f677165c083a75dafcdddb3cce27f2eaec9a084461ced2aa68` |
| Bricks001 | https://ambientcg.com/view?id=Bricks001 | Color, Roughness, NormalGL, Displacement, MaterialX | `d4e4109f305b7d1094e1c18b2f7f6a3468c62477de915ff66975fd0155b6873c` |
| Metal012 | https://ambientcg.com/view?id=Metal012 | Color, Metalness, Roughness, NormalGL, Displacement, MaterialX | `c4792207f133357c6a6bfb008f6192b8691268fb0a2ca95b45825ac797e5028f` |
| Metal033 | https://ambientcg.com/view?id=Metal033 | Color, Metalness, Roughness, NormalGL, Displacement, MaterialX | `ca6e585e45b8aef444c237aa9a404146c347eb9e71307fc6cc1ac6dfa92d041f` |
| Planks021 | https://ambientcg.com/view?id=Planks021 | Color, Roughness, NormalGL, Ambient Occlusion, Displacement, MaterialX | `9a593ffe4194132bd8c881558b47107f2c3c739018bbdad12fca5d877c90746a` |
| Rock027 | https://ambientcg.com/view?id=Rock027 | Color, Roughness, NormalGL, Ambient Occlusion, Displacement, MaterialX | `725cb3fe84ee4312e2f39316463a7d4d502e3ae8a7adeea5145c4f3e0ea1e97e` |

## Import Conventions

- Color maps are sRGB inputs and must be converted to linear before shading.
- Roughness, metalness, normal, opacity, ambient occlusion, and displacement
  maps are linear data.
- `NormalGL` is the canonical tangent-space normal map for this renderer.
- DirectX normal maps and application-specific Blend, USD, and Godot files
  are intentionally omitted.
- Displacement, opacity, and ambient occlusion maps are retained as future
  validation assets even when the current renderer does not consume them.
- The original MaterialX file is retained as a reference OpenPBR graph.
