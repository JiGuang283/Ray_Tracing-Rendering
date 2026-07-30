# Metal 012

- Source: https://ambientcg.com/view?id=Metal012
- Download: `Metal012_1K-JPG.zip`
- License: Creative Commons CC0 1.0
- Resolution: 1K
- Archive SHA-256:
  `c4792207f133357c6a6bfb008f6192b8691268fb0a2ca95b45825ac797e5028f`

The local subset contains the color, metalness, roughness, OpenGL normal,
and displacement maps. `Metal012_1K-JPG.mtlx` is retained as the original
MaterialX/OpenPBR material description.

## Color Space

- `Color`: sRGB input; convert to linear before shading.
- `Metalness`: linear data.
- `Roughness`: linear data.
- `NormalGL`: linear data, OpenGL tangent-space convention.
- `Displacement`: linear data; not yet consumed by the renderer.

The DirectX normal map and application-specific scene files from the archive
are intentionally omitted.
