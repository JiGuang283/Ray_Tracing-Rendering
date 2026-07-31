# Third-party sources

The following source dependencies are vendored to keep offline builds
reproducible.

- cgltf, commit `85cd62382dfea638278962690cf515023f33ed00`
  - Source: https://github.com/jkuhlmann/cgltf
  - License: MIT, see `cgltf/LICENSE`
- MikkTSpace, commit `3e895b49d05ea07e4c2133156cfa94369e19e409`
  - Source: https://github.com/mmikk/MikkTSpace
  - License notice is retained in `mikktspace.c` and `mikktspace.h`
  - Local patch: guard zero-bit hash rotations to avoid undefined 32-bit
    shifts reported by UBSan; tangent-space output is unchanged.

Existing single-header dependencies retain their upstream notices in their
source files.
