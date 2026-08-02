# Render API

The render API is the backend-neutral boundary used by the application:

```text
JSON -> SceneIR
     -> CpuRenderSession  -> double polymorphic runtime
     -> CudaRenderSession -> CompiledScene -> DeviceScene
```

`IRenderSession` is synchronous and owns backend preparation state. The
application may run it on one background thread, but a session rejects
overlapping `render()` calls. A benchmark creates one session outside its run
loop, so scene construction, packed compilation, CUDA upload, and reusable
workspace allocation are not accidentally measured as repeated render work.

## Request And Result

`RenderRequest` is the validated input contract. It carries image extent,
integrator kind, sampling limits, seed, thread count, CUDA batch size, sample
clamp, and color pipeline settings. Integrator IDs 0 through 4 map once to an
`IntegratorPolicy`; CPU double transport and Packed float transport consume
that same policy while retaining independent numerical kernels.

`RenderResult` owns three distinct products:

- `BeautyFilm`: linear radiance sums and per-pixel sample counts.
- `RenderBuffer`: the completed display-referred image after color resolve.
- `RenderStats`: requested/completed/invalid/clamped sample counts plus
  preparation, device, resolve, cancellation, and workspace information.

The Film never aliases a display buffer. `PreviewSurface` is a separate array
of atomic packed RGB pixels: render workers publish completed preview pixels,
and the window thread reads snapshots without touching the Film being written.

## Cancellation And Failure

`CancellationSource` and `CancellationToken` are shared by both backends. CPU
workers check cancellation at pixel boundaries and every 16 samples. Worker
exceptions cancel remaining work, all threads are joined, and the first
exception is rethrown to the application. CUDA observes cancellation between
batches after its single batch synchronization point.

The current public result contains only beauty. `FilmChannel` reserves the
extension boundary for later AOV/debug work without exposing unused channel
storage today.
