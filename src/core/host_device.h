#ifndef HOST_DEVICE_H
#define HOST_DEVICE_H

#if defined(__CUDACC__)
#define RT_HOST_DEVICE __host__ __device__
#define RT_FORCE_INLINE __forceinline__
#define RT_NOINLINE __noinline__ inline
#else
#define RT_HOST_DEVICE
#define RT_FORCE_INLINE inline
#if defined(__GNUC__) || defined(__clang__)
#define RT_NOINLINE __attribute__((noinline)) inline
#else
#define RT_NOINLINE inline
#endif
#endif

#endif
