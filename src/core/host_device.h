#ifndef HOST_DEVICE_H
#define HOST_DEVICE_H

#if defined(__CUDACC__)
#define RT_HOST_DEVICE __host__ __device__
#define RT_FORCE_INLINE __forceinline__
#else
#define RT_HOST_DEVICE
#define RT_FORCE_INLINE inline
#endif

#endif
