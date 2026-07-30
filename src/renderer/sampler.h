#ifndef SAMPLER_H
#define SAMPLER_H

#include "rtweekend.h"

struct CameraSample {
    double u = 0.0;
    double v = 0.0;
};

class Sampler {
  public:
    explicit Sampler(uint32_t seed) : m_rng(seed) {
    }

    CameraSample next_camera_sample(int x, int y, int width,
                                    int height) {
        CameraSample sample;
        sample.u = (x + m_rng.next()) / (width - 1);
        sample.v = (y + m_rng.next()) / (height - 1);
        return sample;
    }

    RNG &rng() {
        return m_rng;
    }

  private:
    RNG m_rng;
};

#endif
