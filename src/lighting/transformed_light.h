#ifndef TRANSFORMED_LIGHT_H
#define TRANSFORMED_LIGHT_H

#include "light.h"
#include "transform.h"

#include <memory>
#include <stdexcept>

class TransformedLight final : public Light {
  public:
    TransformedLight(shared_ptr<Light> light, Transform object_to_world)
        : m_light(std::move(light)),
          m_object_to_world(std::move(object_to_world)) {
        if (!m_light) {
            throw std::invalid_argument(
                "TransformedLight requires a child light.");
        }
        if (!m_object_to_world.is_rigid()) {
            throw std::invalid_argument(
                "Automatic emitter transforms must be rigid.");
        }
    }

    LightSample sample(const point3 &point, const vec2 &u) const override {
        LightSample result = m_light->sample(
            m_object_to_world.point_to_object(point), u);
        result.wi = unit_vector(
            m_object_to_world.vector_to_world(result.wi));
        return result;
    }

    double pdf(const point3 &origin, const vec3 &direction) const override {
        return m_light->pdf(
            m_object_to_world.point_to_object(origin),
            m_object_to_world.vector_to_object(direction));
    }

    bool is_delta() const override {
        return m_light->is_delta();
    }

    bool is_infinite() const override {
        return m_light->is_infinite();
    }

    color Le(const ray &value) const override {
        return m_light->Le(m_object_to_world.ray_to_object(value));
    }

    color power() const override {
        return m_light->power();
    }

  private:
    shared_ptr<Light> m_light;
    Transform m_object_to_world;
};

#endif
