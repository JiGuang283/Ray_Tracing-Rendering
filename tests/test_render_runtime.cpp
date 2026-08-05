#include "test_harness.h"

#include "camera.h"
#include "beauty_film.h"
#include "cpu_path_integrator.h"
#include "hittable_list.h"
#include "material_programs.h"
#include "preview_surface.h"
#include "renderer.h"
#include "sphere.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

class ThrowingIntegrator final : public Integrator {
  public:
    color Li(const ray &, const hittable &, const color &,
             IntegratorContext &) const override {
        throw std::runtime_error("intentional worker failure");
    }

    void set_max_depth(int) override {
    }
};

class SlowIntegrator final : public Integrator {
  public:
    explicit SlowIntegrator(std::atomic<int> &calls) : m_calls(calls) {
    }

    color Li(const ray &, const hittable &, const color &,
             IntegratorContext &) const override {
        ++m_calls;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return color(0.25, 0.5, 0.75);
    }

    void set_max_depth(int) override {
    }

  private:
    std::atomic<int> &m_calls;
};

std::shared_ptr<camera> test_camera(double aspect_ratio) {
    return std::make_shared<camera>(
        point3(0, 0, 2), point3(0, 0, -1), vec3(0, 1, 0), 40.0,
        aspect_ratio, 0.0, 3.0, 0.0, 1.0);
}

RenderRequest test_request(std::uint32_t threads) {
    RenderRequest request;
    request.extent = make_image_extent(16, 9);
    request.integrator = IntegratorKind::Path;
    request.samples_per_pixel = 4;
    request.max_depth = 4;
    request.seed = 123;
    request.threads = threads;
    return request;
}

} // namespace

TEST_CASE(render_request_rejects_invalid_dimensions_and_integrators) {
    bool rejected_width = false;
    try {
        (void)make_image_extent(1, 1);
    } catch (const std::invalid_argument &) {
        rejected_width = true;
    }
    REQUIRE(rejected_width);

    bool rejected_aspect = false;
    try {
        (void)make_image_extent(100, 0.0);
    } catch (const std::invalid_argument &) {
        rejected_aspect = true;
    }
    REQUIRE(rejected_aspect);

    bool rejected_integrator = false;
    try {
        (void)integrator_kind_from_id(5);
    } catch (const std::invalid_argument &) {
        rejected_integrator = true;
    }
    REQUIRE(rejected_integrator);

    RenderRequest request = test_request(1);
    request.samples_per_pixel = 0;
    bool rejected_spp = false;
    try {
        validate_render_request(request);
    } catch (const std::invalid_argument &) {
        rejected_spp = true;
    }
    REQUIRE(rejected_spp);

    request = test_request(1);
    request.max_depth = 0;
    bool rejected_depth = false;
    try {
        validate_render_request(request);
    } catch (const std::invalid_argument &) {
        rejected_depth = true;
    }
    REQUIRE(rejected_depth);
}

TEST_CASE(integrator_ids_map_to_shared_transport_policies) {
    for (int id = 0; id <= 4; ++id) {
        const IntegratorKind kind = integrator_kind_from_id(id);
        const IntegratorDescriptor &descriptor = integrator_descriptor(kind);
        const IntegratorPolicy policy = integrator_policy(kind);
        REQUIRE(valid_integrator_policy(policy));
        REQUIRE(descriptor.id == id);
        REQUIRE(integrator_id(policy.kind) == id);
        REQUIRE(integrator_supported(kind, RenderBackend::CPU));
        REQUIRE(integrator_supported(kind, RenderBackend::CUDA));
    }
    REQUIRE(!integrator_policy(IntegratorKind::Path)
                 .uses_russian_roulette());
    REQUIRE(integrator_policy(IntegratorKind::DirectLighting)
                .uses_direct_lighting());
    REQUIRE(!integrator_policy(IntegratorKind::DirectLighting).uses_mis());
    REQUIRE(integrator_policy(IntegratorKind::MISPath).uses_mis());
}

TEST_CASE(preview_surface_supports_concurrent_publish_and_snapshot) {
    const ImageExtent extent = make_image_extent(32, 18);
    PreviewSurface preview(extent);
    std::atomic<bool> done{false};
    std::thread writer([&]() {
        for (int pass = 0; pass < 64; ++pass) {
            for (int y = 0; y < static_cast<int>(extent.height); ++y) {
                for (int x = 0; x < static_cast<int>(extent.width); ++x) {
                    preview.publish_pixel(x, y, color(0.2, 0.4, 0.6));
                }
            }
        }
        done.store(true, std::memory_order_release);
    });

    do {
        const RenderBuffer snapshot = preview.snapshot();
        REQUIRE(snapshot.width() == static_cast<int>(extent.width));
        REQUIRE(snapshot.height() == static_cast<int>(extent.height));
        for (const color &pixel : snapshot.get_data()) {
            REQUIRE(std::isfinite(pixel.x()));
            REQUIRE(std::isfinite(pixel.y()));
            REQUIRE(std::isfinite(pixel.z()));
        }
    } while (!done.load(std::memory_order_acquire));
    writer.join();
}

TEST_CASE(renderer_propagates_worker_exceptions) {
    Renderer renderer;
    renderer.set_integrator(std::make_shared<ThrowingIntegrator>());
    const auto world = std::make_shared<hittable_list>();

    bool propagated = false;
    try {
        (void)renderer.render(world, test_camera(16.0 / 9.0), color(0, 0, 0),
                              {}, test_request(2));
    } catch (const std::runtime_error &error) {
        propagated = std::string(error.what()) == "intentional worker failure";
    }
    REQUIRE(propagated);
    REQUIRE(!renderer.is_rendering());
}

TEST_CASE(renderer_cancellation_reports_partial_samples) {
    std::atomic<int> calls{0};
    Renderer renderer;
    renderer.set_integrator(std::make_shared<SlowIntegrator>(calls));
    const auto world = std::make_shared<hittable_list>();
    RenderRequest request = test_request(1);
    request.extent = make_image_extent(32, 18);
    request.samples_per_pixel = 32;

    CancellationSource cancellation;
    RenderResult result;
    std::thread rendering([&]() {
        result = renderer.render(world, test_camera(16.0 / 9.0),
                                 color(0, 0, 0), {}, request,
                                 cancellation.token());
    });
    while (calls.load() < 4) {
        std::this_thread::yield();
    }
    cancellation.cancel();
    rendering.join();

    REQUIRE(result.stats.cancelled);
    REQUIRE(result.stats.completed_samples > 0);
    REQUIRE(result.stats.completed_samples < result.stats.requested_samples);
    REQUIRE(result.stats.sample_count == result.stats.completed_samples);
}

TEST_CASE(renderer_seed_is_independent_of_worker_count) {
    auto world = std::make_shared<hittable_list>();
    world->add(std::make_shared<sphere>(
        point3(0, 0, -1), 0.5,
        make_lambertian_material(color(0.7, 0.3, 0.2))));
    const auto cam = test_camera(16.0 / 9.0);

    Renderer single_thread;
    single_thread.set_integrator(
        std::make_shared<CpuPathIntegrator>(IntegratorKind::Path));
    RenderResult one = single_thread.render(world, cam, color(0.1, 0.2, 0.3),
                                             {}, test_request(1));

    Renderer multi_thread;
    multi_thread.set_integrator(
        std::make_shared<CpuPathIntegrator>(IntegratorKind::Path));
    RenderResult four = multi_thread.render(world, cam, color(0.1, 0.2, 0.3),
                                             {}, test_request(4));

    REQUIRE(one.film.pixels().size() == four.film.pixels().size());
    for (std::size_t index = 0; index < one.film.pixels().size(); ++index) {
        const BeautyFilmPixel &a = one.film.pixels()[index];
        const BeautyFilmPixel &b = four.film.pixels()[index];
        REQUIRE(a.sample_count == b.sample_count);
        REQUIRE(a.radiance_sum.x() == b.radiance_sum.x());
        REQUIRE(a.radiance_sum.y() == b.radiance_sum.y());
        REQUIRE(a.radiance_sum.z() == b.radiance_sum.z());
    }
}

TEST_CASE(beauty_film_pfm_contains_linear_sample_average) {
    BeautyFilm film(make_image_extent(2, 2));
    film.set_pixel(0, 0, color(2.0, 4.0, 6.0), 2);
    film.set_pixel(1, 0, color(4.0, 6.0, 8.0), 2);
    film.set_pixel(0, 1, color(3.0, 6.0, 9.0), 3);
    film.set_pixel(1, 1, color(0.0, 0.0, 0.0), 0);

    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        "raytracer_beauty_film_test.pfm";
    film.save_to_pfm(path.string());

    std::ifstream input(path, std::ios::binary);
    std::string magic;
    std::string dimensions;
    std::string scale;
    std::getline(input, magic);
    std::getline(input, dimensions);
    std::getline(input, scale);
    REQUIRE(magic == "PF");
    REQUIRE(dimensions == "2 2");
    REQUIRE(scale == "-1.0");

    std::vector<float> pixels(12);
    input.read(reinterpret_cast<char *>(pixels.data()),
               static_cast<std::streamsize>(pixels.size() * sizeof(float)));
    REQUIRE(input.gcount() ==
            static_cast<std::streamsize>(pixels.size() * sizeof(float)));
    REQUIRE_NEAR(pixels[0], 1.0, 1e-7);
    REQUIRE_NEAR(pixels[1], 2.0, 1e-7);
    REQUIRE_NEAR(pixels[2], 3.0, 1e-7);
    REQUIRE_NEAR(pixels[3], 2.0, 1e-7);
    REQUIRE_NEAR(pixels[4], 3.0, 1e-7);
    REQUIRE_NEAR(pixels[5], 4.0, 1e-7);
    REQUIRE_NEAR(pixels[6], 1.0, 1e-7);
    REQUIRE_NEAR(pixels[7], 2.0, 1e-7);
    REQUIRE_NEAR(pixels[8], 3.0, 1e-7);
    REQUIRE_NEAR(pixels[9], 0.0, 1e-7);
    std::filesystem::remove(path);
}

TEST_CASE(beauty_film_pfm_rejects_non_finite_radiance) {
    BeautyFilm film(make_image_extent(2, 2));
    film.set_pixel(0, 0,
                   color(std::numeric_limits<double>::infinity(), 0.0, 0.0),
                   1);
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        "raytracer_beauty_film_invalid_test.pfm";
    bool rejected = false;
    try {
        film.save_to_pfm(path.string());
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    REQUIRE(rejected);
    std::filesystem::remove(path);
}
