#include "test_harness.h"

#include "cpu_render_session.h"
#include "render_session.h"
#include "scene_ir.h"

#include <memory>

namespace {

SceneIR make_simple_ir() {
    SceneIR ir;
    ir.camera.lookfrom = point3(0, 0, 3);
    ir.camera.lookat = point3(0, 0, 0);
    ir.camera.vfov = 40.0;
    ir.preset.image_width = 8;
    ir.preset.samples_per_pixel = 1;
    ir.preset.background = color(0.1, 0.2, 0.3);

    ConstantTextureIR albedo;
    albedo.value = color(0.8, 0.8, 0.8);
    TextureIRNode texture{"white", albedo};
    ir.textures.push_back(texture);

    LambertianMaterialIR lambertian;
    lambertian.albedo = 0;
    MaterialIR material{"white", lambertian};
    ir.materials.push_back(material);

    SphereObjectIR sphere;
    sphere.center = point3(0, 0, 0);
    sphere.radius = 1.0;
    sphere.material = "white";
    ObjectIRNode node{"sphere", sphere};
    ir.object_nodes.push_back(node);
    ir.objects.push_back(0);
    return ir;
}

RenderRequest make_request() {
    RenderRequest request;
    request.extent = make_image_extent(8, 8);
    request.integrator = IntegratorKind::MISPath;
    request.samples_per_pixel = 1;
    request.max_depth = 2;
    request.seed = 123;
    request.threads = 1;
    return request;
}

} // namespace

TEST_CASE(cpu_session_accepts_camera_and_scene_revisions) {
    const SceneIR ir = make_simple_ir();
    std::unique_ptr<IRenderSession> session =
        make_cpu_render_session(ir);

    RenderFrameRequest frame;
    frame.render = make_request();
    frame.camera = ir.camera;
    frame.frame_index = 1;
    frame.revision.camera = 1;
    (void)session->render_frame(frame, {});

    frame.frame_index = 2;
    frame.revision.camera = 2;
    frame.revision.geometry = 1;
    frame.revision.material = 1;
    frame.revision.lighting = 1;
    const RenderResult result = session->render_frame(frame, {});
    REQUIRE(result.stats.base.completed_samples ==
            result.stats.base.requested_samples);
}
