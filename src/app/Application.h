#ifndef APPLICATION_H
#define APPLICATION_H

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include "WindowsApp.h"
#include "renderer.h"
#include "render_buffer.h"

// UI 状态
struct UIState {
    int scene_id = 0;
    int samples_per_pixel = 100;
    bool restart_render = true;
    std::atomic<bool> is_rendering = {false};
    float last_fps = 0.0f;
    float last_ms = 0.0f;
    int image_width = 400;
    int aspect_w = 16;
    int aspect_h = 9;
};

class Application {
public:
    Application(int initial_scene_id);
    ~Application();

    bool init();
    void run();

private:
    void start_render();
    void update_display_from_buffer();
    void render_ui();

private:
    UIState ui_;
    WindowsApp::ptr win_app_;
    Renderer renderer_;
    std::shared_ptr<RenderBuffer> render_buffer_;
    std::thread render_thread_;
    std::vector<unsigned char> image_data_;
    int width_ = 0, height_ = 0;
};

#endif //APPLICATION_H