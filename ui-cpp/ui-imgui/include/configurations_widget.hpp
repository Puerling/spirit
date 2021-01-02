#pragma once
#ifndef SPIRIT_IMGUI_CONFIGURATIONS_WIDGET_HPP
#define SPIRIT_IMGUI_CONFIGURATIONS_WIDGET_HPP

#include <rendering_layer.hpp>

#include <memory>

struct State;

namespace ui
{

struct ConfigurationsWidget
{
    ConfigurationsWidget( bool & show, std::shared_ptr<State> state, RenderingLayer & rendering_layer );
    void reset_settings();
    void show();
    void update_data();

    bool & show_;

    std::shared_ptr<State> state;
    RenderingLayer & rendering_layer;

    float pos[3]{ 0, 0, 0 };
    float border_rect[3]{ -1, -1, -1 };
    float border_cyl = -1;
    float border_sph = -1;
    bool inverted    = false;

    float temperature = 0;

    float sk_radius = 15;
    float sk_speed  = 1;
    float sk_phase  = 0;
    bool sk_up_down = false;
    bool sk_achiral = false;
    bool sk_rl      = false;

    float hopfion_radius = 10;
    int hopfion_order    = 1;

    float spiral_angle    = 0;
    float spiral_axis[3]  = { 0, 0, 1 };
    float spiral_qmag     = 1;
    float spiral_qvec[3]  = { 1, 0, 0 };
    bool spiral_q2        = false;
    float spiral_qmag2    = 1;
    float spiral_qvec2[3] = { 1, 0, 0 };
};

} // namespace ui

#endif