#version 330 core

// Interpolated values from the vertex shaders
in vec3 fragment_position;
in vec3 observer_direction_cameraspace;
in vec3 fragment_normal;
in vec3 fragment_direction;

// Ouput data
out vec3 color;

// Color
//uniform vec3 cube_color;
uniform vec3 light_color;
uniform vec3 light_direction_cameraspace;
uniform mat3 normal_mv_matrix;

float atan2(float y, float x) {
    return x == 0.0 ? sign(y)*3.14159/2.0 : atan(y, x);
}

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

vec3 colormap(vec3 direction) {
    vec2 xy = normalize(direction.xy);
    float hue = atan2(xy.x, xy.y) / 3.14159 / 2.0;
    if (direction.z > 0.0) {
        return hsv2rgb(vec3(hue, 1.0-direction.z, 1.0));
    } else {
        return hsv2rgb(vec3(hue, 1.0, 1.0+direction.z));
    }
}

vec3 calculate_color(vec3 normal) {
    float diffuse_factor, specular_factor;
    vec3 light_direction_cameraspace_normalized, observer_direction_cameraspace_normalized;
    vec3 halfway_vector;

    vec3 direction_color = colormap(fragment_direction);

    light_direction_cameraspace_normalized = normalize(light_direction_cameraspace);
    observer_direction_cameraspace_normalized = normalize(observer_direction_cameraspace);

    halfway_vector = normalize(light_direction_cameraspace_normalized + observer_direction_cameraspace_normalized);

    diffuse_factor = max(0, dot(light_direction_cameraspace_normalized, normal));
    specular_factor = pow(max(0, dot(halfway_vector, normal)), 100);
    return direction_color * light_color * (0.3 + diffuse_factor) + 0.5 * light_color * specular_factor;
}

void main()
{
    vec3 normal_cameraspace;
    // Calculate transformed normals
    normal_cameraspace = normalize(normal_mv_matrix * fragment_normal);
    // Calculate Color
    color = calculate_color(normal_cameraspace);
}
