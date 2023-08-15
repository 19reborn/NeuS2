/** @file   rotation.cu
 *  @author Yiming Wang <w752531540@gmail.com>
 */

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/multi_stream.h>

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/bounding_box.cuh>


using namespace Eigen;

template <typename T>
__device__ void rotation_6d_to_matrix(
    const T* rotation_6d,
    Matrix3f rotation_matrix
) {
    Vector3f a1(rotation_6d[0], rotation_6d[1], rotation_6d[2]);
    Vector3f a2(rotation_6d[3], rotation_6d[4], rotation_6d[5]);

    Vector3f b1 = a1.normalized();
    Vector3f b2 = (a2 - b1.dot(a2) * b1).normalized();

    Vector3f b3 = b1.cross(b2);

    rotation_matrix(0, 0) = b1[0];
    rotation_matrix(1, 0) = b1[1];
    rotation_matrix(2, 0) = b1[2];

    rotation_matrix(0, 1) = b2[0];
    rotation_matrix(1, 1) = b2[1];
    rotation_matrix(2, 1) = b2[2];

    rotation_matrix(0, 2) = b3[0];
    rotation_matrix(1, 2) = b3[1];
    rotation_matrix(2, 2) = b3[2];
}

template <typename T=float>
__device__ void gradients_for_normalize(
    const Vector3f input,
    const Vector3f gradients_output,
    Vector3f gradients_input
) {

    T input_norm = std::sqrt(input[0] * input[0] + input[1] * input[1] + input[2] * input[2]);
    T input_norm_3 = input_norm * input_norm * input_norm;


    float jacobian_x_x = (input[1] * input[1] + input[2] * input[2] )/ (input_norm_3);
    float jacobian_y_x = - input[0] * input[1] / input_norm_3;
    float jacobian_z_x = - input[0] * input[2] / input_norm_3;

    gradients_input[0] = (gradients_output[0] * jacobian_x_x + gradients_output[1] * jacobian_y_x + gradients_output[2] * jacobian_z_x);

    float jacobian_y_y = (input[0] * input[0] + input[2] * input[2] )/ (input_norm_3);
    float jacobian_x_y = - input[0] * input[1] / input_norm_3;
    float jacobian_z_y = - input[1] * input[2] / input_norm_3;

    gradients_input[1] = (gradients_output[0] * jacobian_x_y + gradients_output[1] * jacobian_y_y + gradients_output[2] * jacobian_z_y);

    float jacobian_z_z = (input[0] * input[0] + input[1] * input[1] )/ (input_norm_3);
    float jacobian_x_z = - input[0] * input[2] / input_norm_3;
    float jacobian_y_z = - input[1] * input[2] / input_norm_3;

    gradients_input[2] = (gradients_output[0] * jacobian_x_z + gradients_output[1] * jacobian_y_z + gradients_output[2] * jacobian_z_z);
} 

template <typename T>
__device__ void gradient_rotation_matrix_to_6d(
    const T* rotation_6d,
    const Matrix3f gradients_matrix,
    T* gradients_6d
) {
    Vector3f a1(rotation_6d[0], rotation_6d[1], rotation_6d[2]);
    Vector3f a2(rotation_6d[3], rotation_6d[4], rotation_6d[5]);

    Vector3f b1 = a1.normalized();
    Vector3f b2 = (a2 - b1.dot(a2) * b1).normalized();

    T d_b1_x = (T)0.f;
    T d_b1_y = (T)0.f;
    T d_b1_z = (T)0.f;

    T d_b2_x = (T)0.f;
    T d_b2_y = (T)0.f;
    T d_b2_z = (T)0.f;

    d_b1_x += gradients_matrix(0, 0);
    d_b1_y += gradients_matrix(1, 0);
    d_b1_z += gradients_matrix(2, 0);

    d_b2_x += gradients_matrix(0, 1);
    d_b2_y += gradients_matrix(1, 1);
    d_b2_z += gradients_matrix(2, 1);

    d_b1_x += b2[1] * gradients_matrix(2, 2) - b2[2] * gradients_matrix(1, 2); // (-b1.x * b2.z)j + (b1.x * b2.y)k
    d_b1_y += b2[2] * gradients_matrix(0, 2) - b2[0] * gradients_matrix(2, 2); // (b1.y * b2.z)i - (b1.y * b2.x)k
    d_b1_z += b2[0] * gradients_matrix(1, 2) - b2[1] * gradients_matrix(0, 2); // (b2.x * b1.z)j - (b1.z * b2.y)i
    
    d_b2_x += b1[2] * gradients_matrix(1, 2) - b1[1] * gradients_matrix(2, 2); // (b1.z * b2.x)j - (b1.y * b2.x)k
    d_b2_y += b1[0] * gradients_matrix(2, 2) - b1[2] * gradients_matrix(0, 2); // (b1.x * b2.y)k - (b1.z * b2.y)i
    d_b2_z += b1[1] * gradients_matrix(0, 2) - b1[0] * gradients_matrix(1, 2); // (b1.y * b2.z)i - (b1.x * b2.z)j

    Vector3f d_a1(0.f, 0.f , 0.f);
    Vector3f d_a2(0.f, 0.f , 0.f);

    Vector3f d_b2_remove_normalized;
    Vector3f d_b1(d_b1_x, d_b1_y, d_b1_z);
    Vector3f d_b2(d_b2_x, d_b2_y, d_b2_z);

    gradients_for_normalize(a2 - b1.dot(a2) * b1, d_b2, d_b2_remove_normalized);

    d_a2 += d_b2_remove_normalized;
    d_a2[0] += - d_b2_remove_normalized[0] * b1[0] * b1[0] - d_b2_remove_normalized[1] * b1[0] * b1[1] -  d_b2_remove_normalized[2] * b1[0] * b1[2];
    d_a2[1] += - d_b2_remove_normalized[0] * b1[0] * b1[1] - d_b2_remove_normalized[1] * b1[1] * b1[1] -  d_b2_remove_normalized[2] * b1[2] * b1[1];
    d_a2[2] += - d_b2_remove_normalized[0] * b1[0] * b1[2] - d_b2_remove_normalized[1] * b1[1] * b1[2] -  d_b2_remove_normalized[2] * b1[2] * b1[2];


    d_b1[0] += - d_b2_remove_normalized[0] * (2.0f * b1[0] * a2[0] + b1[1] * a2[1] + b2[2] * a2[2]) - d_b2_remove_normalized[1] * b1[1] * a2[0] - d_b2_remove_normalized[2] * b1[2] * a2[0];
    d_b1[1] += - d_b2_remove_normalized[1] * (2.0f * b1[1] * a2[1] + b1[0] * a2[0] + b2[2] * a2[2]) - d_b2_remove_normalized[0] * b1[0] * a2[1] - d_b2_remove_normalized[2] * b1[2] * a2[1];
    d_b1[2] += - d_b2_remove_normalized[2] * (2.0f * b1[2] * a2[2] + b1[0] * a2[0] + b2[1] * a2[1]) - d_b2_remove_normalized[0] * b1[0] * a2[2] - d_b2_remove_normalized[1] * b1[1] * a2[2];

    gradients_for_normalize(a1, d_b1, d_a1);

    gradients_6d[0] = d_a1[0];
    gradients_6d[1] = d_a1[1];
    gradients_6d[2] = d_a1[2];

    gradients_6d[3] = d_a2[0];
    gradients_6d[4] = d_a2[1];
    gradients_6d[5] = d_a2[2];
}

