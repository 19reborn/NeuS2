/** @file   common_operation.h
 *  @author Yiming Wang <w752531540@gmail.com>
 */

#pragma once

#include <tiny-cuda-nn/common.h>

#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/multi_stream.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/reduce_sum.h>

#include <neural-graphics-primitives/trainable_buffer.cuh>


using namespace Eigen;


template <typename T, typename TInput = T>
__global__ void extract_delta_xyz(
	const uint32_t n_elements,
	const uint32_t n_pos_dim,
	const tcnn::MatrixView<TInput> delta,
	const tcnn::MatrixView<T> pos,
	tcnn::MatrixView<T> dst
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i / n_pos_dim;
	const uint32_t dim_idx = i - elem_idx * n_pos_dim;
	// printf("delta:%f\n",(float)delta(dim_idx, elem_idx));
	dst(dim_idx, elem_idx) = pos(dim_idx, elem_idx) + (T)delta(dim_idx, elem_idx);
	// dst(dim_idx, elem_idx) = pos(dim_idx, elem_idx);
}

template <typename T>
__device__ void rotation_6d_to_matrix(
    const T* rotation_6d,
    Matrix3f& rotation_matrix
) {
    Vector3f a1((float)rotation_6d[0], (float)rotation_6d[1], (float)rotation_6d[2]);
    Vector3f a2((float)rotation_6d[3], (float)rotation_6d[4], (float)rotation_6d[5]);
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
    Vector3f& gradients_input
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
	// const Eigen:Matrix<float, 3, 3 ,Eigen::RowMajor> gradients_matrix, 
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


    d_b1[0] += - d_b2_remove_normalized[0] * (2.0f * b1[0] * a2[0] + b1[1] * a2[1] + b1[2] * a2[2]) - d_b2_remove_normalized[1] * b1[1] * a2[0] - d_b2_remove_normalized[2] * b1[2] * a2[0];
    d_b1[1] += - d_b2_remove_normalized[1] * (2.0f * b1[1] * a2[1] + b1[0] * a2[0] + b1[2] * a2[2]) - d_b2_remove_normalized[0] * b1[0] * a2[1] - d_b2_remove_normalized[2] * b1[2] * a2[1];
    d_b1[2] += - d_b2_remove_normalized[2] * (2.0f * b1[2] * a2[2] + b1[0] * a2[0] + b1[1] * a2[1]) - d_b2_remove_normalized[0] * b1[0] * a2[2] - d_b2_remove_normalized[1] * b1[1] * a2[2];

    gradients_for_normalize(a1, d_b1, d_a1);

    gradients_6d[0] = d_a1[0];
    gradients_6d[1] = d_a1[1];
    gradients_6d[2] = d_a1[2];

    gradients_6d[3] = d_a2[0];
    gradients_6d[4] = d_a2[1];
    gradients_6d[5] = d_a2[2];
}



template <typename T>
tcnn::MatrixView<T> get_advance(tcnn::MatrixView<T> in_matrix_view, uint32_t m, uint32_t n) {
	in_matrix_view.advance(m, n);
	return in_matrix_view;
}

template <typename T, typename TInput = T>
__global__ void fill_positions_view(
	const uint32_t n_elements,
	const uint32_t n_pos_dim,
	const tcnn::MatrixView<TInput> pos,
	tcnn::MatrixView<T> dst
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i / n_pos_dim;
	const uint32_t dim_idx = i - elem_idx * n_pos_dim;
	dst(dim_idx, elem_idx) = (T)pos(dim_idx, elem_idx);
}

template <typename T, typename TInput = T>
__global__ void fill_positions_view_with_fixed_offset(
	const uint32_t n_elements,
	const uint32_t n_pos_dim,
	const tcnn::MatrixView<TInput> pos,
	tcnn::MatrixView<T> dst
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i / n_pos_dim;
	const uint32_t dim_idx = i - elem_idx * n_pos_dim;
	dst(dim_idx, elem_idx) = (T)pos(dim_idx, elem_idx) - (T)0.5f;
}

template <typename T, typename TInput = T>
__global__ void fill_positions_view_drgb_dnormal(
	const uint32_t n_elements,
	const uint32_t n_pos_dim,
	const tcnn::MatrixView<TInput> pos,
	const tcnn::MatrixView<T> sdf,
	tcnn::MatrixView<T> dst
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i / n_pos_dim;
	const uint32_t dim_idx = i - elem_idx * n_pos_dim;
	
	float value_norm = 0.0f;
	for (uint32_t j = 0; j < n_pos_dim; j++){
		value_norm += sdf(dim_idx, j) * sdf(dim_idx, j);
	}
	value_norm = sqrt(value_norm + 1e-6);


	dst(dim_idx, elem_idx) = (T)pos(dim_idx, elem_idx) / (T)value_norm;
}

template <typename T, typename TInput = T>
__global__ void fill_positions_view_with_normalized_norm(
	const uint32_t n_elements,
	const uint32_t n_pos_dim,
	const tcnn::MatrixView<TInput> pos,
	tcnn::MatrixView<T> dst
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i / n_pos_dim;
	const uint32_t dim_idx = i - elem_idx * n_pos_dim;

	float value_norm = 0.0f;
	for (uint32_t j = 0; j < n_pos_dim; j++){
		value_norm += pos(dim_idx, j) * pos(dim_idx, j);
	}
	value_norm = sqrt(value_norm + 1e-6);

	dst(dim_idx, elem_idx) = (T)pos(dim_idx, elem_idx) / (T)value_norm;

}

template <typename T, typename TInput = T>
__global__ void add_positions_view(
	const uint32_t n_elements,
	const uint32_t n_pos_dim,
	const tcnn::MatrixView<TInput> pos,
	tcnn::MatrixView<T> dst
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i / n_pos_dim;
	const uint32_t dim_idx = i - elem_idx * n_pos_dim;

	dst(dim_idx, elem_idx) += (T)pos(dim_idx, elem_idx);
}

template <typename T, typename TInput = T>
__global__ void add_l1_regulization(
	const uint32_t n_elements,
	const uint32_t n_pos_dim,
	const T loss_weight, // includeing * loss_scale, * loss_weight, / batch_size
	const tcnn::MatrixView<T> xyz,
	const tcnn::MatrixView<T> xyz_delta,
	tcnn::MatrixView<T> dst
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i / n_pos_dim;
	const uint32_t dim_idx = i - elem_idx * n_pos_dim;

	dst(dim_idx, elem_idx) += abs((T)xyz_delta(dim_idx, elem_idx) - (T)xyz(dim_idx, elem_idx)) * loss_weight;
}

template <typename T, typename TInput = T>
__global__ void add_positions_view_ekloss(
	const uint32_t n_elements,
	const uint32_t n_pos_dim,
	const uint32_t batch_size,
	const tcnn::MatrixView<TInput> pos,
	tcnn::MatrixView<T> dst
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i / n_pos_dim;
	const uint32_t dim_idx = i - elem_idx * n_pos_dim;
	dst(dim_idx, elem_idx) += (T)(pos(dim_idx, elem_idx)) / (T)batch_size ;
}

template <typename T>
__global__ void sdf_add_bias(
	const uint32_t  n_elements,
	T bias,
	tcnn::MatrixView<T> sdf_network_output
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	sdf_network_output(0, i) += bias;
}

template <typename T>
__global__ void sdf_to_density_variance_buffer(
	const uint32_t  n_elements,
	const T* variance_output,
	tcnn::MatrixView<T> sdf_network_output
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	T sdf = sdf_network_output(0, i);
	#if SDF_GRID
		sdf_network_output(0, i) = T(1.0f/(abs((float)sdf) + 1e-4));
	#else
		T variance = variance_output[0];
		T s = (T)__expf(variance*(T)10.0f);
		T sigmoid_sdf = tcnn::logistic(sdf * s);
		T density = s * sigmoid_sdf * (T(1.0f) - sigmoid_sdf);
		sdf_network_output(0, i) = density;
	#endif
}

template <typename T, typename TIn = T>
__global__ void extract_dSDF_dPos_view(
	const uint32_t n_elements,
	const tcnn::MatrixView<TIn> dSDF_dPos,
	tcnn::MatrixView<T> rgbd
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i / 3;
	const uint32_t dim_idx = i - elem_idx * 3;

	rgbd(dim_idx, elem_idx) = dSDF_dPos(dim_idx, elem_idx);
}


template <typename T, typename TIn = T>
__global__ void extract_single_variance_view(
	const uint32_t n_elements,
	const T* variance,
	tcnn::MatrixView<T> rgbd
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	rgbd(7, i) = variance[0];
}

template <typename T, typename TIn = T>
__global__ void add_global_movement(
	const uint32_t n_elements,
	const TIn* transition,
	const tcnn::MatrixView<T> input,
	tcnn::MatrixView<T> output
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i / 3;
	const uint32_t dim_idx = i - elem_idx * 3;

	output(dim_idx, elem_idx) = input(dim_idx, elem_idx) + (T)transition[dim_idx];
}

// __device__ Vector3f my_warp_direction(const Vector3f& dir) {
// 	return (dir + Vector3f::Ones()) * 0.5f;
// }

// __device__ Vector3f my_unwarp_direction(const Vector3f& dir) {
// 	return dir * 2.0f - Vector3f::Ones();
// }

// __device__ Vector3f warp_direction_derivative(const Vector3f& dir) {
// 	return Vector3f::Constant(0.5f);
// }

// __device__ Vector3f unwarp_direction_derivative(const Vector3f& dir) {
// 	return Vector3f::Constant(2.0f);
// }

template <typename T, typename TIn = T>
__global__ void add_global_movement_with_rotation_quaternion(
	const uint32_t n_elements,
	const TIn* rotation_quat,
	const TIn* transition,
	const tcnn::MatrixView<T> input,
	tcnn::MatrixView<T> output
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	// T qx = rotation_quat[0], qy = rotation_quat[1], qz = rotation_quat[2], qw = rotation_quat[3];

	TIn rotation_norm = std::sqrt((float)(rotation_quat[0]*rotation_quat[0] + rotation_quat[1]*rotation_quat[1] + rotation_quat[2]*rotation_quat[2] + rotation_quat[3]*rotation_quat[3]));
	T qx = rotation_quat[0] / rotation_norm, qy = rotation_quat[1] / rotation_norm, qz = rotation_quat[2] / rotation_norm, qw = rotation_quat[3] / rotation_norm;

	T vx = input(0, i), vy = input(1, i), vz = input(2, i);
	T tx = transition[0], ty = transition[1], tz = transition[2];

	output(0, i) = qw*(2*qy*vz - 2*qz*vy) + qy*(2*qx*vy - 2*qy*vx) - qz*(-2*qx*vz + 2*qz*vx) + vx + tx;
	output(1, i) = qw*(-2*qx*vz + 2*qz*vx) - qx*(2*qx*vy - 2*qy*vx) + qz*(2*qy*vz - 2*qz*vy) + vy + ty;
	output(2, i) = qw*(2*qx*vy - 2*qy*vx) + qx*(-2*qx*vz + 2*qz*vx) - qy*(2*qy*vz - 2*qz*vy) + vz + tz;

	// if (i == 0){
	// 	printf("rotation:%f,%f,%f,%f\n",(float)rotation_quat[0],(float)rotation_quat[1],(float)rotation_quat[2],(float)rotation_quat[3]);
	// 	printf("transition:%f,%f,%f\n",(float)transition[0],(float)transition[1],(float)transition[2]);
	// }

}

template <typename T, typename TIn = T>
__global__ void add_global_movement_with_rotation_6d(
	const uint32_t n_elements,
	const TIn* rotation_6d,
	const TIn* transition,
	const Eigen::Vector3f first_frame_offset,
	const tcnn::MatrixView<T> input,
	tcnn::MatrixView<T> output
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	Eigen::Matrix3f rotation_matrix;
	rotation_6d_to_matrix<TIn>(rotation_6d, rotation_matrix);
    // if (i==0){
	// 	// printf("rotation_6d:%f,%f,%f,%f,%f,%f\n",(float)rotation_6d[0],(float)rotation_6d[1],(float)rotation_6d[2],(float)rotation_6d[3],(float)rotation_6d[4],(float)rotation_6d[5]);
    //     printf("%f,%f,%f\n",rotation_matrix(0,0),rotation_matrix(0,1),rotation_matrix(0,2));
    //     printf("%f,%f,%f\n",rotation_matrix(1,0),rotation_matrix(1,1),rotation_matrix(1,2));
    //     printf("%f,%f,%f\n",rotation_matrix(2,0),rotation_matrix(2,1),rotation_matrix(2,2));
    // }
	Eigen::Vector3f pos(input(0, i), input(1, i), input(2, i));


	pos[0] += (float)transition[0] - first_frame_offset[0];
	pos[1] += (float)transition[1] - first_frame_offset[1];
	pos[2] += (float)transition[2] - first_frame_offset[2];

	// pos = rotation_matrix * pos;
	pos = rotation_matrix * pos + first_frame_offset;
	// if ( i == 0)
	// 	printf("first_frame_offset:%f,%f,%f\n",first_frame_offset[0],first_frame_offset[1],first_frame_offset[2]);

	output(0, i) = (T)pos[0];
	output(1, i) = (T)pos[1];
	output(2, i) = (T)pos[2];

	Eigen::Vector3f dir(input(3, i), input(4, i), input(5, i));
	
	// if (i == 0){
		// printf("original_dir:%f,%f,%f\n",(float)dir[0],(float)dir[1],(float)dir[2]);
	// }
	dir = dir * 2.0f - Vector3f::Ones();

	dir = rotation_matrix * dir;
	// if ( i == 0)
	// 	printf("first_frame_offset:%f,%f,%f\n",first_frame_offset[0],first_frame_offset[1],first_frame_offset[2]);
	
	dir = (dir + Vector3f::Ones()) * 0.5f;

	// if (i == 0){
	// 	printf("new_dir:%f,%f,%f\n",(float)dir[0],(float)dir[1],(float)dir[2]);
	// }
	output(3, i) = (T)dir[0];
	output(4, i) = (T)dir[1];
	output(5, i) = (T)dir[2];

	// output(0, i) = (T)pos[0];
	// output(1, i) = (T)pos[1];
	// output(2, i) = (T)pos[2];

	// pos = rotation_matrix * pos;

	// output(0, i) = (TIn)pos[0] + transition[0];
	// output(1, i) = (TIn)pos[1] + transition[1];
	// output(2, i) = (TIn)pos[2] + transition[2];

	// if (i == 0){
	// 	printf("rotation:%f,%f,%f,%f\n",(float)rotation_quat[0],(float)rotation_quat[1],(float)rotation_quat[2],(float)rotation_quat[3]);
	// 	printf("transition:%f,%f,%f\n",(float)transition[0],(float)transition[1],(float)transition[2]);
	// }

	// if (i == 0){
	// 	printf("input:%f,%f,%f\n", input(0,i), input(1,i), input(2,i));
	// 	printf("output:%f,%f,%f\n", output(0,i), output(1,i), output(2,i));
	// }

}

// template <typename T, typename TIn = T>
// __global__ void add_global_movement_with_rotation_inplace(
// 	const uint32_t n_elements,
// 	const TIn* rotation_quat,
// 	const TIn* transition,
// 	tcnn::MatrixView<T> xyz
// ) {
// 	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
// 	if (i >= n_elements) return;

// 	T qx = rotation_quat[0], qy = rotation_quat[1], qz = rotation_quat[2], qw = rotation_quat[3];
// 	T vx = xyz(0, i), vy = xyz(1, i), vz = xyz(2, i);
// 	T tx = transition[0], ty = transition[1], tz = transition[2];

// 	xyz(0, i) = qw*(2*qy*vz - 2*qz*vy) + qy*(2*qx*vy - 2*qy*vx) - qz*(-2*qx*vz + 2*qz*vx) + vx + tx;
// 	xyz(1, i) = qw*(-2*qx*vz + 2*qz*vx) - qx*(2*qx*vy - 2*qy*vx) + qz*(2*qy*vz - 2*qz*vy) + vy + ty;
// 	xyz(2, i) = qw*(2*qx*vy - 2*qy*vx) + qx*(-2*qx*vz + 2*qz*vx) - qy*(2*qy*vz - 2*qz*vy) + vz + tz;
// }


template <typename T>
__global__ void accumulate_global_movement_rotation_quaternion_kernel(
	uint32_t n_elements,
	const T* cur_rotation,
	const T* cur_transition,
	T* accumulated_rotation,
	T* accumulated_transition
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) { return; }

	// accumulated_rotation_new * pos + accumulated_transition_new =  cur_rotation * (accmulated_rotation * pos + accmulated_transition) + cur_transition
	// accumulated_rotation_new = cur_rotation * accumulated_rotation
	// accumulated_transition_new = cur_rotation * accumulated_transition + cur_transition

	T rotation_norm = std::sqrt((float)(cur_rotation[0]*cur_rotation[0] + cur_rotation[1]*cur_rotation[1] + cur_rotation[2]*cur_rotation[2] + cur_rotation[3]*cur_rotation[3]));
	T x1 = cur_rotation[0] / rotation_norm, y1 = cur_rotation[1] / rotation_norm, z1 = cur_rotation[2] / rotation_norm, w1 = cur_rotation[3] / rotation_norm;

	T x2 = accumulated_rotation[0], y2 = accumulated_rotation[1], z2 = accumulated_rotation[2], w2 = accumulated_rotation[3];
	// T x2 = cur_rotation[0], y2 = cur_rotation[1], z2 = cur_rotation[2], w2 = cur_rotation[3];
	// T x1 = accumulated_rotation[0], y1 = accumulated_rotation[1], z1 = accumulated_rotation[2], w1 = accumulated_rotation[3];
	accumulated_rotation[0] = + w1 * x2 - z1 * y2 + y1 * z2 + x1 * w2;
	accumulated_rotation[1] = + z1 * x2 + w1 * y2 - x1 * z2 + y1 * w2;
	accumulated_rotation[2] = - y1 * x2 + x1 * y2 + w1 * z2 + z1 * w2;
	accumulated_rotation[3] = - x1 * x2 - y1 * y2 - z1 * z2 + w1 * w2;
	
	T qx = x1, qy = y1, qz = z1, qw = w1;
	T vx = accumulated_transition[0], vy = accumulated_transition[1], vz = accumulated_transition[2];
	T tx = cur_transition[0], ty = cur_transition[1], tz = cur_transition[2];

	accumulated_transition[0] = qw*((T)2.0f*qy*vz - (T)2.0f*qz*vy) + qy*((T)2.0f*qx*vy - (T)2.0f*qy*vx) - qz*(-(T)2.0f*qx*vz + (T)2.0f*qz*vx) + vx + tx;
	accumulated_transition[1] = qw*(-(T)2.0f*qx*vz + (T)2.0f*qz*vx) - qx*((T)2.0f*qx*vy - (T)2.0f*qy*vx) + qz*((T)2.0f*qy*vz - (T)2.0f*qz*vy) + vy + ty;
	accumulated_transition[2] = qw*((T)2.0f*qx*vy - (T)2.0f*qy*vx) + qx*(-(T)2.0f*qx*vz + (T)2.0f*qz*vx) - qy*((T)2.0f*qy*vz - (T)2.0f*qz*vy) + vz + tz;

}

template <typename T>
__global__ void accumulate_global_movement_rotation_6d_kernel(
	uint32_t n_elements,
	const T* cur_rotation,
	const T* cur_transition,
	T* accumulated_rotation,
	T* accumulated_transition
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) { return; }

	Eigen::Matrix3f rotation_matrix;
	rotation_6d_to_matrix<T>(cur_rotation, rotation_matrix);

	// Eigen::Matrix3f before_rotation_matrix = Eigen::Map<Eigen::Matrix<T, 3, 3 ,Eigen::RowMajor>>(accumulated_rotation);
	// Eigen::Matrix3f before_rotation_matrix;
	Eigen::Matrix<float, 3, 3 ,Eigen::RowMajor> before_rotation_matrix;
	before_rotation_matrix << accumulated_rotation[0], accumulated_rotation[1], accumulated_rotation[2],
							  accumulated_rotation[3], accumulated_rotation[4], accumulated_rotation[5],
							  accumulated_rotation[6], accumulated_rotation[7], accumulated_rotation[8];

	Eigen::Matrix<float, 3, 3 ,Eigen::RowMajor> after_rotation_matrix = rotation_matrix * before_rotation_matrix;
	for (uint32_t iter_i = 0; iter_i < 9; iter_i++)
		accumulated_rotation[iter_i] = after_rotation_matrix.data()[iter_i];

	Eigen::Vector3f cur_transition_vector(cur_transition[0],cur_transition[1],cur_transition[2]);
	Eigen::Vector3f before_transition_vector(accumulated_transition[0],accumulated_transition[1],accumulated_transition[2]);
	Eigen::Vector3f after_transition_vector;

	after_transition_vector = rotation_matrix * (before_transition_vector + cur_transition_vector);
	
	accumulated_transition[0] = after_transition_vector[0];
	accumulated_transition[1] = after_transition_vector[1];
	accumulated_transition[2] = after_transition_vector[2];

}

template <typename T>
__global__ void save_global_movement_rotation_6d_kernel(
	uint32_t n_elements,
	const T* cur_rotation,
	const T* cur_transition,
	const T* accumulated_rotation,
	const T* accumulated_transition,
	T* saved_rotation,
	T* saved_transition
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) { return; }

	Eigen::Matrix3f rotation_matrix;
	rotation_6d_to_matrix<T>(cur_rotation, rotation_matrix);

	// Eigen::Matrix3f before_rotation_matrix = Eigen::Map<Eigen::Matrix<T, 3, 3 ,Eigen::RowMajor>>(accumulated_rotation);
	// Eigen::Matrix3f before_rotation_matrix;
	Eigen::Matrix<float, 3, 3 ,Eigen::RowMajor> before_rotation_matrix;
	before_rotation_matrix << accumulated_rotation[0], accumulated_rotation[1], accumulated_rotation[2],
							  accumulated_rotation[3], accumulated_rotation[4], accumulated_rotation[5],
							  accumulated_rotation[6], accumulated_rotation[7], accumulated_rotation[8];

	Eigen::Matrix<float, 3, 3 ,Eigen::RowMajor> after_rotation_matrix = rotation_matrix * before_rotation_matrix;
	for (uint32_t iter_i = 0; iter_i < 9; iter_i++)
		saved_rotation[iter_i] = after_rotation_matrix.data()[iter_i];

	Eigen::Vector3f cur_transition_vector(cur_transition[0],cur_transition[1],cur_transition[2]);
	Eigen::Vector3f before_transition_vector(accumulated_transition[0],accumulated_transition[1],accumulated_transition[2]);
	Eigen::Vector3f after_transition_vector;

	after_transition_vector = rotation_matrix * (before_transition_vector + cur_transition_vector);
	
	saved_transition[0] = after_transition_vector[0];
	saved_transition[1] = after_transition_vector[1];
	saved_transition[2] = after_transition_vector[2];
}

template <typename T, typename TIn = T>
__global__ void extract_deformed_pos(
	const uint32_t n_elements,
	const tcnn::MatrixView<TIn> deformed_pos,
	tcnn::MatrixView<T> output
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i / 3;
	const uint32_t dim_idx = i - elem_idx * 3;

	output(dim_idx, elem_idx) = deformed_pos(dim_idx, elem_idx);
}

template <typename T>
__global__ void set_constant_value_view(
	const uint32_t n_elements,
	const T value,
	tcnn::MatrixView<T> output
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	output(0, i) = value;
}

template <typename T>
__global__ void set_constant_value_view_vector_test(
	const uint32_t n_elements,
	const uint32_t n_pos_dim,
	const T value,
	tcnn::MatrixView<T> output
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i / n_pos_dim;
	const uint32_t dim_idx = i - elem_idx * n_pos_dim;

	output(dim_idx, elem_idx) = value;
}

template <typename T>
__global__ void add_variance_view(
	const uint32_t n_elements,
	const tcnn::MatrixView<T> rgbd,
	tcnn::MatrixView<T> mean_value
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	mean_value(0,0) += rgbd(7, i);
	// (float)rgbd(7, i);
}

template <typename T>
__global__ void add_variance_view_to_loss(
	const uint32_t n_elements,
	const tcnn::MatrixView<T> rgbd,
	tcnn::MatrixView<T> mean_value
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	// mean_value(0,i) += rgbd(7, i);
	mean_value(0,i) = rgbd(7, i);
	// (float)rgbd(7, i);
}

template <typename T, typename TIn = T>
__global__ void add_transition_view_to_loss(
	const uint32_t n_elements,
	const tcnn::MatrixView<TIn> loss,
	const uint32_t dim,
	tcnn::MatrixView<T> transition_gradient
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	// mean_value(0,i) += rgbd(7, i);
	transition_gradient(0,i) = (T)loss(dim, i);
	// (float)rgbd(7, i);
}

template <typename T, typename TIn = T>
__global__ void add_rotation_view_to_loss(
	const uint32_t n_elements,
	const tcnn::MatrixView<TIn> loss,
	const uint32_t dim,
	tcnn::MatrixView<T> rotation_gradient
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	rotation_gradient(0,i) = (T)loss(dim, i);
}

template <typename T, typename TIn = T>
__global__ void add_loss_to_rotation_quaternion_each(
	const uint32_t n_elements,
	const tcnn::MatrixView<TIn> loss,
	const tcnn::MatrixView<TIn> input,
	const T* rotation,
	tcnn::MatrixView<T> rotation_gradient,
	tcnn::MatrixView<T> input_gradient
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	T qx_original = rotation[0], qy_original = rotation[1], qz_original = rotation[2], qw_original = rotation[3];

	float rotation_norm_2 = (float)(rotation[0]*rotation[0] + rotation[1]*rotation[1] + rotation[2]*rotation[2] + rotation[3]*rotation[3]);
	T rotation_norm = std::sqrt(rotation_norm_2);
	T rotation_norm_pow_3_inv = 1.0f / std::pow(rotation_norm, 3.f) ;
	T qx = rotation[0] / rotation_norm, qy = rotation[1] / rotation_norm, qz = rotation[2] / rotation_norm, qw = rotation[3] / rotation_norm;

	T vx = input(0,i), vy = input(1,i), vz = input(2,i);
	T dgradient_x = (T) (loss(0, i)) * ((T)2.0f*qy*vy + (T)2.0f*qz*vz)              + (T) (loss(1, i)) * (-(T)2.0f*qw*vz - (T)4.0f*qx*vy + (T)2.0f*qy*vx)    + (T) (loss(2, i)) * ((T)2.0f*qw*vy - (T)4.0f*qx*vz + (T)2.0f*qz*vx);
	T dgradient_y = (T) (loss(0, i)) * ((T)2.0f*qw*vz + (T)2.0f*qx*vy - (T)4.0f*qy*vx)    + (T) (loss(1, i)) * ((T)2.0f*qx*vx + (T)2.0f*qz*vz)               + (T) (loss(2, i)) * (-(T)2.0f*qw*vx - (T)4.0f*qy*vz + (T)2.0f*qz*vy);
	T dgradient_z = (T) (loss(0, i)) * (-(T)2.0f*qw*vy + (T)2.0f*qx*vz - (T)4.0f*qz*vx)   + (T) (loss(1, i)) * ((T)2.0f*qw*vx + (T)2.0f*qy*vz - (T)4.0f*qz*vy)     + (T) (loss(2, i)) * ((T)2.0f*qx*vx + (T)2.0f*qy*vy);
	T dgradient_w = (T) (loss(0, i)) * ((T)2.0f*qy*vz - (T)2.0f*qz*vy)              + (T) (loss(1, i)) * (-(T)2.0f*qx*vz + (T)2.0f*qz*vx)              + (T) (loss(2, i)) * ((T)2.0f*qx*vy - (T)2.0f*qy*vx);

		
	T jacobian_x_x = ((T)rotation_norm_2 - (T) qx_original * qx_original ) * rotation_norm_pow_3_inv;
	T jacobian_y_x = - qx_original * qy_original * rotation_norm_pow_3_inv;
	T jacobian_z_x = - qx_original * qz_original * rotation_norm_pow_3_inv;
	T jacobian_w_x = - qx_original * qw_original * rotation_norm_pow_3_inv;

	rotation_gradient(0, i) = (dgradient_x * jacobian_x_x + dgradient_y * jacobian_y_x + dgradient_z * jacobian_z_x + dgradient_w * jacobian_w_x);
		
	T jacobian_y_y = ((T)rotation_norm_2 - (T) qy_original * qy_original ) * rotation_norm_pow_3_inv;
	T jacobian_x_y = - qx_original * qy_original * rotation_norm_pow_3_inv;
	T jacobian_z_y = - qy_original * qz_original * rotation_norm_pow_3_inv;
	T jacobian_w_y = - qy_original * qw_original * rotation_norm_pow_3_inv;

	rotation_gradient(1, i) = (dgradient_x * jacobian_x_y + dgradient_y * jacobian_y_y + dgradient_z * jacobian_z_y + dgradient_w * jacobian_w_y);

	T jacobian_z_z = ((T)rotation_norm_2 - (T) qz_original * qz_original ) * rotation_norm_pow_3_inv;
	T jacobian_x_z = - qx_original * qz_original * rotation_norm_pow_3_inv;
	T jacobian_y_z = - qy_original * qz_original * rotation_norm_pow_3_inv;
	T jacobian_w_z = - qz_original * qw_original * rotation_norm_pow_3_inv;

	rotation_gradient(2, i) = (dgradient_x * jacobian_x_z + dgradient_y * jacobian_y_z + dgradient_z * jacobian_z_z + dgradient_w * jacobian_w_z);

	T jacobian_w_w = ((T)rotation_norm_2 - (T) qw_original * qw_original ) * rotation_norm_pow_3_inv;
	T jacobian_x_w = - qx_original * qw_original * rotation_norm_pow_3_inv;
	T jacobian_y_w = - qy_original * qw_original * rotation_norm_pow_3_inv;
	T jacobian_z_w = - qz_original * qw_original * rotation_norm_pow_3_inv;

	rotation_gradient(3, i) = (dgradient_x * jacobian_x_w + dgradient_y * jacobian_y_w + dgradient_z * jacobian_z_w + dgradient_w * jacobian_w_w);

	// rotation_gradient(0, i) *= rotation_norm_pow_3_inv * ((T)rotation_norm_2 - (T)2.0f * qx_original * qx_original);
	// rotation_gradient(1, i) *= rotation_norm_pow_3_inv * ((T)rotation_norm_2 - (T)2.0f * qy_original * qy_original);
	// rotation_gradient(2, i) *= rotation_norm_pow_3_inv * ((T)rotation_norm_2 - (T)2.0f * qz_original * qz_original);
	// rotation_gradient(3, i) *= rotation_norm_pow_3_inv * ((T)rotation_norm_2 - (T)2.0f * qw_original * qw_original);

	input_gradient(0, i) = (T) (loss(0, i)) * (-(T)2.0f*qy*qy - (T)2.0f*qz*qz + (T)1.0f)         + (T) (loss(1, i)) * ((T)2.0f*qw*qz + (T)2.0f*qx*qy)               + (T) (loss(2, i)) * (-(T)2.0f*qw*qy + (T)2.0f*qx*qz);
	input_gradient(1, i) = (T) (loss(0, i)) * (-(T)2.0f*qw*qz + (T)2.0f*qx*qy)             + (T) (loss(1, i)) * (-(T)2.0f*qx*qx - (T)2.0f*qz*qz + (T)1.0f)          + (T) (loss(2, i)) * ((T)2.0f*qw*qx + (T)2.0f*qy*qz);
	input_gradient(2, i) = (T) (loss(0, i)) * ((T)2.0f*qw*qy + (T)2.0f*qx*qz)              + (T) (loss(1, i)) * (-(T)2.0f*qw*qx + (T)2.0f*qy*qz)              + (T) (loss(2, i)) * (-(T)2.0f*qx*qx - (T)2.0f*qy*qy + (T)1.0f);
}


template <typename T, typename TIn = T>
__global__ void add_loss_to_rotation_6d_each(
	const uint32_t n_elements,
	const tcnn::MatrixView<TIn> loss,
	const tcnn::MatrixView<TIn> input,
	const T* rotation_6d,
	const T* transition,
	const Eigen::Vector3f first_frame_offset,
	tcnn::MatrixView<T> rotation_gradient,
	tcnn::MatrixView<T> transition_gradient,
	tcnn::MatrixView<T> input_gradient
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	Eigen::Matrix3f rotation_matrix;
	rotation_6d_to_matrix<T>(rotation_6d, rotation_matrix);

	Eigen::Vector3f loss_vector(loss(0, i), loss(1, i), loss(2, i));
	Eigen::Vector3f input_vector(input(0, i) + (TIn)transition[0], input(1, i) + (TIn)transition[1], input(2, i) + (TIn)transition[2]);
	Eigen::Vector3f input_gradient_vector = rotation_matrix.inverse() * loss_vector;

	input_gradient(0, i) = input_gradient_vector[0];
	input_gradient(1, i) = input_gradient_vector[1];
	input_gradient(2, i) = input_gradient_vector[2];

	transition_gradient(0, i) = input_gradient_vector[0];
	transition_gradient(1, i) = input_gradient_vector[1];
	transition_gradient(2, i) = input_gradient_vector[2];

	// loss_vector.resize(3, 1);
	// input_vector.resize(1, 3);

	// Eigen::Matrix3f rotation_gradient_matrix =  loss_vector * input_vector ;
	Eigen::Matrix3f rotation_gradient_matrix;
	// Eigen:Matrix<float, 3, 3 ,Eigen::RowMajor> rotation_gradient_matrix;

	rotation_gradient_matrix << loss_vector[0] * input_vector[0], loss_vector[0] * input_vector[1], loss_vector[0] * input_vector[2],
								loss_vector[1] * input_vector[0], loss_vector[1] * input_vector[1], loss_vector[1] * input_vector[2],
								loss_vector[2] * input_vector[0], loss_vector[2] * input_vector[1], loss_vector[2] * input_vector[2];

	// printf("rotation_gradients_matrix:%f, loss_vector:%f\n", rotation_gradient_matrix(1,0), (float)loss_vector[1] * input_vector[0]);

	// rotation_gradient_matrix << loss_vector[0] * input_vector[0], loss_vector[0] * input_vector[1], loss_vector[0] * input_vector[2],
	// 							loss_vector[1] * input_vector[0], loss_vector[1] * input_vector[1], loss_vector[1] * input_vector[2],
	// 							loss_vector[2] * input_vector[0], loss_vector[2] * input_vector[1], loss_vector[2] * input_vector[2];

	T gradients_6d[6];

	gradient_rotation_matrix_to_6d<T>(rotation_6d, rotation_gradient_matrix, gradients_6d);
	rotation_gradient(0, i) = gradients_6d[0];
	rotation_gradient(1, i) = gradients_6d[1];
	rotation_gradient(2, i) = gradients_6d[2];
	rotation_gradient(3, i) = gradients_6d[3];
	rotation_gradient(4, i) = gradients_6d[4];
	rotation_gradient(5, i) = gradients_6d[5];

}

template <typename T, typename TIn = T>
__global__ void add_loss_with_viewdir_to_rotation_6d_each(
	const uint32_t n_elements,
	const tcnn::MatrixView<TIn> loss,
	const tcnn::MatrixView<TIn> input,
	const T* rotation_6d,
	const T* transition,
	const Eigen::Vector3f first_frame_offset,
	tcnn::MatrixView<T> rotation_gradient,
	tcnn::MatrixView<T> transition_gradient,
	tcnn::MatrixView<T> input_gradient
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	Eigen::Matrix3f rotation_matrix;
	rotation_6d_to_matrix<T>(rotation_6d, rotation_matrix);

	Eigen::Vector3f loss_vector(loss(0, i), loss(1, i), loss(2, i));
	Eigen::Vector3f viewdir_loss_vector(loss(3, i), loss(4, i), loss(5, i));

	Eigen::Vector3f viewdir_vector(input(3, i), input(4, i), input(5, i));
	Eigen::Vector3f input_vector(input(0, i) + (TIn)transition[0], input(1, i) + (TIn)transition[1], input(2, i) + (TIn)transition[2]);
	Eigen::Vector3f input_gradient_vector = rotation_matrix.inverse() * loss_vector;

	input_gradient(0, i) = input_gradient_vector[0];
	input_gradient(1, i) = input_gradient_vector[1];
	input_gradient(2, i) = input_gradient_vector[2];

	transition_gradient(0, i) = input_gradient_vector[0];
	transition_gradient(1, i) = input_gradient_vector[1];
	transition_gradient(2, i) = input_gradient_vector[2];

	// loss_vector.resize(3, 1);
	// input_vector.resize(1, 3);

	// Eigen::Matrix3f rotation_gradient_matrix =  loss_vector * input_vector ;
	Eigen::Matrix3f rotation_gradient_matrix;
	// Eigen:Matrix<float, 3, 3 ,Eigen::RowMajor> rotation_gradient_matrix;

	// rotation_gradient_matrix << loss_vector[0] * input_vector[0], loss_vector[0] * input_vector[1], loss_vector[0] * input_vector[2],
	// 							loss_vector[1] * input_vector[0], loss_vector[1] * input_vector[1], loss_vector[1] * input_vector[2],
	// 							loss_vector[2] * input_vector[0], loss_vector[2] * input_vector[1], loss_vector[2] * input_vector[2];

	rotation_gradient_matrix << loss_vector[0] * input_vector[0] + viewdir_loss_vector[0] * viewdir_vector[0], loss_vector[0] * input_vector[1] + viewdir_loss_vector[0] * viewdir_vector[1], loss_vector[0] * input_vector[2] + viewdir_loss_vector[0] * viewdir_vector[2],
								loss_vector[1] * input_vector[0] + viewdir_loss_vector[1] * viewdir_vector[0], loss_vector[1] * input_vector[1] + viewdir_loss_vector[1] * viewdir_vector[1], loss_vector[1] * input_vector[2] + viewdir_loss_vector[1] * viewdir_vector[2],
								loss_vector[2] * input_vector[0] + viewdir_loss_vector[2] * viewdir_vector[0], loss_vector[2] * input_vector[1] + viewdir_loss_vector[2] * viewdir_vector[1], loss_vector[2] * input_vector[2] + viewdir_loss_vector[2] * viewdir_vector[2];

	// printf("rotation_gradients_matrix:%f, loss_vector:%f\n", rotation_gradient_matrix(1,0), (float)loss_vector[1] * input_vector[0]);

	// rotation_gradient_matrix << loss_vector[0] * input_vector[0], loss_vector[0] * input_vector[1], loss_vector[0] * input_vector[2],
	// 							loss_vector[1] * input_vector[0], loss_vector[1] * input_vector[1], loss_vector[1] * input_vector[2],
	// 							loss_vector[2] * input_vector[0], loss_vector[2] * input_vector[1], loss_vector[2] * input_vector[2];

	T gradients_6d[6];

	gradient_rotation_matrix_to_6d<T>(rotation_6d, rotation_gradient_matrix, gradients_6d);
	rotation_gradient(0, i) = gradients_6d[0];
	rotation_gradient(1, i) = gradients_6d[1];
	rotation_gradient(2, i) = gradients_6d[2];
	rotation_gradient(3, i) = gradients_6d[3];
	rotation_gradient(4, i) = gradients_6d[4];
	rotation_gradient(5, i) = gradients_6d[5];

}



template <typename T, typename TIn = T>
__global__ void add_transition_view_to_gradient(
	const uint32_t n_elements,
	const tcnn::MatrixView<TIn> loss,
	T* transition_gradient
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i / 3;
	const uint32_t dim_idx = i - elem_idx * 3;

	// mean_value(0,i) += rgbd(7, i);
	atomicAdd(&transition_gradient[dim_idx], (T)(loss(dim_idx, elem_idx)*1.0f));
}

template <typename T>
__global__ void variance_loss_to_gradients(
	const uint32_t n_elements,
	const tcnn::MatrixView<T> rgbd,
	T* mean_value
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	// mean_value[0] += rgbd(7, i);
	atomicAdd(mean_value,rgbd(7, i));
}

template <typename T>
__global__ void extract_sdf_value_view(
	const uint32_t n_elements,
	const tcnn::MatrixView<T> sdf_value,
	tcnn::MatrixView<T> rgbd
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	rgbd(0, i) = sdf_value(0, i);
}

template <typename T>
__global__ void extract_sdf_value_view_with_bias(
	const uint32_t n_elements,
	const tcnn::MatrixView<T> sdf_value,
	T bias,
	tcnn::MatrixView<T> rgbd
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	rgbd(0, i) = sdf_value(0, i) + bias;
}

template <typename T>
__global__ void extract_density(
	const uint32_t n_elements,
	const uint32_t density_stride,
	const uint32_t rgbd_stride,
	const T* __restrict__ density,
	T* __restrict__ rgbd
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	rgbd[i * rgbd_stride] = density[i * density_stride];
}

template <typename T>
__global__ void debug_log(
	const uint32_t  n_elements,
	tcnn::MatrixView<T> output
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	printf("output:%f,%f,%f\n", (float)output(0, i), (float)output(1, i), (float)output(2, i));
}

template <typename T>
__global__ void debug_log_int(
	const uint32_t  n_elements,
	tcnn::MatrixView<T> output
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	
	printf("%d_%d: %d\n",0, i, output(i,0));
	printf("%d_%d: %d\n",1, i, output(i,1));
	printf("%d_%d: %d\n",10, i, output(i,10));
	printf("%d_%d: %d\n",100, i, output(i,100));
	printf("%d_%d: %d\n",1000, i, output(i,1000));
}

template <typename T>
__global__ void extract_rgb(
	const uint32_t n_elements,
	const uint32_t rgb_stride,
	const uint32_t output_stride,
	const T* __restrict__ rgbd,
	T* __restrict__ rgb
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i / 3;
	const uint32_t dim_idx = i - elem_idx * 3;

	rgb[elem_idx*rgb_stride + dim_idx] = rgbd[elem_idx*output_stride + dim_idx];
}

template <typename T>
__global__ void add_density_gradient(
	const uint32_t n_elements,
	const uint32_t rgbd_stride,
	const T* __restrict__ rgbd,
	const uint32_t density_stride,
	T* __restrict__ density
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	density[i * density_stride] += rgbd[i * rgbd_stride + 3];
}

template <typename T>
__global__ void add_sdf_gradient(
	const uint32_t n_elements,
	const uint32_t rgbd_stride,
	const T* __restrict__ rgbd,
	const uint32_t density_stride,
	T* __restrict__ density
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	density[i * density_stride] += rgbd[i * rgbd_stride + 3];
}