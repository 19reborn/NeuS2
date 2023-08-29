/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   nerbf_network.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  A network that first processes 3D position to density and
 *          subsequently direction to color.
 */

#pragma once

#include <tiny-cuda-nn/common.h>

#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/gpu_memory_json.h>
#include <tiny-cuda-nn/multi_stream.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/reduce_sum.h>

#include <neural-graphics-primitives/trainable_buffer.cuh>
#include <neural-graphics-primitives/common_operation.cuh>

#include <neural-graphics-primitives/transform_network.h>

#include <json/json.hpp>

NGP_NAMESPACE_BEGIN

using namespace Eigen;

template <typename T>
class NgpNetwork : public tcnn::Network<float, T> {
public:
	using json = nlohmann::json;

	NgpNetwork(uint32_t n_pos_dims, uint32_t n_dir_dims, uint32_t n_extra_dims, uint32_t dir_offset, const json& pos_encoding, const json& dir_encoding, const json& density_network, const json& rgb_network) : m_n_pos_dims{n_pos_dims}, m_n_dir_dims{n_dir_dims}, m_dir_offset{dir_offset}, m_n_extra_dims{n_extra_dims} {
		m_pos_encoding.reset(tcnn::create_encoding<T>(n_pos_dims, pos_encoding, density_network.contains("otype") && (tcnn::equals_case_insensitive(density_network["otype"], "FullyFusedMLP") || tcnn::equals_case_insensitive(density_network["otype"], "MegakernelMLP")) ? 16u : 8u));
		uint32_t rgb_alignment = tcnn::minimum_alignment(rgb_network);
		m_dir_encoding.reset(tcnn::create_encoding<T>(m_n_dir_dims + m_n_extra_dims, dir_encoding, rgb_alignment));

		json local_density_network_config = density_network;
		local_density_network_config["n_input_dims"] = m_pos_encoding->padded_output_width();
		if (!density_network.contains("n_output_dims")) {
			local_density_network_config["n_output_dims"] = 16;
		}
		m_density_network.reset(tcnn::create_network<T>(local_density_network_config));

		m_rgb_network_input_width = tcnn::next_multiple(m_dir_encoding->padded_output_width() + m_density_network->padded_output_width(), rgb_alignment);

		json local_rgb_network_config = rgb_network;
		local_rgb_network_config["n_input_dims"] = m_rgb_network_input_width;
		local_rgb_network_config["n_output_dims"] = 3;
		m_rgb_network.reset(tcnn::create_network<T>(local_rgb_network_config));


		// m_delta_network = new DeltaNetwork<T>{};
		// m_delta_network = std::make_shared<DeltaNetwork<T>>();
		m_delta_network = std::make_shared<DeltaNetwork<T>>(m_pos_encoding->input_width() + m_dir_encoding->input_width());
		// std::make_shared<DeltaNetwork<T>>();

		#if VARIANCE_MLP
			json m_variance_network_config = {
				{"otype", "CutLassMLP"},
				{"activation", "None"},
				{"output_activation", "None"},
				{"n_neurons", 16},
				{"n_hidden_layers", 0},
				{"n_input_dims", 16},
				{"n_output_dims", 16}
			};

			m_variance_network.reset(tcnn::create_network<T>(m_variance_network_config));
		#else
			m_variance_network = std::make_shared<TrainableBuffer<1, 1, T>>(Eigen::Matrix<int, 1, 1>{(int)4});
		#endif


		// pcg32 rnd{1337};
		// float init_variance = 0.3;
		// trainable_variance->initialize_params(rnd, (float*)&init_variance, (float*)&init_variance, (float*)&init_variance, (float*)&init_variance, (float*)&init_variance);
		
		// std::shared_ptr<tcnn::Optimizer<float>> variance_optimizer.reset(create_optimizer<float>({
		// 	{"otype", "Adam"},
		// 	{"learning_rate", 1e-2},
		// 	{"beta1", 0.9f},
		// 	{"beta2", 0.99f},
		// }));

		// variance_optimizer->allocate(trainable_variance);

		m_variance = 0.3f;
		m_training_step = 0;
		m_sdf_bias =(T)-0.1f; // oleks 
		// m_sdf_bias =(T)-0.15f; // FranziRed
		// m_sdf_bias =(T)-1.0f;
		// m_sdf_bias =(T)-0.5f;
		// m_sdf_bias =(T)-0.2f;


		// load smpl skinning weights

		// tcnn::GPUMatrixDynamic<float> smpl_skinning_weights{ 24, 6890, stream };
		// m_smpl_skinning_weights = tcnn::GPUMatrixDynamic<float>{ 24, 6890};
		// auto skinning_weights_cpu = load_skinning_weights(); //std::vector<std::vector<float>>
		// // printf("skinning_weights_size:%d,%d\n",skinning_weights_cpu.size(),skinning_weights_cpu[0].size()); // 6890 x 24
		// // CUDA_CHECK_THROW(cudaMemcpyAsync(smpl_skinning_weights.data(), m_smpl_skinning_weights.data(), 6890 * 24 * sizeof(float), cudaMemcpyHostToDevice, stream));
		// // CUDA_CHECK_THROW(cudaMemcpy(m_smpl_skinning_weights.data(), &skinning_weights_cpu[0][0], 6890 * 24 * sizeof(float), cudaMemcpyHostToDevice));
		// CUDA_CHECK_THROW(cudaMemcpy(m_smpl_skinning_weights.data(), skinning_weights_cpu.data(), 6890 * 24 * sizeof(float), cudaMemcpyHostToDevice));
		// // checked


		// // load smpl transform

		// // tcnn::GPUMatrixDynamic<float> joints_RT{ 16, 24, stream };		
		// m_joints_RT = tcnn::GPUMatrixDynamic<float> { 16, 24};		

		// // std::string smpl_transform_dir = "data/test_smpl.json";
		// std::string smpl_transform_dir = "/home/wangyiming/instant-ngp-neus/data/test_smpl.json";
		// // std::ifstream f(smpl_transform_dir);
		// std::ifstream f{smpl_transform_dir};
		// if (!f.is_open()){
		// 	printf("Error opening file\n"); 
		// 	exit (1); 
		// }

		// nlohmann::json smpl_transforms = nlohmann::json::parse(f, nullptr, true, true);
		// std::vector<std::vector<float>> joins_RT_cpu = smpl_transforms["joints_RT"]; // 24 x 16
		// uint32_t joints_RT_size = joins_RT_cpu.size() * joins_RT_cpu[0].size();
		// std::vector<float> joins_RT_cpu_one_dimension(joints_RT_size); // 24 x 16
		// for(uint32_t i = 0; i < joins_RT_cpu.size(); i++){
		// 	memcpy(joins_RT_cpu_one_dimension.data() + joins_RT_cpu[0].size() * i, joins_RT_cpu[i].data(), joins_RT_cpu[0].size() * sizeof(float));
		// }

		// CUDA_CHECK_THROW(cudaMemcpy(m_joints_RT.data(), joins_RT_cpu_one_dimension.data(), joints_RT_size * sizeof(float), cudaMemcpyHostToDevice));

		// // load smpl_mesh

		// // std::string mesh_path = "/home/wangyiming/instant-ngp-neus/data/na_crop_with_mask/olkes/000000.obj";
		// // std::string mesh_path = "/home/wangyiming/DataSet/oleks_data/smpl/000150.obj";
		// std::string mesh_path = "data/test_smpl.obj";

		
		// TriangleMesh mesh;
		// loadObj(mesh_path,mesh);

		// m_mesh_face_num = mesh.faces.size();
		// m_mesh_vert_num = mesh.verts.size();

		// // tcnn::GPUMatrixDynamic<uint32_t> m_mesh_triangle_idx{ 3u, mesh_face_num, stream };
		// m_mesh_triangle_idx = tcnn::GPUMatrixDynamic<uint32_t> { 3u, m_mesh_face_num};
		// // CUDA_CHECK_THROW(cudaMemcpyAsync(m_mesh_triangle_idx.data(), mesh.faces.data(),  mesh_face_num * 3 * sizeof(uint32_t), cudaMemcpyHostToDevice, stream)); // to do:: need check
		// CUDA_CHECK_THROW(cudaMemcpy(m_mesh_triangle_idx.data(), mesh.faces.data(),  m_mesh_face_num * 3 * sizeof(uint32_t), cudaMemcpyHostToDevice)); // to do:: need check

		// // tcnn::GPUMatrixDynamic<float> m_mesh_triangle{ 9u, mesh_face_num, stream };
		// m_mesh_triangle = tcnn::GPUMatrixDynamic<float> { 9u, m_mesh_face_num};
		// // m_mesh_triangle.set_size_unsafe(9u, m_mesh_face_num);
		// // change mesh into cuda triangle T x 3 x 3
		// auto triangle_cpu = get_triangle(mesh); // std::vector<std::vector<Eigen::Vector3f>>
		// // printf("triangle_cpu:%f,%f,%f,%f,%f,%f,%f,%f\n",triangle_cpu[0][0])
		// // printf("triangle_cpu:%f,%f,%f\n",triangle_cpu[0][0][0],triangle_cpu[0][0][1],triangle_cpu[0][0][2]);
		// // printf("triangle_cpu:%f,%f,%f\n",triangle_cpu[0][1][0],triangle_cpu[0][1][1],triangle_cpu[0][1][2]);
		// // printf("triangle_cpu:%f,%f,%f\n",triangle_cpu[0][2][0],triangle_cpu[0][2][1],triangle_cpu[0][2][2]);
		// // copy into cuda
		// // CUDA_CHECK_THROW(cudaMemcpyAsync(m_mesh_triangle.data(), triangle_cpu.data(),  mesh_face_num * 9 * sizeof(float), cudaMemcpyHostToDevice, stream));
		// // m_mesh_vert_num(cudaMemcpyAsync(m_mesh_triangle.data(), triangle_cpu.data(),  mesh_face_num * 9 * sizeof(float), cudaMemcpyHostToDevice, stream));
		// CUDA_CHECK_THROW(cudaMemcpy(m_mesh_triangle.data(), triangle_cpu.data(),  m_mesh_face_num * 9 * sizeof(float), cudaMemcpyHostToDevice));

		// m_mesh_vertices = tcnn::GPUMatrixDynamic<float> { 3u, m_mesh_vert_num};

		// // CUDA_CHECK_THROW(cudaMemcpyAsync(m_mesh_triangle.data(), triangle_cpu.data(),  mesh_face_num * 9 * sizeof(float), cudaMemcpyHostToDevice, stream));
		// CUDA_CHECK_THROW(cudaMemcpy(m_mesh_vertices.data(), mesh.verts.data(),  m_mesh_vert_num * 3 * sizeof(float), cudaMemcpyHostToDevice));


		accumulated_transition = std::make_shared<TrainableBuffer<1, 1, T>>(Eigen::Matrix<int, 1, 1>{(int)4});
		#if rotation_reprensentation
			accumulated_rotation = std::make_shared<TrainableBuffer<1, 1, T>>(Eigen::Matrix<int, 1, 1>{(int)12});
		#else
			accumulated_rotation = std::make_shared<TrainableBuffer<1, 1, T>>(Eigen::Matrix<int, 1, 1>{(int)4});
		#endif
		init_accumulation_movement();
	}

	virtual ~NgpNetwork() { }

	// void cal_dy(
	// 	cudaStream_t stream,
	// 	const tcnn::Context& ctx,
	// 	const tcnn::GPUMatrixDynamic<float>& loss,
	// 	tcnn::GPUMatrixDynamic<T>* dy,
	// 	bool use_inference_params = false,
	// 	tcnn::EGradientMode param_gradients_mode = tcnn::EGradientMode::Overwrite
	// ) override {
	// 	printf("cal_dy override succeed!\n");
	// 	return;
	// 	// kernel_cal_dy()
	// }

	void inference_mixed_precision_impl(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>& output, bool use_inference_params = true) override {
		uint32_t batch_size = input.n();
		tcnn::GPUMatrixDynamic<T> density_network_input{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		tcnn::GPUMatrixDynamic<T> rgb_network_input{m_rgb_network_input_width, batch_size, stream, m_dir_encoding->preferred_output_layout()};

		tcnn::GPUMatrixDynamic<T> density_network_output = rgb_network_input.slice_rows(0, m_density_network->padded_output_width());
		tcnn::GPUMatrixDynamic<T> rgb_network_output{output.data(), m_rgb_network->padded_output_width(), batch_size, output.layout()};

		m_pos_encoding->inference_mixed_precision(
			stream,
			input.slice_rows(0, m_pos_encoding->input_width()),
			density_network_input,
			use_inference_params
		);

		m_density_network->inference_mixed_precision(stream, density_network_input, density_network_output, use_inference_params);

		auto dir_out = rgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width());
		m_dir_encoding->inference_mixed_precision(
			stream,
			input.slice_rows(m_dir_offset, m_dir_encoding->input_width()),
			dir_out,
			use_inference_params
		);

		m_rgb_network->inference_mixed_precision(stream, rgb_network_input, rgb_network_output, use_inference_params);

		tcnn::linear_kernel(extract_density<T>, 0, stream,
			batch_size,
			density_network_output.layout() == tcnn::AoS ? density_network_output.stride() : 1,
			output.layout() == tcnn::AoS ? padded_output_width() : 1,
			density_network_output.data(),
			output.data() + 3 * (output.layout() == tcnn::AoS ? 1 : batch_size)
		);
	}

	uint32_t padded_density_output_width() const {
		return m_density_network->padded_output_width();
	}

	std::unique_ptr<tcnn::Context> forward_impl(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>* output = nullptr, bool use_inference_params = false, bool prepare_input_gradients = false) override {
		// Make sure our temporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();

		auto forward = std::make_unique<ForwardContext>();

		forward->density_network_input = tcnn::GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		forward->rgb_network_input = tcnn::GPUMatrixDynamic<T>{m_rgb_network_input_width, batch_size, stream, m_dir_encoding->preferred_output_layout()};

		// printf("density_network_input layout:%d\n",forward->density_network_input.layout() == tcnn::AoS );
		// printf("rgb_network_input layout:%d\n",forward->rgb_network_input.layout() == tcnn::AoS );
		// printf("input layout:%d\n",input.layout() == tcnn::AoS );

		forward->pos_encoding_ctx = m_pos_encoding->forward(
			stream,
			input.slice_rows(0, m_pos_encoding->input_width()),
			&forward->density_network_input,
			use_inference_params,
			prepare_input_gradients
		);

		forward->density_network_output = forward->rgb_network_input.slice_rows(0, m_density_network->padded_output_width());

		// printf("density output layout:%d\n",forward->density_network_output.layout() == tcnn::AoS);
		forward->density_network_ctx = m_density_network->forward(stream, forward->density_network_input, &forward->density_network_output, use_inference_params, prepare_input_gradients);

		auto dir_out = forward->rgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width());
		// printf("forward->rgb_network_input:%d\n",forward->rgb_network_input.layout());
		// printf("input:%d\n",input.layout());
		forward->dir_encoding_ctx = m_dir_encoding->forward(
			stream,
			input.slice_rows(m_dir_offset, m_dir_encoding->input_width()),
			&dir_out,
			use_inference_params,
			prepare_input_gradients
		);

		// printf("dir_out layout:%d\n",dir_out.layout() == tcnn::AoS );

		if (output) {
			forward->rgb_network_output = tcnn::GPUMatrixDynamic<T>{output->data(), m_rgb_network->padded_output_width(), batch_size, output->layout()};
		}
		// printf("m_rgb_network->padded_output_width():%d\n",m_rgb_network->padded_output_width());
		// printf("output width:%d\n",output->m());
		// printf("forward->rgb_network_input layout:%d\n",forward->rgb_network_input.layout() == tcnn::AoS );
		forward->rgb_network_ctx = m_rgb_network->forward(stream, forward->rgb_network_input, output ? &forward->rgb_network_output : nullptr, use_inference_params, prepare_input_gradients);
		// printf("forward->rgb_network_output layout:%d\n",forward->rgb_network_output.layout() == tcnn::AoS );
		// printf("rgb_output\n");
		// tcnn::linear_kernel(debug_log<T>, 0, stream, output->m(), output->view());

		if (output) {
			tcnn::linear_kernel(extract_density<T>, 0, stream,
				batch_size, m_dir_encoding->preferred_output_layout() == tcnn::AoS ? forward->density_network_output.stride() : 1, padded_output_width(), forward->density_network_output.data(), output->data()+3
			);
		}
		// printf("after density_output\n");
		// tcnn::linear_kernel(debug_log<T>, 0, stream, output->m(), output->view());

		return forward;
	}

	void backward_impl(
		cudaStream_t stream,
		const tcnn::Context& ctx,
		const tcnn::GPUMatrixDynamic<float>& input,
		const tcnn::GPUMatrixDynamic<T>& output,
		const tcnn::GPUMatrixDynamic<T>& dL_doutput,
		tcnn::GPUMatrixDynamic<float>* dL_dinput = nullptr,
		bool use_inference_params = false,
		tcnn::EGradientMode param_gradients_mode = tcnn::EGradientMode::Overwrite
	) override {
		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);

		// Make sure our teporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();

		tcnn::GPUMatrix<T> dL_drgb{m_rgb_network->padded_output_width(), batch_size, stream};
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_drgb.data(), 0, dL_drgb.n_bytes(), stream));
		tcnn::linear_kernel(extract_rgb<T>, 0, stream,
			batch_size*3, dL_drgb.m(), dL_doutput.m(), dL_doutput.data(), dL_drgb.data()
		);

		const tcnn::GPUMatrixDynamic<T> rgb_network_output{(T*)output.data(), m_rgb_network->padded_output_width(), batch_size, output.layout()};
		tcnn::GPUMatrixDynamic<T> dL_drgb_network_input{m_rgb_network_input_width, batch_size, stream, m_dir_encoding->preferred_output_layout()};
		m_rgb_network->backward(stream, *forward.rgb_network_ctx, forward.rgb_network_input, rgb_network_output, dL_drgb, &dL_drgb_network_input, use_inference_params, param_gradients_mode);

		// printf("rgb_network output, input layout, dl_drgboutput, dl_drgbinput:%d,%d,%d,%d\n",rgb_network_output.layout(),forward.rgb_network_input.layout(),dL_drgb.laytout(),dL_drgb_network_input.layout());
		// printf("rgb_network output, input layout, dl_drgboutput, dl_drgbinput:%d,%d,%d,%d\n",rgb_network_output.layout(),forward.rgb_network_input.layout(),dL_drgb.layout(),dL_drgb_network_input.layout());
		// Backprop through dir encoding if it is trainable or if we need input gradients
		if (m_dir_encoding->n_params() > 0 || dL_dinput) {
			tcnn::GPUMatrixDynamic<T> dL_ddir_encoding_output = dL_drgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width());
			tcnn::GPUMatrixDynamic<float> dL_ddir_encoding_input;
			if (dL_dinput) {
				dL_ddir_encoding_input = dL_dinput->slice_rows(m_dir_offset, m_dir_encoding->input_width());
			}

			m_dir_encoding->backward(
				stream,
				*forward.dir_encoding_ctx,
				input.slice_rows(m_dir_offset, m_dir_encoding->input_width()),
				forward.rgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width()),
				dL_ddir_encoding_output,
				dL_dinput ? &dL_ddir_encoding_input : nullptr,
				use_inference_params,
				param_gradients_mode
			);
		}

		tcnn::GPUMatrixDynamic<T> dL_ddensity_network_output = dL_drgb_network_input.slice_rows(0, m_density_network->padded_output_width());
		tcnn::linear_kernel(add_density_gradient<T>, 0, stream,
			batch_size,
			dL_doutput.m(),
			dL_doutput.data(),
			dL_ddensity_network_output.layout() == tcnn::RM ? 1 : dL_ddensity_network_output.stride(),
			dL_ddensity_network_output.data()
		);

		tcnn::GPUMatrixDynamic<T> dL_ddensity_network_input;
		if (m_pos_encoding->n_params() > 0 || dL_dinput) {
			dL_ddensity_network_input = tcnn::GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		}

		m_density_network->backward(stream, *forward.density_network_ctx, forward.density_network_input, forward.density_network_output, dL_ddensity_network_output, dL_ddensity_network_input.data() ? &dL_ddensity_network_input : nullptr, use_inference_params, param_gradients_mode);


		// Backprop through pos encoding if it is trainable or if we need input gradients
		if (dL_ddensity_network_input.data()) {
			tcnn::GPUMatrixDynamic<float> dL_dpos_encoding_input;
			if (dL_dinput) {
				dL_dpos_encoding_input = dL_dinput->slice_rows(0, m_pos_encoding->input_width());
			}

			m_pos_encoding->backward(
				stream,
				*forward.pos_encoding_ctx,
				input.slice_rows(0, m_pos_encoding->input_width()),
				forward.density_network_input,
				dL_ddensity_network_input,
				dL_dinput ? &dL_dpos_encoding_input : nullptr,
				use_inference_params,
				param_gradients_mode
			);
		}
	}

	void density(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>& output, bool use_inference_params = true) {
		if (input.layout() != tcnn::CM) {
			throw std::runtime_error("NgpNetwork::density input must be in column major format.");
		}

		uint32_t batch_size = output.n();
		tcnn::GPUMatrixDynamic<T> density_network_input{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};

		m_pos_encoding->inference_mixed_precision(
			stream,
			input.slice_rows(0, m_pos_encoding->input_width()),
			density_network_input,
			use_inference_params
		);

		m_density_network->inference_mixed_precision(stream, density_network_input, output, use_inference_params);
	}

	void sdf(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>& output, bool use_inference_params = true) {
		if (input.layout() != tcnn::CM) {
			throw std::runtime_error("NgpNetwork::density input must be in column major format.");
		}

		uint32_t batch_size = output.n();
		
		density(stream,input,output,use_inference_params);
	}



	std::unique_ptr<tcnn::Context> density_forward(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>* output = nullptr, bool use_inference_params = false, bool prepare_input_gradients = false) {
		if (input.layout() != tcnn::CM) {
			throw std::runtime_error("NgpNetwork::density_forward input must be in column major format.");
		}

		// Make sure our temporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();

		auto forward = std::make_unique<ForwardContext>();

		forward->density_network_input = tcnn::GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};

		forward->pos_encoding_ctx = m_pos_encoding->forward(
			stream,
			input.slice_rows(0, m_pos_encoding->input_width()),
			&forward->density_network_input,
			use_inference_params,
			prepare_input_gradients
		);

		if (output) {
			forward->density_network_output = tcnn::GPUMatrixDynamic<T>{output->data(), m_density_network->padded_output_width(), batch_size, output->layout()};
		}

		forward->density_network_ctx = m_density_network->forward(stream, forward->density_network_input, output ? &forward->density_network_output : nullptr, use_inference_params, prepare_input_gradients);

		return forward;
	}

	void density_backward(
		cudaStream_t stream,
		const tcnn::Context& ctx,
		const tcnn::GPUMatrixDynamic<float>& input,
		const tcnn::GPUMatrixDynamic<T>& output,
		const tcnn::GPUMatrixDynamic<T>& dL_doutput,
		tcnn::GPUMatrixDynamic<float>* dL_dinput = nullptr,
		bool use_inference_params = false,
		tcnn::EGradientMode param_gradients_mode = tcnn::EGradientMode::Overwrite
	) {
		if (input.layout() != tcnn::CM || (dL_dinput && dL_dinput->layout() != tcnn::CM)) {
			throw std::runtime_error("NgpNetwork::density_backward input must be in column major format.");
		}

		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);

		// Make sure our temporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();

		tcnn::GPUMatrixDynamic<T> dL_ddensity_network_input;
		if (m_pos_encoding->n_params() > 0 || dL_dinput) {
			dL_ddensity_network_input = tcnn::GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		}

		m_density_network->backward(stream, *forward.density_network_ctx, forward.density_network_input, output, dL_doutput, dL_ddensity_network_input.data() ? &dL_ddensity_network_input : nullptr, use_inference_params, param_gradients_mode);

		// Backprop through pos encoding if it is trainable or if we need input gradients
		if (dL_ddensity_network_input.data()) {
			tcnn::GPUMatrixDynamic<float> dL_dpos_encoding_input;
			if (dL_dinput) {
				dL_dpos_encoding_input = dL_dinput->slice_rows(0, m_pos_encoding->input_width());
			}

			m_pos_encoding->backward(
				stream,
				*forward.pos_encoding_ctx,
				input.slice_rows(0, m_pos_encoding->input_width()),
				forward.density_network_input,
				dL_ddensity_network_input,
				dL_dinput ? &dL_dpos_encoding_input : nullptr,
				use_inference_params,
				param_gradients_mode
			);
		}
	}

	void set_params(T* params, T* inference_params, T* backward_params, T* gradients) override {
		size_t offset = 0;
		m_density_network->set_params(
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset
		);
		offset += m_density_network->n_params();

		m_rgb_network->set_params(
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset
		);
		offset += m_rgb_network->n_params();

		m_pos_encoding->set_params(
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset
		);
		offset += m_pos_encoding->n_params();

		m_dir_encoding->set_params(
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset
		);
		offset += m_dir_encoding->n_params();
	}

	// std::vector<float> load_sdf_mlp_weight(uint32_t n_elements, float* params){
	std::vector<float> load_sdf_mlp_weight(uint32_t n_elements){
		
		std::vector<float> data(n_elements);
		std::FILE* fp;
		// std::FILE* fp = fopen("utils/mlp_weights_all/mlp_weights_hidden_layer_num_2.txt", "r");
		if (m_density_network_input_width == 32){
			fp = fopen("utils/mlp_weights_hidden_layer_num_1_hidden_size_32.txt", "r");
		}
		else if (m_density_network_input_width == 48) {
			fp = fopen("utils/mlp_weights.txt", "r");
		}
		else {
			printf("only support input of 32 or 48\n");
			exit(1);
		}
		// std::ifstream in("utils/mlp_weights.txt");

		printf("network_params_elements:%d\n",n_elements);
		if (!fp)
			printf("no file");
		else {
			uint32_t i;
			for (i = 0; i < n_elements; i++) {
				float input_data;
				fscanf(fp, "%f", &data[i]);
				// fscanf(fp, "%f", &input_data);

				// printf("load_data:%f\n",input_data);
			}
			fclose(fp);
		}
		return data;
	}

	// std::vector<std::vector<float>> load_skinning_weights(uint32_t verts_num = 6890, uint32_t joints_num = 24){
		
	// 	std::vector<std::vector<float>> data;
	// 	std::FILE* fp = fopen("utils/skinning_weights.txt", "r");
	// 	// std::ifstream in("utils/mlp_weights.txt");

	// 	if (!fp)
	// 		printf("no file");
	// 	else {
	// 		uint32_t i;
	// 		for (i = 0; i < verts_num; i++) {
	// 			std::vector<float> joints_data;
	// 			for (uint32_t j =0 ;j<joints_num; j++){
	// 				float input_data;
	// 				fscanf(fp, "%f", &input_data);
	// 				joints_data.push_back(input_data);
	// 			}
	// 			data.push_back(joints_data);
	// 		}
	// 		fclose(fp);
	// 	}
	// 	return data;
	// }
	std::vector<float> load_skinning_weights(uint32_t verts_num = 6890, uint32_t joints_num = 24){
		
		std::vector<float> data;
		std::FILE* fp = fopen("utils/skinning_weights.txt", "r");
		// std::FILE* fp = fopen("data/test_skinning_weights.txt", "r");
		// std::ifstream in("utils/mlp_weights.txt");

		if (!fp)
			printf("no file");
		else {
			uint32_t i;
			for (i = 0; i < verts_num; i++) {
				for (uint32_t j =0 ;j<joints_num; j++){
					float input_data;
					fscanf(fp, "%f", &input_data);
					data.push_back(input_data);
				}
			}
			fclose(fp);
		}
		return data;
	}

	void initialize_params(tcnn::pcg32& rnd, float* params_full_precision, T* params, T* inference_params, T* backward_params, T* gradients, float scale = 1) override {
		size_t offset = 0;
		m_density_network->initialize_params(
			rnd,
			params_full_precision + offset,
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset,
			scale
		);
		offset += m_density_network->n_params();

		m_rgb_network->initialize_params(
			rnd,
			params_full_precision + offset,
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset,
			scale
		);
		offset += m_rgb_network->n_params();

		m_pos_encoding->initialize_params(
			rnd,
			params_full_precision + offset,
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset,
			scale
		);
		offset += m_pos_encoding->n_params();

		m_dir_encoding->initialize_params(
			rnd,
			params_full_precision + offset,
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset,
			scale
		);
		offset += m_dir_encoding->n_params();
	}

	void initialize_sdf_mlp_params(tcnn::pcg32& rnd, float* params_full_precision, T* params, T* inference_params, T* backward_params, T* gradients, float scale = 1) {
		size_t offset = 0;
		printf("initialize model!\n");

		// printf("scale:%f\n",scale); scale=1.0f
		m_density_network->initialize_params(
			rnd,
			params_full_precision + offset,
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset,
			scale
		);

		// geometry initialization
		#if GEOMETRY_INIT
			std::vector<float> sdf_mlp_weight = load_sdf_mlp_weight(m_density_network->n_params());
			CUDA_CHECK_THROW(cudaMemcpy(params_full_precision + offset, sdf_mlp_weight.data(), m_density_network->n_params() * sizeof(float), cudaMemcpyHostToDevice));
			// CUDA_CHECK_THROW(cudaMemcpyAsync(params_full_precision + offset, sdf_mlp_weight.data(), m_density_network->n_params() * sizeof(float), cudaMemcpyHostToDevice));
		#endif

		offset += m_density_network->n_params();		
	}


	size_t n_params() const override {
		return m_pos_encoding->n_params() + m_density_network->n_params() + m_dir_encoding->n_params() + m_rgb_network->n_params();
	}

	size_t n_params_canonical() const override{
		return m_pos_encoding->n_params() + m_density_network->n_params() + m_dir_encoding->n_params() + m_rgb_network->n_params();
	}

	size_t n_params_delta() const override{
		// return 8;
		return 0;
		// return m_delta_network->n_params();
	}

	tcnn::json n_params_components() const override{
		// make ensure the order is the same as the initilize params.
		return {
			{0, {"density_network", m_density_network->n_params()}},
			{1, {"rgb_network", m_rgb_network->n_params()}},
			{2, {"variance_network", m_variance_network->n_params()}},
			{3, {"pos_encoding", m_pos_encoding->n_params()}},
			{4, {"dir_encoding", m_dir_encoding->n_params()}},
			// {5, {"delta_network", m_delta_network->n_params()}},
		};
	}

	uint32_t padded_output_width() const override {
		return std::max(m_rgb_network->padded_output_width(), (uint32_t)4);
	}

	uint32_t input_width() const override {
		return m_dir_offset + m_n_dir_dims + m_n_extra_dims;
	}

	uint32_t dir_offset() const {
		return m_dir_offset;
	}

	uint32_t output_width() const override {
		return 4;
	}


	uint32_t n_extra_dims() const {
		return m_n_extra_dims;
	}

	uint32_t required_input_alignment() const override {
		return 1; // No alignment required due to encoding
	}

	std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override {
		auto layers = m_density_network->layer_sizes();
		auto rgb_layers = m_rgb_network->layer_sizes();
		layers.insert(layers.end(), rgb_layers.begin(), rgb_layers.end());
		// auto delta_layers = m_delta_network->layer_sizes();
		// layers.insert(layers.end(), delta_layers.begin(), delta_layers.end());
		return layers;
	}

	std::vector<std::pair<uint32_t, uint32_t>> layer_sizes_canonical() const override {
		auto layers = m_density_network->layer_sizes();
		auto rgb_layers = m_rgb_network->layer_sizes();
		layers.insert(layers.end(), rgb_layers.begin(), rgb_layers.end());
		return layers;
	}

	// std::vector<std::pair<uint32_t, uint32_t>> layer_sizes_delta() const override {
		// return m_delta_network->layer_sizes();
	// }

	uint32_t width(uint32_t layer) const override {
		if (layer == 0) {
			return m_pos_encoding->padded_output_width();
		} else if (layer < m_density_network->num_forward_activations() + 1) {
			return m_density_network->width(layer - 1);
		} else if (layer == m_density_network->num_forward_activations() + 1) {
			return m_rgb_network_input_width;
		} else {
			return m_rgb_network->width(layer - 2 - m_density_network->num_forward_activations());
		}
	}

	uint32_t num_forward_activations() const override {
		return m_density_network->num_forward_activations() + m_rgb_network->num_forward_activations() + 2;
	}

	std::pair<const T*, tcnn::MatrixLayout> forward_activations(const tcnn::Context& ctx, uint32_t layer) const override {
		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
		if (layer == 0) {
			return {forward.density_network_input.data(), m_pos_encoding->preferred_output_layout()};
		} else if (layer < m_density_network->num_forward_activations() + 1) {
			return m_density_network->forward_activations(*forward.density_network_ctx, layer - 1);
		} else if (layer == m_density_network->num_forward_activations() + 1) {
			return {forward.rgb_network_input.data(), m_dir_encoding->preferred_output_layout()};
		} else {
			return m_rgb_network->forward_activations(*forward.rgb_network_ctx, layer - 2 - m_density_network->num_forward_activations());
		}
	}

	const std::shared_ptr<tcnn::Encoding<T>>& encoding() const {
		return m_pos_encoding;
	}

	const std::shared_ptr<tcnn::Encoding<T>>& dir_encoding() const {
		return m_dir_encoding;
	}
	
	// const std::shared_ptr<tcnn::DifferentiableObject<float,T,float>>& delta_network() const {
	const std::shared_ptr<DeltaNetwork<T>>& delta_network() const {
		return m_delta_network;
	}

	void reset_delta_network() {
		m_delta_network = std::make_shared<DeltaNetwork<T>>();
		
		tcnn::GPUMemory<char> params_buffer;
		uint32_t n_params = 8;

		params_buffer.resize(sizeof(T) * n_params * 3 + sizeof(float) * n_params * 1);
		params_buffer.memset(0);

		float* new_m_params_full_precision = (float*)(params_buffer.data());
		T* new_m_params                = (T*)(params_buffer.data() + sizeof(float) * n_params);
		T* new_m_params_backward       = (T*)(params_buffer.data() + sizeof(float) * n_params + sizeof(T) * n_params);
		T* new_m_param_gradients       = (T*)(params_buffer.data() + sizeof(float) * n_params + sizeof(T) * n_params * 2);

		uint32_t offset = 0;
		tcnn::pcg32 rnd{1337};

		m_delta_network->initialize_params(
			rnd,
			new_m_params_full_precision + offset,
			new_m_params + offset,
			new_m_params + offset,
			new_m_params_backward + offset,
			new_m_param_gradients + offset
		);

		// tcnn::linear_kernel(debug_log_params<T>, 0, stream, 1u, m_delta_network->rotation()->params(), m_delta_network->transition()->params());
	}

	void init_accumulation_movement() {

		uint32_t n_params = accumulated_transition->n_params() + accumulated_rotation->n_params();

		m_accumulation_params_buffer.resize(sizeof(T) * n_params * 3 + sizeof(float) * n_params * 1);
		m_accumulation_params_buffer.memset(0);

		float* params_full_precision = (float*)(m_accumulation_params_buffer.data());
		T* params                = (T*)(m_accumulation_params_buffer.data() + sizeof(float) * n_params);
		T* backward_params       = (T*)(m_accumulation_params_buffer.data() + sizeof(float) * n_params + sizeof(T) * n_params);
		T* gradients       = (T*)(m_accumulation_params_buffer.data() + sizeof(float) * n_params + sizeof(T) * n_params * 2);

		uint32_t offset = 0;
		float scale = 1.0f;
		tcnn::pcg32 rnd{1337};
		printf("accumulated transition param num:%d\n",accumulated_transition->n_params());
		
		accumulated_transition->initialize_params(
			rnd,
			params_full_precision + offset,
			params + offset,
			params + offset,
			backward_params + offset,
			gradients + offset,
			scale
		);

		tcnn::generate_random_uniform<float>(rnd, 4u, params_full_precision + offset, 0.000f, 0.000f);

		offset += accumulated_transition->n_params();

		accumulated_rotation->initialize_params(
			rnd,
			params_full_precision + offset,
			params + offset,
			params + offset,
			backward_params + offset,
			gradients + offset,
			scale
		);


		#if rotation_reprensentation
			// unit matrix (1,0,0,0,1,0,0,0,1)
			tcnn::generate_random_uniform<float>(rnd, 1u, params_full_precision + offset, 1.000f, 1.000f);
			tcnn::generate_random_uniform<float>(rnd, 3u, params_full_precision + offset + 1u, 0.000f, 0.000f);
			// tcnn::generate_random_uniform<float>(rnd, 3u, params_full_precision + offset + 1u, 1.000f, 1.000f);
			tcnn::generate_random_uniform<float>(rnd, 1u, params_full_precision + offset + 4u, 1.000f, 1.000f);
			tcnn::generate_random_uniform<float>(rnd, 3u, params_full_precision + offset + 5u, 0.000f, 0.000f);
			// tcnn::generate_random_uniform<float>(rnd, 3u, params_full_precision + offset + 5u, 1.000f, 1.000f);
			tcnn::generate_random_uniform<float>(rnd, 1u, params_full_precision + offset + 8u, 1.000f, 1.000f);
		#else
			tcnn::generate_random_uniformc<float>(rnd, 3u, params_full_precision + offset, 0.000f, 0.000f);
			tcnn::generate_random_uniform<float>(rnd, 1u, params_full_precision + offset + 3u, 1.000f, 1.000f);
		#endif


		// initialize_params is only expected to initialize m_params_full_precision. Cast and copy these over!
		parallel_for_gpu(n_params, [params_fp=params_full_precision, params=params] __device__ (size_t i) {
			params[i] = (T)params_fp[i];
		});
		CUDA_CHECK_THROW(cudaDeviceSynchronize());

		printf("******finish init accumulation parameters\n*****");
		return;
	}

	const std::shared_ptr<TrainableBuffer<1, 1, T>>& rotation() const {
		return accumulated_rotation;
	}

	const std::shared_ptr<TrainableBuffer<1, 1, T>>& transition() const {
		return accumulated_transition;
	}

	const float& variance() const{
		return m_variance;
	}


	const float cos_anneal_ratio() const{
        if (m_anneal_end == 0) {
            return 1.0;
		}
        else {
			// printf("m_training_step:%d, m_anneal_end:%d",m_training_step,m_anneal_end);
			// printf("m_nerf_network cos_anneal_ratio:%f", min(1.0, (float)m_training_step / m_anneal_end));
            return min(1.0, (float)m_training_step / m_anneal_end);
		}
	}

	void set_anneal_end(const int& anneal_end){
		m_anneal_end = anneal_end; 
	}

	tcnn::json hyperparams() const override {
		json density_network_hyperparams = m_density_network->hyperparams();
		density_network_hyperparams["n_output_dims"] = m_density_network->padded_output_width();
		return {
			{"otype", "NgpNetwork"},
			{"pos_encoding", m_pos_encoding->hyperparams()},
			{"dir_encoding", m_dir_encoding->hyperparams()},
			{"density_network", density_network_hyperparams},
			{"rgb_network", m_rgb_network->hyperparams()},
			{"variance_network", m_variance_network->hyperparams()},
			// {"", m_delta_network->hyperparams()},
		};
	}
	#if VARIANCE_MLP
		std::unique_ptr<tcnn::Network<T>> m_variance_network;
	#else
		std::shared_ptr<TrainableBuffer<1, 1, T>> m_variance_network;
	#endif

	const uint32_t& training_step() const{
		return m_training_step;
	}
	uint32_t m_training_step;
	uint32_t m_anneal_end;
	uint32_t indeed_batch_size;

	tcnn::GPUMatrixDynamic<float> m_smpl_skinning_weights;
	tcnn::GPUMatrixDynamic<float> m_joints_RT;

	uint32_t m_mesh_face_num;
	uint32_t m_mesh_vert_num;
	tcnn::GPUMatrixDynamic<uint32_t> m_mesh_triangle_idx;
	tcnn::GPUMatrixDynamic<float> m_mesh_triangle;
	tcnn::GPUMatrixDynamic<float> m_mesh_vertices;

	bool m_train_canonical = true;
	bool m_train_delta = false;
	bool m_use_delta = true;

	bool m_train_residual= false;
	bool m_use_residual = false;

	void accumulate_global_movement(cudaStream_t stream){
		#if rotation_reprensentation
			tcnn::linear_kernel(accumulate_global_movement_rotation_6d_kernel<T>, 0, stream, 1u,
					m_delta_network->rotation()->params(), m_delta_network->transition()->params(),
					// accumulated_rotation->params_inference(), accumulated_transition->params_inference());
					accumulated_rotation->params(), accumulated_transition->params());

		CUDA_CHECK_THROW(cudaMemcpy(accumulated_rotation->params_inference(),accumulated_rotation->params(), sizeof(T)*accumulated_rotation->n_params(), cudaMemcpyDeviceToDevice));
		CUDA_CHECK_THROW(cudaMemcpy(accumulated_transition->params_inference(),accumulated_transition->params(), sizeof(T)*accumulated_transition->n_params(), cudaMemcpyDeviceToDevice));
		#else
			tcnn::linear_kernel(accumulate_global_movement_rotation_quaternion_kernel<T>, 0, stream, 1u,
					m_delta_network->rotation()->params(), m_delta_network->transition()->params(),
					accumulated_rotation->params(), accumulated_transition->params());
		#endif

	}

	void set_time_log_file(std::string file_name){
		m_time_log_file = file_name;
	}

	const char* time_log_file() const{
		return m_time_log_file.c_str();
	}
	
	void save_global_movement(cudaStream_t stream, tcnn::json & network_config) {
		// when save_global_movement, we have not accumulated the global movement, so we need to accumulate it first

		tcnn::GPUMemory<T> save_accumulated_rotation;
		tcnn::GPUMemory<T> save_accumulated_transition;

		save_accumulated_rotation.resize(sizeof(T) * accumulated_rotation->n_params());
		save_accumulated_transition.resize(sizeof(T) * accumulated_transition->n_params());

		#if rotation_reprensentation
			tcnn::linear_kernel(save_global_movement_rotation_6d_kernel<T>, 0, stream, 1u,
					m_delta_network->rotation()->params(), m_delta_network->transition()->params(),
					accumulated_rotation->params(), accumulated_transition->params(),
					save_accumulated_rotation.data(), save_accumulated_transition.data());

		// CUDA_CHECK_THROW(cudaMemcpy(accumulated_rotation->params_inference(),accumulated_rotation->params(), sizeof(T)*accumulated_rotation->n_params(), cudaMemcpyDeviceToDevice));
		// CUDA_CHECK_THROW(cudaMemcpy(accumulated_transition->params_inference(),accumulated_transition->params(), sizeof(T)*accumulated_transition->n_params(), cudaMemcpyDeviceToDevice));
		#else
			printf("not implemented!\n");
			exit(1);
			tcnn::linear_kernel(accumulate_global_movement_rotation_quaternion_kernel<T>, 0, stream, 1u,
					accumulated_rotation->params(), accumulated_transition->params(),
					save_accumulated_rotation.data(), save_accumulated_transition.data());
		#endif

		// network_config["snapshot"]["rotation"] = tcnn::gpu_memory_to_json_binary(save_accumulated_rotation.data(), sizeof(T) * accumulated_rotation->n_params());
		// network_config["snapshot"]["transition"] = tcnn::gpu_memory_to_json_binary(save_accumulated_transition.data(), sizeof(T) * accumulated_transition->n_params());
	
		network_config["snapshot"]["rotation"] = tcnn::gpu_memory_to_json_binary(accumulated_rotation->params(), sizeof(T) * accumulated_rotation->n_params());
		network_config["snapshot"]["transition"] = tcnn::gpu_memory_to_json_binary(accumulated_transition->params(), sizeof(T) * accumulated_transition->n_params());
	
	}

	void load_global_movement(const tcnn::json network_config) {
		printf("******start load global movement parameters\n*****");

		tcnn::GPUMemory<T> params_hp = network_config["snapshot"]["rotation"];

		size_t n_params = params_hp.size();
		
		printf("rotation n_params: %d\n", n_params);

		parallel_for_gpu(n_params, [params=accumulated_rotation->params(), params_hp=params_hp.data()] __device__ (size_t i) {
			params[i] = (T)params_hp[i];
		});

		params_hp = network_config["snapshot"]["transition"];

		n_params = params_hp.size();

		printf("transition n_params: %d\n", n_params);

		parallel_for_gpu(n_params, [params=accumulated_transition->params(), params_hp=params_hp.data()] __device__ (size_t i) {
			params[i] = (T)params_hp[i];
		});

		precision_t* rotation_quat_gpu = accumulated_rotation->params();
		precision_t* transition_gpu = accumulated_transition->params();
		// std::vector<precision_t> rotation_quat(4);
		// std::vector<precision_t> transition(3);
		// CUDA_CHECK_THROW(cudaMemcpy(rotation_quat.data(), rotation_quat_gpu, (4) * sizeof(precision_t), cudaMemcpyDeviceToHost));
		// CUDA_CHECK_THROW(cudaMemcpy(transition.data(), transition_gpu, (3) * sizeof(precision_t), cudaMemcpyDeviceToHost));
		// fprintf(fp, "current frame: %d\n", current_training_time_frame);
		// fprintf(fp, "rotation: %f, %f, %f, %f\n", (float)rotation_quat[0],(float)rotation_quat[1],(float)rotation_quat[2],(float)rotation_quat[3]);
		std::vector<precision_t> rotation_quat(9);
		std::vector<precision_t> transition(3);
		CUDA_CHECK_THROW(cudaMemcpy(rotation_quat.data(), rotation_quat_gpu, (9) * sizeof(precision_t), cudaMemcpyDeviceToHost));
		CUDA_CHECK_THROW(cudaMemcpy(transition.data(), transition_gpu, (3) * sizeof(precision_t), cudaMemcpyDeviceToHost));
		printf("rotation: %f, %f, %f\n", (float)rotation_quat[0],(float)rotation_quat[1],(float)rotation_quat[2]);
		printf("rotation: %f, %f, %f\n", (float)rotation_quat[3],(float)rotation_quat[4],(float)rotation_quat[5]);
		printf("rotation: %f, %f, %f\n", (float)rotation_quat[6],(float)rotation_quat[7],(float)rotation_quat[8]);
		printf("transition: %f, %f, %f\n\n", (float)transition[0],(float)transition[1],(float)transition[2]);

	}

	void save_local_movement(cudaStream_t stream, tcnn::json & network_config) {
		// when save_global_movement, we have not accumulated the global movement, so we need to accumulate it first

		// network_config["snapshot"]["rotation"] = tcnn::gpu_memory_to_json_binary(save_accumulated_rotation.data(), sizeof(T) * accumulated_rotation->n_params());
		// network_config["snapshot"]["transition"] = tcnn::gpu_memory_to_json_binary(save_accumulated_transition.data(), sizeof(T) * accumulated_transition->n_params());
	
		network_config["snapshot"]["local_rotation"] = tcnn::gpu_memory_to_json_binary(m_delta_network->rotation()->params(), sizeof(T) * m_delta_network->rotation()->n_params());
		network_config["snapshot"]["local_transition"] = tcnn::gpu_memory_to_json_binary(m_delta_network->transition()->params(), sizeof(T) * m_delta_network->transition()->n_params());
	
	}

	void load_local_movement(const tcnn::json network_config) {
		printf("******start load local movement parameters\n*****");

		tcnn::GPUMemory<T> params_hp = network_config["snapshot"]["local_rotation"];

		size_t n_params = params_hp.size();
		
		printf("rotation n_params: %d\n", n_params);

		parallel_for_gpu(n_params, [params=m_delta_network->rotation()->params(), params_hp=params_hp.data()] __device__ (size_t i) {
			params[i] = (T)params_hp[i];
		});

		params_hp = network_config["snapshot"]["local_transition"];

		n_params = params_hp.size();

		printf("transition n_params: %d\n", n_params);

		parallel_for_gpu(n_params, [params=m_delta_network->transition()->params(), params_hp=params_hp.data()] __device__ (size_t i) {
			params[i] = (T)params_hp[i];
		});

		// precision_t* rotation_quat_gpu = accumulated_rotation->params();
		// precision_t* transition_gpu = accumulated_transition->params();
		// // std::vector<precision_t> rotation_quat(4);
		// // std::vector<precision_t> transition(3);
		// // CUDA_CHECK_THROW(cudaMemcpy(rotation_quat.data(), rotation_quat_gpu, (4) * sizeof(precision_t), cudaMemcpyDeviceToHost));
		// // CUDA_CHECK_THROW(cudaMemcpy(transition.data(), transition_gpu, (3) * sizeof(precision_t), cudaMemcpyDeviceToHost));
		// // fprintf(fp, "current frame: %d\n", current_training_time_frame);
		// // fprintf(fp, "rotation: %f, %f, %f, %f\n", (float)rotation_quat[0],(float)rotation_quat[1],(float)rotation_quat[2],(float)rotation_quat[3]);
		// std::vector<precision_t> rotation_quat(9);
		// std::vector<precision_t> transition(3);
		// CUDA_CHECK_THROW(cudaMemcpy(rotation_quat.data(), rotation_quat_gpu, (9) * sizeof(precision_t), cudaMemcpyDeviceToHost));
		// CUDA_CHECK_THROW(cudaMemcpy(transition.data(), transition_gpu, (3) * sizeof(precision_t), cudaMemcpyDeviceToHost));
		// printf("rotation: %f, %f, %f\n", (float)rotation_quat[0],(float)rotation_quat[1],(float)rotation_quat[2]);
		// printf("rotation: %f, %f, %f\n", (float)rotation_quat[3],(float)rotation_quat[4],(float)rotation_quat[5]);
		// printf("rotation: %f, %f, %f\n", (float)rotation_quat[6],(float)rotation_quat[7],(float)rotation_quat[8]);
		// printf("transition: %f, %f, %f\n\n", (float)transition[0],(float)transition[1],(float)transition[2]);

	}

private:


	std::unique_ptr<tcnn::Network<T>> m_density_network;
	std::unique_ptr<tcnn::Network<T>> m_rgb_network;
	std::shared_ptr<tcnn::Encoding<T>> m_pos_encoding;
	std::shared_ptr<tcnn::Encoding<T>> m_dir_encoding;

	std::shared_ptr<DeltaNetwork<T>> m_delta_network;
	// std::shared_ptr<tcnn::DifferentiableObject<float,T,float>> m_delta_network;
	// DeltaNetwork<T>* m_delta_network;

	// std::shared_ptr<TrainableBuffer<1, 1, T>> accumulated_transition;
	// std::shared_ptr<TrainableBuffer<1, 1, T>> accumulated_rotation;
	std::shared_ptr<TrainableBuffer<1, 1, T>> accumulated_transition;
	std::shared_ptr<TrainableBuffer<1, 1, T>> accumulated_rotation;


	tcnn::GPUMemory<char> m_accumulation_params_buffer;

	std::string m_time_log_file;

	// variance
	float m_variance;
	T m_sdf_bias;

	uint32_t m_rgb_network_input_width;
	uint32_t m_density_network_input_width;
	uint32_t m_n_pos_dims;
	uint32_t m_n_dir_dims;
	uint32_t m_n_extra_dims; // extra dimensions are assumed to be part of a compound encoding with dir_dims
	uint32_t m_dir_offset;

	// // Storage of forward pass data
	struct ForwardContext : public tcnn::Context {
		tcnn::GPUMatrixDynamic<T> density_network_input;
		tcnn::GPUMatrixDynamic<T> density_network_output;
		tcnn::GPUMatrixDynamic<T> rgb_network_input;
		tcnn::GPUMatrix<T> rgb_network_output;
		tcnn::GPUMatrixDynamic<T> variance_network_input;
		tcnn::GPUMatrixDynamic<T> variance_network_output;
		tcnn::GPUMatrixDynamic<float> dSDF_dPos;
		tcnn::GPUMatrixDynamic<float> delta_network_input;
		tcnn::GPUMatrixDynamic<float> delta_network_output;

		std::unique_ptr<Context> pos_encoding_ctx;
		std::unique_ptr<Context> dir_encoding_ctx;

		std::unique_ptr<Context> density_network_ctx;
		std::unique_ptr<Context> rgb_network_ctx;
		std::unique_ptr<Context> variance_network_ctx;
		std::unique_ptr<Context> delta_network_ctx;
	};
};

NGP_NAMESPACE_END
