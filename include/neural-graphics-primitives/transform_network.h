/** @file   transform_network.h
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
#include <neural-graphics-primitives/common_operation.cuh>

#define residual_encoding 0
#define rotation_reprensentation 1 // 1 refers to rotation 6d 

NGP_NAMESPACE_BEGIN

template <typename T>
// class DeltaNetwork: public tcnn::Network<float, T> {
class DeltaNetwork: public tcnn::DifferentiableObject<float, T, float> {
// class DeltaNetwork {
	public:
	

		DeltaNetwork(uint32_t n_pos_dims = 6u) : m_n_pos_dims{n_pos_dims}{

			m_transition = std::make_shared<TrainableBuffer<1, 1, T>>(Eigen::Matrix<int, 1, 1>{(int)4});
			
			#if rotation_reprensentation
				m_rotation = std::make_shared<TrainableBuffer<1, 1, T>>(Eigen::Matrix<int, 1, 1>{(int)8});
			#else
				m_rotation = std::make_shared<TrainableBuffer<1, 1, T>>(Eigen::Matrix<int, 1, 1>{(int)4});
			#endif

			printf("delta_network m_n_pos_dims: %d\n", m_n_pos_dims);

		}

		void inference_mixed_precision_impl(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<float>& output, bool use_inference_params = true)  {
		// void inference_mixed_precision(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<float>& output, bool use_inference_params = true)  {
			forward_impl(stream, input, &output, use_inference_params, false);
			return; // Since we need gradient in inference. Should it be replaced by forward_impl?
		}


		std::unique_ptr<tcnn::Context> forward_impl(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<float>* output = nullptr, bool use_inference_params = false, bool prepare_input_gradients = false)  {
		// std::unique_ptr<tcnn::Context> forward(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<float>* output = nullptr, bool use_inference_params = false, bool prepare_input_gradients = false)  {
			uint32_t batch_size = input.n();

			// debug
			// tcnn::GPUMatrixDynamic<float> test_input{input.m(),input.n(), stream};

			// tcnn::linear_kernel(set_constant_value_view_vector_test<float>, 0, stream,
			// 	batch_size * input.m(), input.m(), 1.0f, test_input.view());

			auto forward = std::make_unique<ForwardContext>();

			// if (!m_train_delta){
			// 	tcnn::linear_kernel(fill_positions_view<float,float>, 0, stream, batch_size * 3u, 3u,
			// 			input.view(), output->view() );
			// 	return forward;
			// }


			// tcnn::linear_kernel(add_global_movement<float, T>, 0, stream, batch_size * 3u,
						// m_transition->params(), input.view(), output->view());
			#if rotation_reprensentation
				tcnn::linear_kernel(add_global_movement_with_rotation_6d<float, T>, 0, stream, batch_size,
							m_rotation->params(), m_transition->params(), first_frame_offset, input.view(), output->view());
			#else
				printf("some error not implemented! 1\n");
				tcnn::linear_kernel(add_global_movement_with_rotation_quaternion<float, T>, 0, stream, batch_size,
							m_rotation->params(), m_transition->params(), input.view(), output->view());
			#endif

			// forward->mlp_network_input = tcnn::GPUMatrixDynamic<T>{m_mlp_network_input_width, batch_size, stream, m_pos_encoding->preferred_output_layout()};


			// forward->pos_encoding_ctx = m_pos_encoding->forward(
			// 	stream,
			// 	input,
			// 	// test_input, //debug
			// 	&forward->mlp_network_input,
			// 	use_inference_params,
			// 	prepare_input_gradients
			// 	// true
			// );

			// // auto forward = std::make_unique<ForwardContext>();

			// // forward->mlp_network_input = tcnn::GPUMatrixDynamic<T>{m_mlp_network_input_width, batch_size, stream, m_pos_encoding->preferred_output_layout()};


			// // forward->pos_encoding_ctx = m_pos_encoding->forward(
			// // 	stream,
			// // 	input,
			// // 	&forward->mlp_network_input,
			// // 	use_inference_params,
			// // 	prepare_input_gradients
			// // 	// true
			// // );

			// forward->mlp_network_output = tcnn::GPUMatrixDynamic<T>{m_mlp_network->padded_output_width(), batch_size, stream, output->layout()};
			// forward->mlp_network_ctx = m_mlp_network->forward(stream, forward->mlp_network_input, &forward->mlp_network_output, use_inference_params, prepare_input_gradients);

			// tcnn::linear_kernel(extract_delta_xyz<float, T>, 0, stream, batch_size * input_width(), input_width(),
			// 			forward->mlp_network_output.view(), input.view(), output->view());

			// tcnn::linear_kernel(debug_log<T>, 0, stream, m_mlp_network->padded_output_width(), forward->mlp_network_output.view());
			return forward;

		}


		void backward_impl(
		// void backward(
			cudaStream_t stream,
			const tcnn::Context& ctx,
			const tcnn::GPUMatrixDynamic<float>& input,
			const tcnn::GPUMatrixDynamic<float>& output,
			const tcnn::GPUMatrixDynamic<float>& dL_doutput,
			tcnn::GPUMatrixDynamic<float>* dL_dinput = nullptr,
			bool use_inference_params = false,
			tcnn::EGradientMode param_gradients_mode = tcnn::EGradientMode::Overwrite
		)  {

			uint32_t batch_size = input.n();

			// if (!m_train_delta){
			// 	// if (param_gradients_mode != tcnn::EGradientMode::Overwrite) {
			// 	if (dL_dinput != nullptr) {
			// 		tcnn::linear_kernel(fill_positions_view<float, float>, 0, stream,
			// 				batch_size * 3u, 3u, dL_doutput.view(), dL_dinput->view());
			// 	}
			// 	return;
			// }


			// if (param_gradients_mode != tcnn::EGradientMode::Ignore) {
			// 	// atomic operation
		
			// 	// CUDA_CHECK_THROW(cudaMemset(m_transition->gradients(), 0, sizeof(T)*m_transition->n_params()));
			// 	CUDA_CHECK_THROW(cudaMemsetAsync(m_transition->gradients(), 0, sizeof(T)*m_transition->n_params(), stream));

			// 	for (uint32_t i = 0; i < 3; i++){
			// 		T gradients = T(0.0f);

			// 		tcnn::GPUMatrixDynamic<T> dL_dtransition{1, batch_size, stream};

			// 		tcnn::linear_kernel(add_transition_view_to_loss<T,float>, 0, stream,
			// 			batch_size,
			// 			dL_doutput.view(),
			// 			m_rotation->params(),
			// 			i,
			// 			dL_dtransition.view()
			// 		);
			// 		// printf("dL_dvariance shape:%d,%d\n",dL_dvariance.m(),dL_dvariance.n());
			// 		gradients = tcnn::reduce_sum(dL_dtransition.data(),batch_size,stream);
			// 		// printf("atomicAdd:%f, reduce_sum:%f\n",(float)(variance_gradients_cpu[0]),(float)gradients);
			// 		cudaMemcpy(m_transition->gradients() + i, &gradients, sizeof(T), cudaMemcpyHostToDevice); //将数据从CPU传递到GPU
			// 	}
			// }


			// tcnn::GPUMatrixDynamic<T> dL_drotation_quat_each{4, batch_size, stream};
			tcnn::GPUMatrixDynamic<T> dL_drotation_quat_each;
			tcnn::GPUMatrixDynamic<T> dL_dtransition_each{3, batch_size, stream};
			tcnn::GPUMatrixDynamic<T> dL_dinput_each{3, batch_size, stream};

			uint32_t rotation_dof;
			#if rotation_reprensentation
				rotation_dof = 6;
				dL_drotation_quat_each = tcnn::GPUMatrixDynamic<T>{rotation_dof, batch_size, stream};

				tcnn::linear_kernel(add_loss_to_rotation_6d_each<T,float>, 0, stream,
					batch_size,
					dL_doutput.view(),
					input.view(),
					m_rotation->params(),
					m_transition->params(),
					first_frame_offset,
					dL_drotation_quat_each.view(),
					dL_dtransition_each.view(),
					dL_dinput_each.view()
				);
			
			#else
				printf("some error not implemented! 2\n");
				rotation_dof = 4;
				dL_drotation_quat_each = tcnn::GPUMatrixDynamic<T>{rotation_dof, batch_size, stream};

				tcnn::linear_kernel(add_loss_to_rotation_quaternion_each<T,float>, 0, stream,
					batch_size,
					dL_doutput.view(),
					input.view(),
					m_rotation->params(),
					dL_drotation_quat_each.view(),
					dL_dinput_each.view()
				);
			
			#endif

			if (param_gradients_mode != tcnn::EGradientMode::Ignore) {
				// calculate each position's gradients
				CUDA_CHECK_THROW(cudaMemsetAsync(m_rotation->gradients(), 0, sizeof(T)*m_rotation->n_params(), stream));

				for (uint32_t i = 0; i < rotation_dof; i++){
					T gradients = T(0.0f);

					tcnn::GPUMatrixDynamic<T> dL_drotation{1, batch_size, stream};

					tcnn::linear_kernel(add_rotation_view_to_loss<T,T>, 0, stream,
						batch_size,
						dL_drotation_quat_each.view(),
						i,
						dL_drotation.view()
					);
					gradients = tcnn::reduce_sum(dL_drotation.data(),batch_size,stream);

					cudaMemcpy(m_rotation->gradients() + i, &gradients, sizeof(T), cudaMemcpyHostToDevice); //将数据从CPU传递到GPU
				}


				CUDA_CHECK_THROW(cudaMemsetAsync(m_transition->gradients(), 0, sizeof(T)*m_transition->n_params(), stream));

				for (uint32_t i = 0; i < 3; i++){
					T gradients = T(0.0f);

					tcnn::GPUMatrixDynamic<T> dL_dtransition{1, batch_size, stream};

					tcnn::linear_kernel(add_transition_view_to_loss<T,T>, 0, stream,
						batch_size,
						dL_dtransition_each.view(),
						i,
						dL_dtransition.view()
					);
					// printf("dL_dvariance shape:%d,%d\n",dL_dvariance.m(),dL_dvariance.n());
					gradients = tcnn::reduce_sum(dL_dtransition.data(),batch_size,stream);
					// printf("atomicAdd:%f, reduce_sum:%f\n",(float)(variance_gradients_cpu[0]),(float)gradients);
					cudaMemcpy(m_transition->gradients() + i, &gradients, sizeof(T), cudaMemcpyHostToDevice); //将数据从CPU传递到GPU
				}
			}

			if (dL_dinput != nullptr) {
				printf("some error not implemented! 3\n");
				tcnn::linear_kernel(fill_positions_view<float, T>, 0, stream,
							batch_size * 3u, 3u, dL_dinput_each.view(), dL_dinput->view());
				// tcnn::linear_kernel(fill_positions_view<float, float>, 0, stream,
				// 			batch_size * 3u, 3u, dL_doutput.view(), dL_dinput->view());
			}


		}

		// void backward_backward_input(
		// 	cudaStream_t stream,
		// 	const tcnn::Context& ctx,
		// 	const tcnn::GPUMatrixDynamic<float>& input,
		// 	const tcnn::GPUMatrixDynamic<float>& dL_ddLdinput,
		// 	// const tcnn::GPUMatrixDynamic<T>& dL_doutput,
		// 	// tcnn::GPUMatrixDynamic<T>* dL_ddLdoutput = nullptr,
		// 	// tcnn::GPUMatrixDynamic<T>* dL_dinput = nullptr,
		// 	bool use_inference_params = false,
		// 	tcnn::EGradientMode param_gradients_mode = tcnn::EGradientMode::Overwrite
		// ) {

		// 	const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
		// 	uint32_t batch_size = input.n();

		// 	tcnn::GPUMatrixDynamic<T> dL_denc_output{ m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		// 	tcnn::GPUMatrixDynamic<T> dL_dmlp_output{ m_mlp_network->padded_output_width(), batch_size, stream};

		// 	dL_dmlp_output.memset_async(stream, 0);
		// 	tcnn::linear_kernel(set_constant_value_view<T>, 0, stream,
		// 		batch_size, 1.0f, dL_dmlp_output.view());

		// 	m_mlp_network->backward(stream, *forward.mlp_network_ctx, forward.mlp_network_input, 
		// 			forward.mlp_network_output, 
		// 			dL_dmlp_output, 
		// 			&dL_denc_output, use_inference_params, tcnn::EGradientMode::Ignore);
		
		// 	tcnn::GPUMatrixDynamic<T> pos_encoding_dy{m_pos_encoding->padded_output_width(), batch_size, stream};
		// 	m_pos_encoding->backward_backward_input(
		// 		stream, 
		// 		*forward.pos_encoding_ctx,
		// 		input.slice_rows(0, m_pos_encoding->input_width()),
		// 		dL_ddLdinput, 
		// 		dL_denc_output, 
		// 		&pos_encoding_dy, // assume dl_ddensity_dx_dx === 0
		// 		nullptr,
		// 		// dL_dsdf_dinput_d_input,
		// 		use_inference_params, 
		// 		tcnn::EGradientMode::Accumulate
		// 	);

		// 	m_mlp_network->backward_backward_input(
		// 		stream, 
		// 		*forward.mlp_network_ctx,
		// 		forward.mlp_network_input, 
		// 		pos_encoding_dy,
		// 		dL_dmlp_output,
		// 		nullptr, // assume dl_ddensity_dx_dx === 0
		// 		nullptr,
		// 		use_inference_params, 
		// 		tcnn::EGradientMode::Accumulate
		// 	);

		// }

		void set_params(T* params, T* inference_params, T* backward_params, T* gradients)  {
			size_t offset = 0;

			m_transition->set_params(
				params + offset,
				inference_params + offset,
				backward_params + offset,
				gradients + offset
			);
			offset += m_transition->n_params();

			m_rotation->set_params(
				params + offset,
				inference_params + offset,
				backward_params + offset,
				gradients + offset
			);
			offset += m_rotation->n_params();


		}

		void initialize_params(tcnn::pcg32& rnd, float* params_full_precision, T* params, T* inference_params, T* backward_params, T* gradients, float scale = 1)  {
			size_t offset = 0;

			m_transition->initialize_params(
				rnd,
				params_full_precision + offset,
				params + offset,
				inference_params + offset,
				backward_params + offset,
				gradients + offset,
				scale
			);
			int transition_n_params = m_transition->n_params();

			tcnn::pcg32 m_rng{1337};
			tcnn::generate_random_uniform<float>(m_rng, transition_n_params, params_full_precision + offset, 0.000f, 0.000f);

			offset += transition_n_params;
			printf("transition_n_params:%d\n",transition_n_params); // 1

			m_rotation->initialize_params(
				rnd,
				params_full_precision + offset,
				params + offset,
				inference_params + offset,
				backward_params + offset,
				gradients + offset,
				scale
			);
			int rotation_n_params = m_rotation->n_params();

			#if rotation_reprensentation
				// unit 6d (1,0,0,0,1,0)
				tcnn::generate_random_uniform<float>(m_rng, 1u, params_full_precision + offset, 1.000f, 1.000f);
				tcnn::generate_random_uniform<float>(m_rng, 3u, params_full_precision + offset + 1u, 0.000f, 0.000f);
				tcnn::generate_random_uniform<float>(m_rng, 1u, params_full_precision + offset + 4u, 1.000f, 1.000f);
				tcnn::generate_random_uniform<float>(m_rng, 1u, params_full_precision + offset + 5u, 0.000f, 0.000f);

			#else
				// unit quat (0,0,0,1)
				tcnn::generate_random_uniform<float>(m_rng, 3u, params_full_precision + offset, 0.000f, 0.000f);
				tcnn::generate_random_uniform<float>(m_rng, 1u, params_full_precision + offset + 3u, 1.000f, 1.000f);
			#endif

			offset += rotation_n_params;
			printf("rotation_n_params:%d\n",rotation_n_params); // 1


		}

		size_t n_params() const  {
			return m_transition->n_params() + m_rotation->n_params();
		}

		uint32_t padded_output_width() const  {
			// return 4;
			return 6;
			// return (uint32_t)16;
		}

		uint32_t input_width() const  {
			return m_n_pos_dims;
		}

		uint32_t output_width() const  {
			return 6;
		}

		uint32_t n_extra_dims() const {
			return 0u;
		}

		uint32_t required_input_alignment() const  {
			return 1; // No alignment required due to encoding
		}

		std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const  {
			return std::vector<std::pair<uint32_t, uint32_t>>();
		}

		// uint32_t width(uint32_t layer) const  {
		// 	if (layer == 0) {
		// 		return m_pos_encoding->padded_output_width();
		// 	} 
		// 	else if (layer < m_mlp_network->num_forward_activations() + 1) {
		// 		return m_mlp_network->width(layer - 1);
		// 	}
		// }

		// uint32_t num_forward_activations() const  {
		// 	return m_mlp_network->num_forward_activations();
		// }

		std::pair<const T*, tcnn::MatrixLayout> forward_activations(const tcnn::Context& ctx, uint32_t layer) const  {
			printf("forward_activations not implemented in DeltaNetwork!\n");
			exit(1);
			// const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
			// if (layer == 0) {
			// 	return {forward.density_network_input.data(), m_pos_encoding->preferred_output_layout()};
			// } else if (layer < m_density_network->num_forward_activations() + 1) {
			// 	return m_density_network->forward_activations(*forward.density_network_ctx, layer - 1);
			// } else if (layer == m_density_network->num_forward_activations() + 1) {
			// 	return {forward.rgb_network_input.data(), m_dir_encoding->preferred_output_layout()};
			// } else {
			// 	return m_rgb_network->forward_activations(*forward.rgb_network_ctx, layer - 2 - m_density_network->num_forward_activations());
			// }
		}

		tcnn::json hyperparams() const  {
			return {
				{"otype", "DeltaNetwork"},
				{"m_transition", m_transition->hyperparams()},
				{"m_rotation", m_rotation->hyperparams()},
			};
		}
		// bool m_train_delta = true; 

		const std::shared_ptr<TrainableBuffer<1, 1, T>>& rotation() const {
			return m_rotation;
		}

		const std::shared_ptr<TrainableBuffer<1, 1, T>>& transition() const {
			return m_transition;
		}

		Eigen::Vector3f first_frame_offset = Eigen::Vector3f::Constant(0);


	private:
		std::shared_ptr<TrainableBuffer<1, 1, T>> m_transition;
		std::shared_ptr<TrainableBuffer<1, 1, T>> m_rotation;

		uint32_t m_n_pos_dims;
		struct ForwardContext : public tcnn::Context {
			tcnn::GPUMatrixDynamic<T> network_input;
			tcnn::GPUMatrixDynamic<T> network_output;
			tcnn::GPUMatrixDynamic<float> dSDF_dPos;

			std::unique_ptr<Context> mlp_network_ctx;
		};

};



template <typename T>
// class DeltaNetwork: public tcnn::Network<float, T> {
class ResidualNetwork: public tcnn::DifferentiableObject<float, T, float> {
// class DeltaNetwork {
	public:
	

		ResidualNetwork(const tcnn::json residual_network_config, uint32_t n_pos_dims = 3u) : m_n_pos_dims{n_pos_dims}{
			tcnn::json m_pos_encoding_config;
			if (residual_network_config.contains("pos_encoding_config")) {
				m_pos_encoding_config = residual_network_config["pos_encoding_config"];
			}
			else{	
				#if residual_encoding
					tcnn::json m_pos_encoding_config = {
						{"otype", "HashGrid"},
						{"n_levels", 16},
						{"n_features_per_level", 2},
						{"log2_hashmap_size", 19},
						{"base_resolution", 16}
					};
				#else
					m_pos_encoding_config = {
						// {"otype", "Frequency"}, // Component type.
						{"otype", "HannWindowFrequency"}, // Component type.
						// {"otype", "Identity"}, // Component type.
						// {"n_frequencies", 12}   // Number of frequencies (sin & cos)
						{"n_frequencies", 6}   // Number of frequencies (sin & cos)
											// per encoded dimension.
					};
			#endif
			}
	

			tcnn::json m_mlp_network_config;
			if (residual_network_config.contains("mlp_network_config")) {
				m_mlp_network_config = residual_network_config["mlp_network_config"];
			}
			else{
				m_mlp_network_config = {
					{"otype", "FullyFusedMLP"},
					// {"otype", "CutLassMLP"},
					{"activation", "None"},
					{"output_activation", "None"},
					// {"n_neurons", 16},
					// {"n_hidden_layers", 0},
					{"n_neurons", 64},
					{"n_hidden_layers", 4},
					// {"n_input_dims", m_mlp_network_input_width},
					// {"n_input_dims", 16},
					{"n_output_dims", 16}
				};
			}


			m_last_layer_params_num = 64*16;
			m_pos_encoding.reset(tcnn::create_encoding<T>(n_pos_dims, m_pos_encoding_config, m_mlp_network_config.contains("otype") && (tcnn::equals_case_insensitive(m_mlp_network_config["otype"], "FullyFusedMLP") || tcnn::equals_case_insensitive(m_mlp_network_config["otype"], "MegakernelMLP")) ? 16u : 8u));

			m_mlp_network_input_width = m_pos_encoding->padded_output_width();
			m_mlp_network_config["n_input_dims"] = m_mlp_network_input_width;
			printf("Residual_network: m_mlp_network_input_width: %d\n", m_mlp_network_input_width);

			m_mlp_network.reset(tcnn::create_network<T>(m_mlp_network_config));
			tlog::info()
				<< "Residual_network_config: " << n_pos_dims
				<< "--[" << std::string(m_pos_encoding_config["otype"])
				<< "]-->" << m_pos_encoding->padded_output_width()
				<< "--[" << std::string(m_mlp_network_config["otype"])
				<< "(neurons=" << (int)m_mlp_network_config["n_neurons"] << ",layers=" << ((int)m_mlp_network_config["n_hidden_layers"]+2) << ")"
				<< "]-->" << 3
				;
		}

		void inference_mixed_precision_impl(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<float>& output, bool use_inference_params = true)  {
		// void inference_mixed_precision(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<float>& output, bool use_inference_params = true)  {
			forward_impl(stream, input, &output, use_inference_params, false);
			return; // Since we need gradient in inference. Should it be replaced by forward_impl?
		}


		std::unique_ptr<tcnn::Context> forward_impl(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<float>* output = nullptr, bool use_inference_params = false, bool prepare_input_gradients = false)  {
		// std::unique_ptr<tcnn::Context> forward(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<float>* output = nullptr, bool use_inference_params = false, bool prepare_input_gradients = false)  {
			uint32_t batch_size = input.n();

			auto forward = std::make_unique<ForwardContext>();

			forward->mlp_network_input = tcnn::GPUMatrixDynamic<T>{m_mlp_network_input_width, batch_size, stream, m_pos_encoding->preferred_output_layout()};

			forward->pos_encoding_ctx = m_pos_encoding->forward(
				stream,
				input,
				&forward->mlp_network_input,
				use_inference_params,
				prepare_input_gradients
				// true
			);


			forward->mlp_network_output = tcnn::GPUMatrixDynamic<T>{m_mlp_network->padded_output_width(), batch_size, stream, output->layout()};
			forward->mlp_network_output.memset_async(stream, 0);
			forward->mlp_network_ctx = m_mlp_network->forward(stream, forward->mlp_network_input, &forward->mlp_network_output, use_inference_params, prepare_input_gradients);

			tcnn::linear_kernel(extract_delta_xyz<float, T>, 0, stream, batch_size * input_width(), input_width(),
						forward->mlp_network_output.view(), input.view(), output->view());

			// tcnn::linear_kernel(debug_log<T>, 0, stream, m_mlp_network->padded_output_width(), forward->mlp_network_output.view());
			return forward;

		}


		void backward_impl(
		// void backward(
			cudaStream_t stream,
			const tcnn::Context& ctx,
			const tcnn::GPUMatrixDynamic<float>& input,
			const tcnn::GPUMatrixDynamic<float>& output,
			const tcnn::GPUMatrixDynamic<float>& dL_doutput,
			tcnn::GPUMatrixDynamic<float>* dL_dinput = nullptr,
			bool use_inference_params = false,
			tcnn::EGradientMode param_gradients_mode = tcnn::EGradientMode::Overwrite
		)  {

			const auto& forward = dynamic_cast<const ForwardContext&>(ctx);

			// Make sure our teporary buffers have the correct size for the given batch size
			uint32_t batch_size = input.n();

			tcnn::GPUMatrix<T> dL_dmlp_output{m_mlp_network->padded_output_width(), batch_size, stream};
			// CUDA_CHECK_THROW(cudaMemsetAsync(dL_dmlp_output.data(), 0, dL_dmlp_output.n_bytes(), stream));
			dL_dmlp_output.memset_async(stream, 0);
			tcnn::linear_kernel(fill_positions_view<T,float>, 0, stream,
				batch_size*3, 3, dL_doutput.view(), dL_dmlp_output.view()
			);

			tcnn::GPUMatrixDynamic<T> dL_dmlp_network_input{m_mlp_network_input_width, batch_size, stream, m_pos_encoding->preferred_output_layout()};

			// const tcnn::GPUMatrixDynamic<float> mlp_network_output{output.data(), m_mlp_network->padded_output_width(), batch_size, output.layout()};
			// m_mlp_network->backward(stream, *forward.mlp_network_ctx, forward.mlp_network_input, mlp_network_output, dL_drgb, &dL_drgb_network_input, use_inference_params, param_gradients_mode);
			// tcnn::linear_kernel(debug_log<T>, 0, stream, batch_size, dL_dmlp_output.view());

			m_mlp_network->backward(stream, *forward.mlp_network_ctx, forward.mlp_network_input, forward.mlp_network_output, dL_dmlp_output, &dL_dmlp_network_input, use_inference_params, param_gradients_mode);

			tcnn::GPUMatrixDynamic<float> dL_dpos_encoding_input;
			if (dL_dinput){
				dL_dpos_encoding_input = dL_dinput->slice_rows(0, m_pos_encoding->input_width());
			}

			m_pos_encoding->backward(
				stream,
				*forward.pos_encoding_ctx,
				input.slice_rows(0, m_pos_encoding->input_width()),
				forward.mlp_network_input,
				dL_dmlp_network_input,
				dL_dinput ? &dL_dpos_encoding_input : nullptr,
				use_inference_params,
				param_gradients_mode
			);



		}


		void set_params(T* params, T* inference_params, T* backward_params, T* gradients)  {
			size_t offset = 0;

			m_mlp_network->set_params(
				params + offset,
				inference_params + offset,
				backward_params + offset,
				gradients + offset
			);
			offset += m_mlp_network->n_params();

			#if residual_encoding
				m_pos_encoding->set_params(
					params + offset,
					inference_params + offset,
					backward_params + offset,
					gradients + offset
				);
				offset += m_pos_encoding->n_params();
			#endif


		}

		void initialize_params(tcnn::pcg32& rnd, float* params_full_precision, T* params, T* inference_params, T* backward_params, T* gradients, float scale = 1)  {
			size_t offset = 0;
			printf("residual network initialize model!\n");

			// uint32_t n_params = m_mlp_network->n_params();
			// tcnn::GPUMemory<char> m_accumulation_params_buffer;

			// m_accumulation_params_buffer.resize(sizeof(T) * n_params * 3 + sizeof(float) * n_params * 1);
			// m_accumulation_params_buffer.memset(0);

			// params_full_precision = (float*)(m_accumulation_params_buffer.data());
			// params                = (T*)(m_accumulation_params_buffer.data() + sizeof(float) * n_params);
			// backward_params       = (T*)(m_accumulation_params_buffer.data() + sizeof(float) * n_params + sizeof(T) * n_params);
			// gradients       = (T*)(m_accumulation_params_buffer.data() + sizeof(float) * n_params + sizeof(T) * n_params * 2);


			
			// m_mlp_network->initialize_params(
			// 	rnd,
			// 	params_full_precision + offset,
			// 	params + offset,
			// 	params + offset,
			// 	backward_params + offset,
			// 	gradients + offset,
			// 	scale
			// );


			m_mlp_network->initialize_params(
				rnd,
				params_full_precision + offset,
				params + offset,
				inference_params + offset,
				backward_params + offset,
				gradients + offset,
				scale
			);

			tcnn::pcg32 m_rng{1337};
			printf(" m_mlp_network->n_params():%d\n", m_mlp_network->n_params());
			tcnn::generate_random_uniform<float>(m_rng, m_mlp_network->n_params(), params_full_precision + offset + m_mlp_network->n_params()- m_last_layer_params_num, 0.000f, 0.000f);

			offset += m_mlp_network->n_params();
			#if residual_encoding
				m_pos_encoding->initialize_params(
					rnd,
					params_full_precision + offset,
					params + offset,
					inference_params + offset,
					backward_params + offset,
					gradients + offset,
					scale
				);

				// tcnn::generate_random_uniform<float>(m_rng, m_pos_encoding->n_params(), params_full_precision + offset, 0.000f, 0.000f);

				offset += m_pos_encoding->n_params();
			#endif 


		}

		size_t n_params() const  {
			#if residual_encoding
				return m_pos_encoding->n_params() + m_mlp_network->n_params();
			#else	
				return m_mlp_network->n_params();
			#endif
		}

		uint32_t padded_output_width() const  {
			// return std::max(m_mlp_network->padded_output_width(), (uint32_t)4);
			return 3;
			// return (uint32_t)16;
		}

		uint32_t input_width() const  {
			return m_n_pos_dims;
		}

		uint32_t output_width() const  {
			return 3;
			// return 7; 
		}

		uint32_t n_extra_dims() const {
			return 0u;
		}

		uint32_t required_input_alignment() const  {
			return 1; // No alignment required due to encoding
		}

		std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const  {
			auto layers = m_mlp_network->layer_sizes();
			return layers;
		}

		uint32_t width(uint32_t layer) const  {
			if (layer == 0) {
				return m_pos_encoding->padded_output_width();
			} 
			else if (layer < m_mlp_network->num_forward_activations() + 1) {
				return m_mlp_network->width(layer - 1);
			}
		}

		uint32_t num_forward_activations() const  {
			return m_mlp_network->num_forward_activations();
		}

		std::pair<const T*, tcnn::MatrixLayout> forward_activations(const tcnn::Context& ctx, uint32_t layer) const  {
			printf("forward_activations not implemented in DeltaNetwork!\n");
			exit(1);
			// const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
			// if (layer == 0) {
			// 	return {forward.density_network_input.data(), m_pos_encoding->preferred_output_layout()};
			// } else if (layer < m_density_network->num_forward_activations() + 1) {
			// 	return m_density_network->forward_activations(*forward.density_network_ctx, layer - 1);
			// } else if (layer == m_density_network->num_forward_activations() + 1) {
			// 	return {forward.rgb_network_input.data(), m_dir_encoding->preferred_output_layout()};
			// } else {
			// 	return m_rgb_network->forward_activations(*forward.rgb_network_ctx, layer - 2 - m_density_network->num_forward_activations());
			// }
		}

		tcnn::json hyperparams() const  {
			tcnn::json mlp_network_hyperparams = m_mlp_network->hyperparams();
			mlp_network_hyperparams["n_output_dims"] = m_mlp_network->padded_output_width();
			return {
				{"otype", "DeltaNetwork"},
				{"pos_encoding", m_pos_encoding->hyperparams()},
				{"mlp_network", mlp_network_hyperparams},
			};
		}

		const std::shared_ptr<tcnn::Encoding<T>>& encoding() const {
			return m_pos_encoding;
		}

	private:
		std::unique_ptr<tcnn::Network<T>> m_mlp_network;
		std::shared_ptr<tcnn::Encoding<T>> m_pos_encoding;

		uint32_t m_mlp_network_input_width;
		uint32_t m_n_pos_dims;
		uint32_t m_last_layer_params_num;

		struct ForwardContext : public tcnn::Context {
			tcnn::GPUMatrixDynamic<T> mlp_network_input;
			tcnn::GPUMatrixDynamic<T> mlp_network_output;

			std::unique_ptr<Context> pos_encoding_ctx;

			std::unique_ptr<Context> mlp_network_ctx;
		};

};

NGP_NAMESPACE_END
