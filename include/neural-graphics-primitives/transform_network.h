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
class DeltaNetwork: public tcnn::DifferentiableObject<float, T, float> {
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
			forward_impl(stream, input, &output, use_inference_params, false);
			return; // Since we need gradient in inference. Should it be replaced by forward_impl?
		}


		std::unique_ptr<tcnn::Context> forward_impl(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<float>* output = nullptr, bool use_inference_params = false, bool prepare_input_gradients = false)  {
			uint32_t batch_size = input.n();


			auto forward = std::make_unique<ForwardContext>();
			#if rotation_reprensentation
				tcnn::linear_kernel(add_global_movement_with_rotation_6d<float, T>, 0, stream, batch_size,
							m_rotation->params(), m_transition->params(), first_frame_offset, input.view(), output->view());
			#else
				printf("some error not implemented! 1\n");
				tcnn::linear_kernel(add_global_movement_with_rotation_quaternion<float, T>, 0, stream, batch_size,
							m_rotation->params(), m_transition->params(), input.view(), output->view());
			#endif

			return forward;

		}


		void backward_impl(
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

		std::pair<const T*, tcnn::MatrixLayout> forward_activations(const tcnn::Context& ctx, uint32_t layer) const  {
			printf("forward_activations not implemented in DeltaNetwork!\n");
			exit(1);
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


NGP_NAMESPACE_END
