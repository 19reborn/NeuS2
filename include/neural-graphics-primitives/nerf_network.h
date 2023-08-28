/** @file   nerf_network.h
 *  @author Yiming Wang <w752531540@gmail.com>
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

#define NERF_DEBUG_BACKWARD 0
#define GEOMETRY_INIT 1
#define global_delta 1
#define viewdir_backward 0


using namespace Eigen;

template <typename T>
class NerfNetwork : public tcnn::Network<float, T> {
public:
	using json = nlohmann::json;

	NerfNetwork(uint32_t n_pos_dims, uint32_t n_dir_dims, uint32_t n_extra_dims, uint32_t dir_offset, const json& pos_encoding, const json& dir_encoding, const json& density_network, const json& rgb_network) : m_n_pos_dims{n_pos_dims}, m_n_dir_dims{n_dir_dims}, m_dir_offset{dir_offset}, m_n_extra_dims{n_extra_dims} {
		uint32_t rgb_alignment = tcnn::minimum_alignment(rgb_network);
		m_dir_encoding.reset(tcnn::create_encoding<T>(m_n_dir_dims + m_n_extra_dims, dir_encoding, rgb_alignment));

		json local_density_network_config = density_network;
		#if GEOMETRY_INIT
			m_pos_encoding.reset(tcnn::create_encoding<T>(n_pos_dims, pos_encoding, density_network.contains("otype") && (tcnn::equals_case_insensitive(density_network["otype"], "FullyFusedMLP") || tcnn::equals_case_insensitive(density_network["otype"], "MegakernelMLP")) ? 2u : 2u));
			m_density_network_input_width = tcnn::next_multiple(m_n_pos_dims + m_pos_encoding->output_width(), rgb_alignment); // 40
		#else
			m_pos_encoding.reset(tcnn::create_encoding<T>(n_pos_dims, pos_encoding, density_network.contains("otype") && (tcnn::equals_case_insensitive(density_network["otype"], "FullyFusedMLP") || tcnn::equals_case_insensitive(density_network["otype"], "MegakernelMLP")) ? 16u : 8u));
			m_density_network_input_width = m_pos_encoding->padded_output_width();
		#endif
		printf("m_density_network_input_width: %d", m_density_network_input_width);
		local_density_network_config["n_input_dims"] = m_density_network_input_width;
		if (!density_network.contains("n_output_dims")) {
			local_density_network_config["n_output_dims"] = 16;
		}
		m_density_network.reset(tcnn::create_network<T>(local_density_network_config));

		// density(feature), xyz, normal, dir
		m_rgb_network_input_width = tcnn::next_multiple(m_n_pos_dims + m_n_pos_dims + m_dir_encoding->padded_output_width() + m_density_network->padded_output_width(), rgb_alignment);

		json local_rgb_network_config = rgb_network;
		local_rgb_network_config["n_input_dims"] = m_rgb_network_input_width;
		local_rgb_network_config["n_output_dims"] = 16;
		m_rgb_network.reset(tcnn::create_network<T>(local_rgb_network_config));


		m_delta_network = std::make_shared<DeltaNetwork<T>>(m_pos_encoding->input_width() + m_dir_encoding->input_width());

		m_variance_network = std::make_shared<TrainableBuffer<1, 1, T>>(Eigen::Matrix<int, 1, 1>{(int)4});

		m_variance = 0.3f;
		m_training_step = 0;
		m_sdf_bias =(T)(local_density_network_config.value("sdf_bias", -0.1f)); 

		accumulated_transition = std::make_shared<TrainableBuffer<1, 1, T>>(Eigen::Matrix<int, 1, 1>{(int)4});
		#if rotation_reprensentation
			accumulated_rotation = std::make_shared<TrainableBuffer<1, 1, T>>(Eigen::Matrix<int, 1, 1>{(int)12});
		#else
			accumulated_rotation = std::make_shared<TrainableBuffer<1, 1, T>>(Eigen::Matrix<int, 1, 1>{(int)4});
		#endif
		init_accumulation_movement();
	}

	virtual ~NerfNetwork() { }

	void inference_mixed_precision_impl(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>& output, bool use_inference_params = true) override {

		forward_impl(stream, input, &output, use_inference_params, false);
		return; // Since we need gradient in inference. Should it be replaced by forward_impl?


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
		
		forward->density_network_input = tcnn::GPUMatrixDynamic<T>{m_density_network_input_width, batch_size, stream, m_pos_encoding->preferred_output_layout()};
		forward->density_network_input.memset_async(stream, 0);
		forward->rgb_network_input = tcnn::GPUMatrixDynamic<T>{m_rgb_network_input_width, batch_size, stream, m_dir_encoding->preferred_output_layout()};
		forward->rgb_network_input.memset_async(stream, 0);

		// deform: used for global transformation
		tcnn::GPUMatrixDynamic<float> deformed_xyz{ m_pos_encoding->input_width() , batch_size, stream, input.layout() };
		tcnn::GPUMatrixDynamic<float> deformed_viewdir{ m_dir_encoding->input_width() , batch_size, stream, input.layout() };

		// xyz + delta
		forward->delta_network_output = tcnn::GPUMatrixDynamic<float> { m_delta_network->padded_output_width(), batch_size, stream, input.layout() };

		if (m_use_delta){

			forward->delta_network_input =  tcnn::GPUMatrixDynamic<float> {m_delta_network->input_width(), batch_size, stream, input.layout()};
			tcnn::linear_kernel(fill_positions_view<float,float>, 0, stream, batch_size * m_pos_encoding->input_width(), m_pos_encoding->input_width(),
						input.view(), forward->delta_network_input.view());	

			tcnn::linear_kernel(fill_positions_view<float,float>, 0, stream, batch_size * m_dir_encoding->input_width(), m_dir_encoding->input_width(),
						get_advance(input.view(), m_dir_offset, 0), get_advance(forward->delta_network_input.view(), m_pos_encoding->input_width(), 0));	

			forward->delta_network_ctx = m_delta_network->forward(stream,
					forward->delta_network_input,
					&forward->delta_network_output,
					use_inference_params,
					prepare_input_gradients);

			tcnn::linear_kernel(fill_positions_view<float,float>, 0, stream, batch_size * m_pos_encoding->input_width(), m_pos_encoding->input_width(),
						forward->delta_network_output.view(), deformed_xyz.view());
			tcnn::linear_kernel(fill_positions_view<float,float>, 0, stream, batch_size * m_dir_encoding->input_width(), m_dir_encoding->input_width(),
						get_advance(forward->delta_network_output.view(), m_pos_encoding->input_width(), 0), deformed_viewdir.view());
			
		}
		else {
			deformed_xyz = input.slice_rows(0, m_pos_encoding->input_width());
			
			tcnn::linear_kernel(fill_positions_view<float,float>, 0, stream, batch_size * m_dir_encoding->input_width(), m_dir_encoding->input_width(),
						get_advance(input.view(), m_dir_offset, 0), deformed_viewdir.view());

			tcnn::linear_kernel(fill_positions_view<float,float>, 0, stream, batch_size * m_pos_encoding->input_width(), m_pos_encoding->input_width(),
						deformed_xyz.view(), forward->delta_network_output.view());
			tcnn::linear_kernel(fill_positions_view<float,float>, 0, stream, batch_size * m_dir_encoding->input_width(), m_dir_encoding->input_width(),
						deformed_viewdir.view(), get_advance(forward->delta_network_output.view(), m_pos_encoding->input_width(), 0));
		}

		#if GEOMETRY_INIT
			tcnn::GPUMatrixDynamic<T> encoded_xyz{ m_pos_encoding->padded_output_width(), batch_size, stream, forward->density_network_input.layout() };
			forward->pos_encoding_ctx = m_pos_encoding->forward(
				stream,
				deformed_xyz.slice_rows(0, m_pos_encoding->input_width()),
				&encoded_xyz,
				use_inference_params,
				true
			);

			// xyz + encoding
			tcnn::linear_kernel(fill_positions_view_with_fixed_offset<T, float>, 0, stream,
				batch_size*m_pos_encoding->input_width(), m_pos_encoding->input_width(),
				deformed_xyz.slice_rows(0, m_pos_encoding->input_width()).view(), forward->density_network_input.view());
		
			tcnn::linear_kernel(fill_positions_view<T, T>, 0, stream,
				batch_size*m_pos_encoding->padded_output_width(), m_pos_encoding->padded_output_width(),
				encoded_xyz.view(), get_advance(forward->density_network_input.view(),m_pos_encoding->input_width(), 0));

		#else
			forward->pos_encoding_ctx = m_pos_encoding->forward(
				stream,
				deformed_xyz.slice_rows(0, m_pos_encoding->input_width()),
				&forward->density_network_input,
				use_inference_params,
				true
			);
		#endif

		forward->density_network_output = forward->rgb_network_input.slice_rows(0, m_density_network->padded_output_width());
		forward->density_network_ctx = m_density_network->forward(stream, forward->density_network_input, &forward->density_network_output, use_inference_params, true);
		// end density network forward

		tcnn::GPUMatrixDynamic<T> dSDF_dSDF{ m_density_network->padded_output_width(), batch_size, stream, forward->density_network_output.layout() };
		tcnn::GPUMatrixDynamic<float> dSDF_dDeformedPos{ m_pos_encoding->input_width(), batch_size, stream, forward->density_network_output.layout() };
		dSDF_dSDF.memset_async(stream, 0);
		dSDF_dDeformedPos.memset_async(stream, 0);

		tcnn::linear_kernel(set_constant_value_view<T>, 0, stream,
			batch_size, 1.0f, dSDF_dSDF.view());

		tcnn::GPUMatrixDynamic<T> dSDF_dPosEncoding{ m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout() };

		forward->dSDF_dPos = tcnn::GPUMatrixDynamic<float>{m_pos_encoding->input_width(), batch_size, stream, input.layout() }; // = gradient.

		#if GEOMETRY_INIT
			tcnn::GPUMatrixDynamic<T> dSDF_dSDFInput{ m_density_network_input_width, batch_size, stream, m_pos_encoding->preferred_output_layout() };
			m_density_network->backward(stream, *forward->density_network_ctx, forward->density_network_input, forward->density_network_output, dSDF_dSDF, &dSDF_dSDFInput, use_inference_params, tcnn::EGradientMode::Ignore);
			tcnn::linear_kernel(fill_positions_view<T, T>, 0, stream,
				batch_size*m_pos_encoding->padded_output_width(), m_pos_encoding->padded_output_width(),
				get_advance(dSDF_dSDFInput.view(), m_pos_encoding->input_width(), 0) , dSDF_dPosEncoding.view());

			m_pos_encoding->backward(stream, *forward->pos_encoding_ctx, deformed_xyz.slice_rows(0, m_pos_encoding->input_width()), encoded_xyz, dSDF_dPosEncoding, &dSDF_dDeformedPos, use_inference_params, tcnn::EGradientMode::Ignore);

			tcnn::linear_kernel(add_positions_view<float, T>, 0, stream,
				batch_size*m_pos_encoding->input_width(), m_pos_encoding->input_width(),
				dSDF_dSDFInput.view() , dSDF_dDeformedPos.view());

			forward->dSDF_dPos = dSDF_dDeformedPos.slice_rows(0, m_pos_encoding->input_width());
		#else
			m_density_network->backward(stream, *forward->density_network_ctx, forward->density_network_input, forward->density_network_output, dSDF_dSDF, &dSDF_dPosEncoding, use_inference_params, tcnn::EGradientMode::Ignore);
			m_pos_encoding->backward(stream, *forward->pos_encoding_ctx, deformed_xyz.slice_rows(0, m_pos_encoding->input_width()), forward->density_network_input, dSDF_dPosEncoding, &dSDF_dDeformedPos, use_inference_params, tcnn::EGradientMode::Ignore);

			forward->dSDF_dPos = dSDF_dDeformedPos.slice_rows(0, m_pos_encoding->input_width());
		#endif
		// end density network backward

		auto dir_out = forward->rgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width());

		forward->dir_encoding_ctx = m_dir_encoding->forward(
			stream,
			deformed_viewdir,
			&dir_out,
			use_inference_params,
			prepare_input_gradients
		);

		// fill xyz into rgb_network_input
		tcnn::linear_kernel(fill_positions_view<T, float>, 0, stream,
			batch_size*m_pos_encoding->input_width(), m_pos_encoding->input_width(),
			deformed_xyz.view(), get_advance(forward->rgb_network_input.view(), m_density_network->padded_output_width() + m_dir_encoding->padded_output_width(), 0));

		// fill d(sdf)_d(xyz) into rgb_network_input
		tcnn::linear_kernel(fill_positions_view<T, float>, 0, stream,
			batch_size*m_pos_encoding->input_width(), m_pos_encoding->input_width(),
			forward->dSDF_dPos.view(), get_advance(forward->rgb_network_input.view(), m_density_network->padded_output_width() + m_dir_encoding->padded_output_width() + m_pos_encoding->input_width(), 0));

	
		tcnn::GPUMatrixDynamic<T> rgb_network_output = tcnn::GPUMatrixDynamic<T>{output->data(), m_rgb_network->padded_output_width(), batch_size, output->layout()};
		forward->rgb_network_ctx = m_rgb_network->forward(stream, forward->rgb_network_input, output ? &rgb_network_output : nullptr, use_inference_params, prepare_input_gradients);
		// end rgb network forward

		if (output) {
			// geo init with sdf bias
			#if GEOMETRY_INIT
				tcnn::linear_kernel(extract_sdf_value_view_with_bias<T>, 0, stream,
					batch_size,
					forward->density_network_output.view(),
					m_sdf_bias,
					get_advance(output->view(), m_pos_encoding->input_width(), 0)
				);
			#else
				tcnn::linear_kernel(extract_sdf_value_view<T>, 0, stream,
					batch_size,
					forward->density_network_output.view(),
					get_advance(output->view(), m_pos_encoding->input_width(), 0)
				);
			#endif

	

			tcnn::linear_kernel(extract_dSDF_dPos_view<T, float>, 0, stream,
				batch_size*3,
				forward->dSDF_dPos.view(),
				get_advance(output->view(), 1 + m_pos_encoding->input_width(), 0)
			);

			tcnn::linear_kernel(extract_single_variance_view<T, T>, 0, stream,
				batch_size,
				m_variance_network->params(),
				output->view()
			);

			// extract diffinite viewdir
			tcnn::linear_kernel(fill_positions_view<T, float>, 0, stream,
				batch_size * m_dir_encoding->input_width(),
				m_dir_encoding->input_width(),
				deformed_viewdir.view(),
				get_advance(output->view(), 8, 0)
			);
		}
		
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
	
	
		// Backprop through dir encoding if it is trainable or if we need input gradients

		#if viewdir_backward
			tcnn::GPUMatrixDynamic<float> dL_ddir_encoding_input;
			if (m_dir_encoding->n_params() > 0 || dL_dinput || m_use_delta && m_train_delta) {
				printf("some not implemented happend!\n");
				tcnn::GPUMatrixDynamic<T> dL_ddir_encoding_output = dL_drgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width());
				dL_ddir_encoding_input = tcnn::GPUMatrixDynamic<float>{m_dir_encoding->input_width(), batch_size, stream, input.layout()};

				m_dir_encoding->backward(
					stream,
					*forward.dir_encoding_ctx,
					forward.delta_network_output.slice_rows(m_pos_encoding->input_width(), m_dir_encoding->input_width()),
					forward.rgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width()),
					dL_ddir_encoding_output,
					dL_ddir_encoding_input,
					use_inference_params,
					param_gradients_mode
				);
			}

		#else

			if (m_dir_encoding->n_params() > 0 || dL_dinput) {
				printf("some not implemented happend!\n");
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

		#endif

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
			dL_ddensity_network_input = tcnn::GPUMatrixDynamic<T>{m_density_network_input_width, batch_size, stream, m_pos_encoding->preferred_output_layout()};
		}

		m_density_network->backward(stream, *forward.density_network_ctx, forward.density_network_input, forward.density_network_output, dL_ddensity_network_output, dL_ddensity_network_input.data() ? &dL_ddensity_network_input : nullptr, use_inference_params, param_gradients_mode);


		// Backprop through pos encoding if it is trainable or if we need input gradients
		tcnn::GPUMatrixDynamic<float> dL_dpos_encoding_input{m_pos_encoding->input_width(), batch_size, stream, input.layout()};
		dL_dpos_encoding_input.memset_async(stream, 0);

		#if GEOMETRY_INIT	
			tcnn::GPUMatrixDynamic<T> dL_dposencoding_output{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};

			tcnn::linear_kernel(fill_positions_view<T, T>, 0, stream,
				batch_size*m_pos_encoding->padded_output_width(),
				m_pos_encoding->padded_output_width(),
				get_advance(dL_ddensity_network_input.view(), m_pos_encoding->input_width(), 0),//
				dL_dposencoding_output.view()
			);

			m_pos_encoding->backward(
				stream,
				*forward.pos_encoding_ctx,
				forward.delta_network_output.slice_rows(0, m_pos_encoding->input_width()),
				forward.density_network_input.slice_rows(m_pos_encoding->input_width(), m_pos_encoding->padded_output_width()), // To do:: need check
				dL_dposencoding_output,
				// dL_dinput ? &dL_dpos_encoding_input : nullptr,
				&dL_dpos_encoding_input,
				use_inference_params,
				param_gradients_mode
			);

		#else
			m_pos_encoding->backward(
				stream,
				*forward.pos_encoding_ctx,
				forward.delta_network_output.slice_rows(0, m_pos_encoding->input_width()),
				forward.density_network_input,
				dL_ddensity_network_input,
				// dL_dinput ? &dL_dpos_encoding_input : nullptr,
				&dL_dpos_encoding_input,
				use_inference_params,
				param_gradients_mode
			);

		#endif

		if (m_train_canonical){
			// update variance
			{
				// directly reduce_sum
				T gradients = T(0.0f);

				tcnn::GPUMatrixDynamic<T> dL_dvariance{1, batch_size, stream};

				tcnn::linear_kernel(add_variance_view_to_loss<T>, 0, stream,
					batch_size,
					dL_doutput.view(),
					dL_dvariance.view()
				);
				gradients = tcnn::reduce_sum(dL_dvariance.data(),batch_size,stream);
				cudaMemcpyAsync(m_variance_network->gradients(), &gradients, sizeof(T), cudaMemcpyHostToDevice, stream);
			}

			// fill in dloss_d(normal)
			{
				tcnn::GPUMatrixDynamic<float> dL_dsdf_dinput{ m_pos_encoding->input_width(), batch_size};
				dL_dsdf_dinput.memset_async(stream, 0);

				// d(rgb_output)_d(normal)
				tcnn::linear_kernel(fill_positions_view<float, T>, 0, stream,
					batch_size*m_pos_encoding->input_width(),
					m_pos_encoding->input_width(),
					get_advance(dL_drgb_network_input.view(), m_density_network->padded_output_width() + m_dir_encoding->padded_output_width() + m_pos_encoding->input_width(), 0),//
					dL_dsdf_dinput.view()
				);
		
				// d(ek_loss)_d(normal)
				tcnn::linear_kernel(add_positions_view_ekloss<float, T>, 0, stream,
					batch_size*m_pos_encoding->input_width(),
					m_pos_encoding->input_width(),
					indeed_batch_size, // ek_loss's backsize (N) is #samples, instead of #rays
					get_advance(dL_doutput.view(), 4, 0),//
					dL_dsdf_dinput.view()
				);

				// d(rgb)_d(true_cos) * d(true_cos)_d(normal)
				tcnn::linear_kernel(add_positions_view<float, T>, 0, stream,
					batch_size*m_pos_encoding->input_width(),
					m_pos_encoding->input_width(),
					get_advance(dL_doutput.view(), 8, 0),//
					dL_dsdf_dinput.view()
				);
			// }

			// double backward
			// {
				// d(mlp_output)_d(encoding)
				tcnn::GPUMatrixDynamic<T> dL_denc_output{ m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
				// d(mlp_output)_d(mlp_output)
				tcnn::GPUMatrixDynamic<T> dL_dmlp_output{ m_density_network->padded_output_width(), batch_size, stream};

				dL_denc_output.memset_async(stream, 0);
				dL_dmlp_output.memset_async(stream, 0);
				tcnn::linear_kernel(set_constant_value_view<T>, 0, stream,
					batch_size, 1.0f, dL_dmlp_output.view());

				#if GEOMETRY_INIT
					tcnn::GPUMatrixDynamic<T> dL_ddensity_input{ m_density_network_input_width, batch_size, stream, m_pos_encoding->preferred_output_layout()};

					m_density_network->backward(stream, *forward.density_network_ctx, forward.density_network_input, 
							forward.density_network_output, 
							dL_dmlp_output, 
							&dL_ddensity_input, use_inference_params, tcnn::EGradientMode::Ignore);
				

					tcnn::linear_kernel(fill_positions_view<T, T>, 0, stream,
							batch_size*m_pos_encoding->padded_output_width(),
							m_pos_encoding->padded_output_width(),
							get_advance(dL_ddensity_input.view(), m_pos_encoding->input_width(), 0),//
							dL_denc_output.view()
						);
				#else
					m_density_network->backward(stream, *forward.density_network_ctx, forward.density_network_input, 
							forward.density_network_output, 
							dL_dmlp_output, 
							&dL_denc_output, use_inference_params, tcnn::EGradientMode::Ignore);
				
				#endif

				tcnn::GPUMatrixDynamic<float> dL_dsdf_dinput_d_input{m_pos_encoding->input_width(), batch_size, stream};
				dL_dsdf_dinput_d_input.memset_async(stream, 0);

				tcnn::GPUMatrixDynamic<T> pos_encoding_dy{m_pos_encoding->padded_output_width(), batch_size, stream};
				pos_encoding_dy.memset_async(stream, 0);
				m_pos_encoding->backward_backward_input(
					stream, 
					*forward.pos_encoding_ctx,
					forward.delta_network_output.slice_rows(0, m_pos_encoding->input_width()),
					dL_dsdf_dinput, // dL_d(d(mlp_output)_dx)
					dL_denc_output, // d(mlp_output)_d(encoding)
					&pos_encoding_dy, // dl_d(dy'_dy)
					&dL_dsdf_dinput_d_input,
					use_inference_params, 
					tcnn::EGradientMode::Accumulate
				);

				#if GEOMETRY_INIT
					tcnn::GPUMatrixDynamic<T> d_density_input{m_density_network_input_width, batch_size, stream};
					d_density_input.memset_async(stream, 0);

					tcnn::linear_kernel(fill_positions_view<T, T>, 0, stream,
						batch_size*m_pos_encoding->padded_output_width(), m_pos_encoding->padded_output_width(),
						pos_encoding_dy.view(), get_advance(d_density_input.view(), m_pos_encoding->input_width(), 0));

					// dIdentity = (1.0,1.0,1.0) * dL_dsdf_dinput
					tcnn::linear_kernel(fill_positions_view<T, float>, 0, stream,
						batch_size*m_pos_encoding->input_width(), m_pos_encoding->input_width(),
						dL_dsdf_dinput.view(), d_density_input.view());
					

					m_density_network->backward_backward_input(
						stream, 
						*forward.density_network_ctx,
						forward.density_network_input, 
						d_density_input,  // dl_d(dy'_dx)
						dL_dmlp_output, // d(mlp_output)_d(mlp_output) -> dy'_dy
						nullptr,
						nullptr,
						use_inference_params, 
						//  param_gradients_mode
						tcnn::EGradientMode::Accumulate
					);

				#else
					m_density_network->backward_backward_input(
						stream, 
						*forward.density_network_ctx,
						forward.density_network_input, 
						pos_encoding_dy,
						dL_dmlp_output,
						nullptr, // assume dl_ddensity_dx_dx === 0
						nullptr,
						use_inference_params, 
						tcnn::EGradientMode::Accumulate
					);

				#endif
			}
		} // backward
		// chain rule loss contain two loss (1. first order loss: dL_ddelta_network_output 2. second order loss: dl_(dsdf_dx))
				
		if (m_train_delta && m_use_delta){
			tcnn::GPUMatrixDynamic<float> dL_ddelta_network_output{m_delta_network->padded_output_width(), batch_size, stream,  dL_dpos_encoding_input.layout()};
			dL_ddelta_network_output.memset_async(stream, 0);

			// add dl_dxyz in pos_encoding_input
			tcnn::linear_kernel(fill_positions_view<float, float>, 0, stream, batch_size * 3u, 3u,
					dL_dpos_encoding_input.view(),
					dL_ddelta_network_output.view());

			// add dl_dxyz in rgb_network_input [xyz] in [feature_vector + encoded_viewdir + xyz + normals]
			tcnn::linear_kernel(add_positions_view<float, T>, 0, stream, batch_size * 3u, 3u,
					get_advance(dL_drgb_network_input.view(), m_density_network->padded_output_width() + m_dir_encoding->padded_output_width(), 0),
					dL_ddelta_network_output.view());

			#if viewdir_backward
			// TODO:: add viewdir backward
				tcnn::linear_kernel(fill_positions_view<float, float>, 0, stream, batch_size * m_dir_encoding->input_width(), m_dir_encoding->input_width(),
						dL_ddir_encoding_input.view(),
						get_advance(dL_ddelta_network_output.view(), m_pos_encoding->input_width(), 0));
			#endif 

			#if GEOMETRY_INIT
				// add dl_dxyz in density_network_input [xyz] of [xyz + pos_encoding]
				tcnn::linear_kernel(add_positions_view<float, T>, 0, stream, batch_size * 3u, 3u,
						dL_ddensity_network_input.view(), dL_ddelta_network_output.view());
				// add l1_regulization to delta_output
				// TODO
			#endif

			tcnn::GPUMatrixDynamic<float> dL_ddelta_network_input;
			if (dL_dinput) {
				printf("something not implemented!\n");
				exit(1);
				// dL_ddelta_network_input = dL_dinput->slice_rows(0, m_pos_encoding->input_width());
			}

			m_delta_network->backward(
				stream,
				*forward.delta_network_ctx,
				// input.slice_rows(0, m_pos_encoding->input_width()),
				forward.delta_network_input,
				forward.delta_network_output,
				dL_ddelta_network_output,
				dL_dinput ? &dL_ddelta_network_input : nullptr,
				use_inference_params,
				param_gradients_mode
			);
		}


	}

	void sdf(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>& output, bool use_inference_params = true) {
		if (input.layout() != tcnn::CM) {
			throw std::runtime_error("NerfNetwork::density input must be in column major format.");
		}

		uint32_t batch_size = output.n();
		
		tcnn::GPUMatrixDynamic<float> deformed_xyz;
		if (m_use_delta){
			deformed_xyz = tcnn::GPUMatrixDynamic<float>{ m_delta_network->padded_output_width(), batch_size, stream, input.layout()};
			tcnn::GPUMatrixDynamic<float> delta_network_input {m_delta_network->input_width(), batch_size, stream, input.layout()};
			delta_network_input.memset_async(stream, 0);
			tcnn::linear_kernel(fill_positions_view<float,float>, 0, stream, batch_size * 3u, 3u, input.view(), delta_network_input.view());

			m_delta_network->inference_mixed_precision(stream, // TODO:: accelerate inference w/o cal dy_dx
					delta_network_input,
					deformed_xyz,
					use_inference_params);
		}
		else {
			deformed_xyz = tcnn::GPUMatrixDynamic<float>{ m_pos_encoding->input_width(), batch_size, stream, input.layout()};
			deformed_xyz = input.slice_rows(0, m_pos_encoding->input_width());
		}

		#if GEOMETRY_INIT
			tcnn::GPUMatrixDynamic<T> density_network_input{m_density_network_input_width, batch_size, stream, m_pos_encoding->preferred_output_layout()};
			density_network_input.memset_async(stream, 0);
			tcnn::GPUMatrixDynamic<T> encoded_xyz{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
			m_pos_encoding->inference_mixed_precision(
				stream,
				deformed_xyz.slice_rows(0, m_pos_encoding->input_width()),
				encoded_xyz,
				use_inference_params
			);

			tcnn::linear_kernel(fill_positions_view_with_fixed_offset<T, float>, 0, stream,
				batch_size*m_pos_encoding->input_width(), m_pos_encoding->input_width(),
				deformed_xyz.slice_rows(0, m_pos_encoding->input_width()).view(), density_network_input.view());

			tcnn::linear_kernel(fill_positions_view<T, T>, 0, stream,
				batch_size*m_pos_encoding->padded_output_width(), m_pos_encoding->padded_output_width(),
				encoded_xyz.view(), get_advance(density_network_input.view(),m_pos_encoding->input_width(), 0));

			m_density_network->inference_mixed_precision(stream, density_network_input, output, use_inference_params);

			#if GEOMETRY_INIT
				tcnn::linear_kernel(sdf_add_bias<T>, 0, stream,
					batch_size,
					m_sdf_bias,
					output.view()
				);
			#endif

		#else
			tcnn::GPUMatrixDynamic<T> density_network_input{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
			m_pos_encoding->inference_mixed_precision(
				stream,
				deformed_xyz.slice_rows(0, m_pos_encoding->input_width()),
				density_network_input,
				use_inference_params
			);

			m_density_network->inference_mixed_precision(stream, density_network_input, output, use_inference_params);


		#endif
	}

	void density(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>& output, bool use_inference_params = true) {
		if (input.layout() != tcnn::CM) {
			throw std::runtime_error("NerfNetwork::density input must be in column major format.");
		}

		uint32_t batch_size = output.n();

		sdf(stream,input,output,use_inference_params);
	
		tcnn::linear_kernel(sdf_to_density_variance_buffer<T>, 0, stream,
			batch_size,
			m_variance_network->params(),
			output.view()
		);

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


		m_variance_network->set_params(
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset
		);

		offset += m_variance_network->n_params();

	}

	std::vector<float> load_sdf_mlp_weight(uint32_t n_elements){
		
		std::vector<float> data(n_elements);
		std::FILE* fp;
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

		printf("network_params_elements: %d\n", n_elements);
		if (!fp) {
			printf("[ERROR] Load SDF MLP weight failed!\n");
			printf("[ERROR] Please run in the base directory of NeuS2 so that the `utils/mlp_weights.txt` can be found!\n");
			exit(1);
		}
		else {
			uint32_t i;
			for (i = 0; i < n_elements; i++) {
				int readlen = fscanf(fp, "%f", &data[i]);
			}
			fclose(fp);
		}
		return data;
	}

	void initialize_params(tcnn::pcg32& rnd, float* params_full_precision, T* params, T* inference_params, T* backward_params, T* gradients, float scale = 1) override {
		size_t offset = 0;
		printf("initialize NeuS Network!\n");

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
		#endif

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

		m_variance_network->initialize_params(
			rnd,
			params_full_precision + offset,
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset,
			scale
		);
		int variance_n_params = m_variance_network->n_params();

		tcnn::pcg32 m_rng{1337};
		tcnn::generate_random_uniform<float>(m_rng, variance_n_params, params_full_precision + offset, 0.300f, 0.300f);

		offset += m_variance_network->n_params();
		printf("m_variance_network_n_params: %lu\n",m_variance_network->n_params()); // 1
	}

	void initialize_sdf_mlp_params(tcnn::pcg32& rnd, float* params_full_precision, T* params, T* inference_params, T* backward_params, T* gradients, float scale = 1) {
		size_t offset = 0;
		printf("initialize density network!\n");

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
		#endif

		offset += m_density_network->n_params();		
	}


	size_t n_params() const override {
		return m_pos_encoding->n_params() + m_density_network->n_params() + m_dir_encoding->n_params() + m_rgb_network->n_params() + m_variance_network->n_params();
	}

	size_t n_params_canonical() const override{
		return m_pos_encoding->n_params() + m_density_network->n_params() + m_dir_encoding->n_params() + m_rgb_network->n_params() + m_variance_network->n_params();
	}

	size_t n_params_delta() const override{
		return 0;
	}

	tcnn::json n_params_components() const override{
		// make ensure the order is the same as the initilize params.
		return {
			{0, {"density_network", m_density_network->n_params()}},
			{1, {"rgb_network", m_rgb_network->n_params()}},
			{2, {"variance_network", m_variance_network->n_params()}},
			{3, {"pos_encoding", m_pos_encoding->n_params()}},
			{4, {"dir_encoding", m_dir_encoding->n_params()}},
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
		return 7; 
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
		return layers;
	}

	std::vector<std::pair<uint32_t, uint32_t>> layer_sizes_canonical() const override {
		auto layers = m_density_network->layer_sizes();
		auto rgb_layers = m_rgb_network->layer_sizes();
		layers.insert(layers.end(), rgb_layers.begin(), rgb_layers.end());
		return layers;
	}

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
		printf("accumulated transition params num: %lu\n",accumulated_transition->n_params());
		
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
			tcnn::generate_random_uniform<float>(rnd, 1u, params_full_precision + offset + 4u, 1.000f, 1.000f);
			tcnn::generate_random_uniform<float>(rnd, 3u, params_full_precision + offset + 5u, 0.000f, 0.000f);
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

		printf("******finish init accumulation parameters*****\n");
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

	float cos_anneal_ratio() const{
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
			{"otype", "NerfNetwork"},
			{"pos_encoding", m_pos_encoding->hyperparams()},
			{"dir_encoding", m_dir_encoding->hyperparams()},
			{"density_network", density_network_hyperparams},
			{"rgb_network", m_rgb_network->hyperparams()},
			{"variance_network", m_variance_network->hyperparams()},
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

	bool m_train_canonical = true;
	bool m_train_delta = false;
	bool m_use_delta = true;

	void accumulate_global_movement(cudaStream_t stream){
		#if rotation_reprensentation
			tcnn::linear_kernel(accumulate_global_movement_rotation_6d_kernel<T>, 0, stream, 1u,
					m_delta_network->rotation()->params(), m_delta_network->transition()->params(),
					accumulated_rotation->params(), accumulated_transition->params());

			CUDA_CHECK_THROW(cudaMemcpy(accumulated_rotation->params_inference(),accumulated_rotation->params(), sizeof(T)*accumulated_rotation->n_params(), cudaMemcpyDeviceToDevice));
			CUDA_CHECK_THROW(cudaMemcpy(accumulated_transition->params_inference(),accumulated_transition->params(), sizeof(T)*accumulated_transition->n_params(), cudaMemcpyDeviceToDevice));
		#else
			tcnn::linear_kernel(accumulate_global_movement_rotation_quaternion_kernel<T>, 0, stream, 1u,
					m_delta_network->rotation()->params(), m_delta_network->transition()->params(),
					accumulated_rotation->params(), accumulated_transition->params());
		#endif

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

		#else
			printf("not implemented!\n");
			exit(1);
			tcnn::linear_kernel(accumulate_global_movement_rotation_quaternion_kernel<T>, 0, stream, 1u,
					accumulated_rotation->params(), accumulated_transition->params(),
					save_accumulated_rotation.data(), save_accumulated_transition.data());
		#endif

		network_config["snapshot"]["rotation"] = tcnn::gpu_memory_to_json_binary(accumulated_rotation->params(), sizeof(T) * accumulated_rotation->n_params());
		network_config["snapshot"]["transition"] = tcnn::gpu_memory_to_json_binary(accumulated_transition->params(), sizeof(T) * accumulated_transition->n_params());
	
	}

	void load_global_movement(const tcnn::json network_config) {
		printf("******start load global movement parameters*****\n");

		tcnn::GPUMemory<T> params_hp = network_config["snapshot"]["rotation"];

		size_t n_params = params_hp.size();
		
		printf("rotation n_params: %lu\n", n_params);

		parallel_for_gpu(n_params, [params=accumulated_rotation->params(), params_hp=params_hp.data()] __device__ (size_t i) {
			params[i] = (T)params_hp[i];
		});

		params_hp = network_config["snapshot"]["transition"];

		n_params = params_hp.size();

		printf("transition n_params: %lu\n", n_params);

		parallel_for_gpu(n_params, [params=accumulated_transition->params(), params_hp=params_hp.data()] __device__ (size_t i) {
			params[i] = (T)params_hp[i];
		});

		precision_t* rotation_quat_gpu = accumulated_rotation->params();
		precision_t* transition_gpu = accumulated_transition->params();
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
		network_config["snapshot"]["local_rotation"] = tcnn::gpu_memory_to_json_binary(m_delta_network->rotation()->params(), sizeof(T) * m_delta_network->rotation()->n_params());
		network_config["snapshot"]["local_transition"] = tcnn::gpu_memory_to_json_binary(m_delta_network->transition()->params(), sizeof(T) * m_delta_network->transition()->n_params());
	
	}

	void load_local_movement(const tcnn::json network_config) {
		printf("******start load local movement parameters*****\n");

		tcnn::GPUMemory<T> params_hp = network_config["snapshot"]["local_rotation"];

		size_t n_params = params_hp.size();
		
		printf("rotation n_params: %lu\n", n_params);

		parallel_for_gpu(n_params, [params=m_delta_network->rotation()->params(), params_hp=params_hp.data()] __device__ (size_t i) {
			params[i] = (T)params_hp[i];
		});

		params_hp = network_config["snapshot"]["local_transition"];

		n_params = params_hp.size();

		printf("transition n_params: %lu\n", n_params);

		parallel_for_gpu(n_params, [params=m_delta_network->transition()->params(), params_hp=params_hp.data()] __device__ (size_t i) {
			params[i] = (T)params_hp[i];
		});
	}

private:
	std::unique_ptr<tcnn::Network<T>> m_density_network;
	std::unique_ptr<tcnn::Network<T>> m_rgb_network;
	std::shared_ptr<tcnn::Encoding<T>> m_pos_encoding;
	std::shared_ptr<tcnn::Encoding<T>> m_dir_encoding;

	std::shared_ptr<DeltaNetwork<T>> m_delta_network;
	std::shared_ptr<TrainableBuffer<1, 1, T>> accumulated_transition;
	std::shared_ptr<TrainableBuffer<1, 1, T>> accumulated_rotation;

	tcnn::GPUMemory<char> m_accumulation_params_buffer;

	// variance
	float m_variance;
	T m_sdf_bias;

	uint32_t m_rgb_network_input_width;
	uint32_t m_density_network_input_width;
	uint32_t m_n_pos_dims;
	uint32_t m_n_dir_dims;
	uint32_t m_n_extra_dims; // extra dimensions are assumed to be part of a compound encoding with dir_dims
	uint32_t m_dir_offset;

	// Storage of forward pass data
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
