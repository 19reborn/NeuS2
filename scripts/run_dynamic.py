#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
import commentjson as json

import torch
import numpy as np
import sys
import time

from common import *
from render_utils import *

from shutil import copyfile
from tqdm import tqdm
import shutil

import pyngp as ngp # noqa
import trimesh

from torch.utils.tensorboard import SummaryWriter

os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

def parse_args():
	parser = argparse.ArgumentParser(description="Run neural graphics primitives testbed with additional configuration & output options")

	parser.add_argument("--name", default="neus", type=str, required=True)
	parser.add_argument("--scene", "--training_data", default="", help="The scene to load. Can be the scene's name or a full path to the training data.")
	parser.add_argument("--mode", default="", const="nerf", nargs="?", choices=["nerf", "sdf", "image", "volume"], help="Mode can be 'nerf', 'sdf', or 'image' or 'volume'. Inferred from the scene if unspecified.")
	parser.add_argument("--network", default="", help="Path to the network config. Uses the scene's default if unspecified.")

	parser.add_argument("--load_snapshot", default="", help="Load this snapshot before training. recommended extension: .msgpack")
	parser.add_argument("--save_snapshot_per_frame", action="store_true", help="Save this snapshot after training every frame. recommended extension: .msgpack")

	parser.add_argument("--train_from_frame", default="", help="Save this snapshot after training every frame. recommended extension: .msgpack")
	parser.add_argument("--test_from_frame", default="", help="Save this snapshot after training every frame. recommended extension: .msgpack")
	
	## test config
	parser.add_argument("--dynamic_test", action="store_true")
	parser.add_argument("--dynamic_save_mesh", action='store_true')
	parser.add_argument("--dynamic_save_mesh_only", action='store_true')
	parser.add_argument('--test_camera_view', type=int, default=0)
	parser.add_argument('--test_psnr', action='store_true')
	parser.add_argument("--white_bkgd", action='store_true')
	parser.add_argument('--render_img_HW', type=int, default=None)

	parser.add_argument("--save_mesh", action="store_true")
	parser.add_argument("--save_mesh_path", default="", help="Output a marching-cubes based mesh from the NeRF or SDF model. Supports OBJ and PLY format.")
	parser.add_argument("--marching_cubes_res", default=256, type=int, help="Sets the resolution for the marching cubes grid.")
	parser.add_argument("--bbox_min", type=float, default=0.0)
	parser.add_argument("--bbox_max", type=float, default=1.0)
	parser.add_argument("--shaded_mesh", action='store_true')

	parser.add_argument("--nerf_compatibility", action="store_true", help="Matches parameters with original NeRF. Can cause slowness and worse results on some scenes.")
	parser.add_argument("--test_transforms", default="", help="Path to a nerf style transforms json from which we will compute PSNR.")
	parser.add_argument("--near_distance", default=-1, type=float, help="set the distance from the camera at which training rays start for nerf. <0 means use ngp default")
	parser.add_argument("--exposure", default=0.0, type=float, help="Controls the brightness of the image. Positive numbers increase brightness, negative numbers decrease it.")

	parser.add_argument("--screenshot_transforms", default="", help="Path to a nerf style transforms.json from which to save screenshots.")
	parser.add_argument("--screenshot_frames", nargs="*", help="Which frame(s) to take screenshots of.")
	parser.add_argument("--screenshot_dir", default="", help="Which directory to output screenshots to.")
	parser.add_argument("--screenshot_spp", type=int, default=16, help="Number of samples per pixel in screenshots.")

	parser.add_argument("--video_camera_path", default="", help="The camera path to render, e.g., base_cam.json.")
	parser.add_argument("--video_camera_smoothing", action="store_true", help="Applies additional smoothing to the camera trajectory with the caveat that the endpoint of the camera path may not be reached.")
	parser.add_argument("--video_loop_animation", action="store_true", help="Connect the last and first keyframes in a continuous loop.")
	parser.add_argument("--video_fps", type=int, default=60, help="Number of frames per second.")
	parser.add_argument("--video_n_seconds", type=int, default=1, help="Number of seconds the rendered video should be long.")
	parser.add_argument("--video_spp", type=int, default=8, help="Number of samples per pixel. A larger number means less noise, but slower rendering.")
	parser.add_argument("--video_output", type=str, default="video.mp4", help="Filename of the output video.")


	parser.add_argument("--n_steps", type=int, default=-1, help="Number of steps to train for before quitting.") # not used

	parser.add_argument("--sharpen", default=0, help="Set amount of sharpening applied to NeRF training images.")


	args = parser.parse_args()
	return args


if __name__ == "__main__":
	args = parse_args()

	args.output_path = os.path.join('output',args.name)
	os.makedirs(os.path.join(args.output_path,"checkpoints"), exist_ok=True)
	os.makedirs(os.path.join(args.output_path,"mesh"), exist_ok=True)
	
	time_name = time.strftime("%m_%d_%H_%M", time.localtime())
	writer_all = SummaryWriter(log_dir=os.path.join('output',  args.name, 'logs', time_name, 'all'))

	mode = ngp.TestbedMode.Nerf 
	configs_dir = os.path.join(ROOT_DIR, "configs", "nerf")

	base_network = os.path.join(configs_dir, "base.json")
	network = args.network if args.network else base_network
	if not os.path.isabs(network):
		network = os.path.join(configs_dir, network)
	
	def file_backup(output_path, config_path):
		rec_dir = os.path.join(output_path, 'recording')
		os.makedirs(rec_dir, exist_ok=True)
		copyfile(config_path, os.path.join(rec_dir, 'config.json'))
		
		filepath = os.path.join('src', 'testbed_nerf.cu')
		if os.path.exists(filepath):
			copyfile(filepath, os.path.join(rec_dir, 'testbed_nerf.cu'))
		filepath = os.path.join('src', 'testbed.cu')
		if os.path.exists(filepath):
			copyfile(filepath, os.path.join(rec_dir, 'testbed.cu'))
		filepath = os.path.join('include', 'neural-graphics-primitives', 'nerf_network.h')
		if os.path.exists(filepath):
			copyfile(filepath, os.path.join(rec_dir, 'nerf_network.h'))
	
	file_backup(args.output_path, network)
 
	testbed = ngp.Testbed(mode)
	testbed.nerf.sharpen = float(args.sharpen)

	if mode == ngp.TestbedMode.Sdf:
		testbed.tonemap_curve = ngp.TonemapCurve.ACES

	if args.scene:
		scene = args.scene
		testbed.load_training_data(scene)


	if args.load_snapshot and not args.dynamic_test:
		print("Loading snapshot ", args.load_snapshot)
		testbed.load_snapshot(args.load_snapshot)
	else:
		testbed.reload_network_from_file(network, args.output_path)

	if args.dynamic_test:
		print("Running dynamic test")

		all_results = {}
		all_psnr = 0
		all_frame = 0
		os.makedirs(os.path.join(args.output_path, "evaluation"), exist_ok=True)
		if args.test_from_frame != "":
			testbed.change_to_frame(int(args.test_from_frame))

		while True:
			if os.path.isdir(args.scene):
				all_transform_path = sorted(list(filter(lambda path:"downsample" not in path.split('/')[-1], glob.glob(args.scene+'/*.json'))))
			else:
				all_transform_path = [args.scene]
			if args.load_snapshot != "":
				load_snapshot_path = os.path.join(args.load_snapshot, "checkpoints") + f"/frame_{testbed.current_training_time_frame}.msgpack"
			else:
				load_snapshot_path = os.path.join(args.output_path,"checkpoints") + f"/frame_{testbed.current_training_time_frame}.msgpack"

			print("Loading snapshot ", load_snapshot_path)

			testbed.load_snapshot(load_snapshot_path)

			if args.dynamic_save_mesh:
				import platform
		
				res = args.marching_cubes_res

				save_mesh_path = os.path.join(args.output_path,"evaluation","mesh",f"frame_{testbed.current_training_time_frame:04}.obj")
				os.makedirs(os.path.join(args.output_path,"evaluation","mesh"), exist_ok=True)

				print(f"Generating mesh via marching cubes and saving to {save_mesh_path}. Resolution=[{res},{res},{res}]")
				aabb = ngp.BoundingBox(np.array([0,args.bbox_min,0]), np.array([1.0,args.bbox_max,1.0]))
				testbed.compute_and_save_marching_cubes_mesh(save_mesh_path, [res, res, res], aabb = aabb)
				if args.dynamic_save_mesh_only:
					if not testbed.training_network_next_frame() and testbed.current_training_time_frame >= testbed.all_training_time_frame - 1:
						break
					continue



			testbed.load_snapshot(load_snapshot_path)
			testbed.prepare_for_test()
			
			ref_images = load_ref_images(args, all_transform_path[testbed.current_training_time_frame])		
			log_ptr, avg_psnr_this_frame = cal_psnr(testbed, ref_images, 1, True, save_dir = os.path.join(args.output_path, "evaluation", "psnr", f"frame_{testbed.current_training_time_frame}"),spp = 8, white_bkgd=args.white_bkgd)
			all_results['frame_'+str(testbed.current_training_time_frame)] = avg_psnr_this_frame
			all_psnr += avg_psnr_this_frame
			all_frame += 1

			with open(os.path.join(args.output_path, "evaluation", f"dynamic_{testbed.current_training_time_frame}_eval_log.json"), "w") as f:
				json.dump(log_ptr, f, indent = 4)

			if not testbed.training_network_next_frame() and testbed.current_training_time_frame >= testbed.all_training_time_frame - 1:
				break

		mean_psnr = all_psnr / all_frame
		all_results['mean_psnr'] = mean_psnr

		with open(os.path.join(args.output_path, "evaluation", "dynamic_eval_log.json"), "w") as f:
			json.dump(all_results, f, indent = 4)


	else:
		ref_transforms = {}
		if args.screenshot_transforms: # try to load the given file straight away
			print("Screenshot transforms from ", args.screenshot_transforms)
			with open(args.screenshot_transforms) as f:
				ref_transforms = json.load(f)

	
		testbed.shall_train = True


		testbed.nerf.render_with_camera_distortion = True

		network_stem = os.path.splitext(os.path.basename(network))[0]

		if args.near_distance >= 0.0:
			print("NeRF training ray near_distance ", args.near_distance)
			testbed.nerf.training.near_distance = args.near_distance

		if args.nerf_compatibility:
			print(f"NeRF compatibility mode enabled")

			# Prior nerf papers accumulate/blend in the sRGB
			# color space. This messes not only with background
			# alpha, but also with DOF effects and the likes.
			# We support this behavior, but we only enable it
			# for the case of synthetic nerf data where we need
			# to compare PSNR numbers to results of prior work.
			testbed.color_space = ngp.ColorSpace.SRGB

			# No exponential cone tracing. Slightly increases
			# quality at the cost of speed. This is done by
			# default on scenes with AABB 1 (like the synthetic
			# ones), but not on larger scenes. So force the
			# setting here.
			testbed.nerf.cone_angle_constant = 0

			# Optionally match nerf paper behaviour and train on a
			# fixed white bg. We prefer training on random BG colors.
			# testbed.background_color = [1.0, 1.0, 1.0, 1.0]
			# testbed.nerf.training.random_bg_color = False

		old_training_step = 0
		n_steps = args.n_steps
		if n_steps < 0:
			n_steps = testbed.next_frame_max_training_step + 1

		args.save_snapshot = os.path.join(args.output_path,"checkpoints",f"{n_steps}.msgpack")
		args.save_mesh_path = os.path.join(args.output_path,"mesh",f"{n_steps}.obj")

		trainig_log_path = os.path.join(args.output_path, f"training_log.txt")
		training_log_ptr = open(trainig_log_path, "w+")
		eval_log_path = os.path.join(args.output_path, f"eval_log.txt")
		eval_log_ptr = open(eval_log_path, "w+")

		print("begin training!\b",file=training_log_ptr)
		training_log_ptr.flush()

		tqdm_last_update = 0
		total_steps = 0
		# if args.scene is a directory
		if os.path.isdir(args.scene):
			all_transform_path = sorted(list(filter(lambda path:"downsample" not in path.split('/')[-1], glob.glob(args.scene+'/*.json'))))
		else:
			all_transform_path = [args.scene]
		all_pred_img = []
		all_gt_img = []
		all_pred_mesh = []

		if args.test_psnr:
			ref_images = load_ref_images(args, all_transform_path[testbed.current_training_time_frame])

		if args.train_from_frame != "":
			testbed.change_to_frame(int(args.train_from_frame))
			testbed.load_snapshot(args.load_snapshot)

		writer_frame = SummaryWriter(log_dir=os.path.join('output',  args.name, 'logs', time_name, f'frame_{0}'))
		if n_steps > 0:
			now = time.monotonic()
			print_now = time.monotonic()
			while testbed.frame():
				if testbed.want_repl():
					repl(testbed)

				total_steps += 1
				if total_steps % 20 == 0:
					print(f"training time_frame: {testbed.current_training_time_frame}, training step: {testbed.training_step}, training_time: {time.monotonic()-print_now}, trainig_rgb_loss:{testbed.loss}")
					writer_all.add_scalar('loss/rgb_loss', testbed.loss, total_steps)
					writer_all.add_scalar('loss/ek_loss', testbed.ek_loss, total_steps)
					writer_all.add_scalar('loss/mask_loss', testbed.mask_loss, total_steps)

					writer_frame.add_scalar('loss/rgb_loss', testbed.loss, testbed.training_step)
					writer_frame.add_scalar('loss/ek_loss', testbed.ek_loss, testbed.training_step)
					writer_frame.add_scalar('loss/mask_loss', testbed.mask_loss, testbed.training_step)

				if (testbed.current_training_time_frame == 0 and testbed.training_step == testbed.first_frame_max_training_step) or (testbed.current_training_time_frame != 0 and testbed.training_step == testbed.next_frame_max_training_step):

					print(f"training frame {testbed.current_training_time_frame} for time :{time.monotonic() - now}\n", file=training_log_ptr)
				
					# change to next frame: do some validate
					if args.save_snapshot_per_frame:
						save_snapshot_path = os.path.join(args.output_path,"checkpoints",f"frame_{testbed.current_training_time_frame}.msgpack")
						print(f"Saving snapshot to {save_snapshot_path} ...")
						testbed.save_snapshot(save_snapshot_path, False)

					# save transform
					os.makedirs(os.path.join(args.output_path,"pred_transform"), exist_ok=True)
					save_transform_path = os.path.join(args.output_path,"pred_transform",f"frame_{testbed.current_training_time_frame}.txt")
					print(f"Saving transform to {save_transform_path} ...")
					testbed.save_transform(save_transform_path)

					if args.save_mesh:
						res = args.marching_cubes_res
						save_mesh_path = os.path.join(args.output_path,"mesh",f"frame_{testbed.current_training_time_frame:04}.obj")
						print(f"Generating mesh via marching cubes and saving to {save_mesh_path}. Resolution=[{res},{res},{res}]")
						testbed.compute_and_save_marching_cubes_mesh(save_mesh_path, [res, res, res])
						args.save_mesh_path = save_mesh_path

					
					training_log_ptr.flush()
					pred_img, gt_img, pred_mesh = render_img_training_view(args, testbed, eval_log_ptr, all_transform_path[testbed.current_training_time_frame], testbed.current_training_time_frame)

					# all_pred_img.append(pred_img.astype(np.uint8))
					# all_gt_img.append(gt_img.astype(np.uint8))
					all_pred_img.append(pred_img)
					all_gt_img.append(gt_img)
					if pred_mesh is not None:
						all_pred_mesh.append(pred_mesh[:,:,[2,1,0]].astype(np.uint8))

					# update frame writer
					if testbed.current_training_time_frame != testbed.all_training_time_frame - 1:
						writer_frame = SummaryWriter(log_dir=os.path.join('output',  args.name, 'logs', time_name, f'frame_{testbed.current_training_time_frame + 1}'))
						if args.test_psnr:					
							ref_images = load_ref_images(args, all_transform_path[testbed.current_training_time_frame + 1])
					now = time.monotonic()
					print_now = time.monotonic()

				if (testbed.current_training_time_frame > 0 and testbed.current_training_time_frame == testbed.all_training_time_frame - 1 and testbed.training_step >= testbed.next_frame_max_training_step)  or (testbed.current_training_time_frame == 0 and testbed.current_training_time_frame == testbed.all_training_time_frame - 1 and testbed.training_step >= testbed.first_frame_max_training_step):
					break


			video_path = os.path.join(args.output_path, 'video',f'{args.test_camera_view:04}')
			os.makedirs(video_path, exist_ok=True)
			# pred_writer = imageio.get_writer(os.path.join(video_path,'pred_img.gif'), fps=15)
			# gt_writer = imageio.get_writer(os.path.join(video_path,'gt_img.gif'), fps=15)
			# if len(all_pred_mesh) > 0:
				# mesh_writer = imageio.get_writer(os.path.join(video_path,'pred_mesh.gif'), fps=15)
			# for i in range(len(all_pred_img)):
				# pred_writer.append_data(all_pred_img[i])
				# gt_writer.append_data(all_gt_img[i])
				# if len(all_pred_mesh) > 0:
					# mesh_writer.append_data(all_pred_mesh[i])

			imageio.mimsave(os.path.join(video_path,'pred_img.gif'),all_pred_img,fps=10)
			imageio.mimsave(os.path.join(video_path,'gt_img.gif'),all_gt_img,fps=10)
			if len(all_pred_mesh) > 0:
				imageio.mimsave(os.path.join(video_path,'pred_mesh.gif'),all_pred_mesh,fps=10)



	