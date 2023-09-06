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
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import commentjson as json

import numpy as np

import sys
import time

from common import *
from render_utils import render_img_training_view

from shutil import copyfile
from tqdm import tqdm

import pyngp as ngp # noqa

from torch.utils.tensorboard import SummaryWriter

def parse_args():
	parser = argparse.ArgumentParser(description="Run neural graphics primitives testbed with additional configuration & output options")

	parser.add_argument("--name", default="neus", type=str, required=True)
	parser.add_argument("--scene", "--training_data", default="", help="The scene to load. Can be the scene's name or a full path to the training data.")
	parser.add_argument("--mode", default="", const="nerf", nargs="?", choices=["nerf", "sdf", "image", "volume"], help="Mode can be 'nerf', 'sdf', or 'image' or 'volume'. Inferred from the scene if unspecified.")
	parser.add_argument("--network", default="", help="Path to the network config. Uses the scene's default if unspecified.")

	parser.add_argument("--load_snapshot", default="", help="Load this snapshot before training. recommended extension: .msgpack")
	parser.add_argument("--save_snapshot", default="", help="Save this snapshot after training. recommended extension: .msgpack")

	parser.add_argument("--nerf_compatibility", action="store_true", help="Matches parameters with original NeRF. Can cause slowness and worse results on some scenes.")
	parser.add_argument("--test_transforms", default="", help="Path to a nerf style transforms json from which we will compute PSNR.")
	parser.add_argument("--near_distance", default=-1, type=float, help="set the distance from the camera at which training rays start for nerf. <0 means use ngp default")

	parser.add_argument("--screenshot_transforms", default="", help="Path to a nerf style transforms.json from which to save screenshots.")
	parser.add_argument("--screenshot_frames", nargs="*", help="Which frame(s) to take screenshots of.")
	parser.add_argument("--screenshot_dir", default="", help="Which directory to output screenshots to.")
	parser.add_argument("--screenshot_spp", type=int, default=16, help="Number of samples per pixel in screenshots.")

	parser.add_argument("--save_mesh", action="store_true")
	parser.add_argument("--save_mesh_path", default="", help="Output a marching-cubes based mesh from the NeRF or SDF model. Supports OBJ and PLY format.")
	parser.add_argument("--marching_cubes_res", default=512, type=int, help="Sets the resolution for the marching cubes grid.")

	parser.add_argument("--width", "--screenshot_w", type=int, default=0, help="Resolution width of GUI and screenshots.")
	parser.add_argument("--height", "--screenshot_h", type=int, default=0, help="Resolution height of GUI and screenshots.")

	parser.add_argument("--gui", action="store_true", help="Run the testbed GUI interactively.")
	parser.add_argument("--train", action="store_true", help="If the GUI is enabled, controls whether training starts immediately.")
	parser.add_argument("--n_steps", type=int, default=-1, help="Number of steps to train for before quitting.")

	parser.add_argument("--sharpen", default=0, help="Set amount of sharpening applied to NeRF training images.")

	## na_test
	parser.add_argument('--test_camera_view', type=int,default=0)
	parser.add_argument('--test', action='store_true')
	parser.add_argument('--render_img_HW', type=int, default=None)
	parser.add_argument("--shaded_mesh", action='store_true')
	parser.add_argument("--white_bkgd", action='store_true')

	args = parser.parse_args()
	return args


if __name__ == "__main__":
	args = parse_args()

	args.output_path = os.path.join('output',args.name)
	os.makedirs(os.path.join(args.output_path,"checkpoints"), exist_ok=True)
	os.makedirs(os.path.join(args.output_path,"mesh"), exist_ok=True)
	
	time_name = time.strftime("%m_%d_%H_%M", time.localtime())
	writer = SummaryWriter(log_dir=os.path.join('output',  args.name, 'logs', time_name))

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


	if args.load_snapshot:
		print("Loading snapshot ", args.load_snapshot)
		testbed.load_snapshot(args.load_snapshot)
	else:
		testbed.reload_network_from_file(network)


	if args.test:
		log_path = os.path.join(args.output_path, f"eval_log.txt")
		log_ptr = open(log_path, "w+")

		if args.load_snapshot:
			print("Loading snapshot ", args.load_snapshot)
			testbed.load_snapshot(args.load_snapshot)
		else:
			print("specify a checkpoint path")
			exit(1)
		
		args.save_mesh_path = os.path.join(args.output_path,"mesh",f"{-1}.obj")
		if args.save_mesh_path and args.save_mesh:
			res = args.marching_cubes_res or 256
			print(f"Generating mesh via marching cubes and saving to {args.save_mesh_path}. Resolution=[{res},{res},{res}]")
			testbed.compute_and_save_marching_cubes_mesh(args.save_mesh_path, [res, res, res])
			
		render_img_training_view(args, testbed, log_ptr, args.scene)

	else:
		ref_transforms = {}
		if args.screenshot_transforms: # try to load the given file straight away
			print("Screenshot transforms from ", args.screenshot_transforms)
			with open(args.screenshot_transforms) as f:
				ref_transforms = json.load(f)

		if args.gui:
			# Pick a sensible GUI resolution depending on arguments.
			sw = args.width or 1920
			sh = args.height or 1080
			while sw*sh > 1920*1080*4:
				sw = int(sw / 2)
				sh = int(sh / 2)
			testbed.init_window(sw, sh)

		testbed.shall_train = args.train if args.gui else True


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
			n_steps = 100000

		args.save_snapshot = os.path.join(args.output_path,"checkpoints",f"{n_steps}.msgpack")
		args.save_mesh_path = os.path.join(args.output_path,"mesh",f"{n_steps}.obj")

		tqdm_last_update = 0
		if n_steps > 0:
			with tqdm(desc="Training", total=n_steps, unit="step") as t:
				while testbed.frame():
					if testbed.want_repl():
						repl(testbed)
					# What will happen when training is done?
					if testbed.training_step >= n_steps:
						if args.gui:
							testbed.shall_train = False
						else:
							break

					# Update progress bar
					if testbed.training_step < old_training_step or old_training_step == 0:
						old_training_step = 0
						t.reset()

					now = time.monotonic()
					if now - tqdm_last_update > 0.1:
						t.update(testbed.training_step - old_training_step)
						t.set_postfix(loss=testbed.loss)
						old_training_step = testbed.training_step
						tqdm_last_update = now

					# writer.add_scalar('psnr', s_val.mean(), self.iter_step)
					if testbed.training_step % 20 == 0:
						writer.add_scalar('loss/rgb_loss', testbed.loss, testbed.training_step)
						writer.add_scalar('loss/ek_loss', testbed.ek_loss, testbed.training_step)
						writer.add_scalar('loss/mask_loss', testbed.mask_loss, testbed.training_step)


		if args.save_snapshot:
			print("Saving snapshot ", args.save_snapshot)
			testbed.save_snapshot(args.save_snapshot, False)

		
		res = args.marching_cubes_res or 256
		print(f"Generating mesh via marching cubes and saving to {args.save_mesh_path}. Resolution=[{res},{res},{res}]")
		testbed.compute_and_save_marching_cubes_mesh(args.save_mesh_path, [res, res, res])

		log_path = os.path.join(args.output_path, f"eval_log.txt")
		log_ptr = open(log_path, "w+")
		render_img_training_view(args, testbed, log_ptr, args.scene)
		


		if args.test_transforms:
			print("Evaluating test transforms from ", args.test_transforms)
			with open(args.test_transforms) as f:
				test_transforms = json.load(f)
			data_dir=os.path.dirname(args.test_transforms)
			totmse = 0
			totpsnr = 0
			totssim = 0
			totcount = 0
			minpsnr = 1000
			maxpsnr = 0

			# Evaluate metrics on black background
			testbed.background_color = [0.0, 0.0, 0.0, 1.0]

			# Prior nerf papers don't typically do multi-sample anti aliasing.
			# So snap all pixels to the pixel centers.
			testbed.snap_to_pixel_centers = True
			spp = 8

			testbed.nerf.rendering_min_transmittance = 1e-4
			import pdb
			pdb.set_trace()
			if "from_na" in test_transforms.keys():
				pass
			else:
				testbed.fov_axis = 0
				testbed.fov = test_transforms["camera_angle_x"] * 180 / np.pi
				testbed.shall_train = False

			with tqdm(list(enumerate(test_transforms["frames"])), unit="images", desc=f"Rendering test frame") as t:
				for i, frame in t:
					p = frame["file_path"]
					if "." not in p:
						p = p + ".png"
					ref_fname = os.path.join(data_dir, p)
					if not os.path.isfile(ref_fname):
						ref_fname = os.path.join(data_dir, p + ".png")
						if not os.path.isfile(ref_fname):
							ref_fname = os.path.join(data_dir, p + ".jpg")
							if not os.path.isfile(ref_fname):
								ref_fname = os.path.join(data_dir, p + ".jpeg")
								if not os.path.isfile(ref_fname):
									ref_fname = os.path.join(data_dir, p + ".exr")

					ref_image = read_image(ref_fname)

					# NeRF blends with background colors in sRGB space, rather than first
					# transforming to linear space, blending there, and then converting back.
					# (See e.g. the PNG spec for more information on how the `alpha` channel
					# is always a linear quantity.)
					# The following lines of code reproduce NeRF's behavior (if enabled in
					# testbed) in order to make the numbers comparable.
					if testbed.color_space == ngp.ColorSpace.SRGB and ref_image.shape[2] == 4:
						# Since sRGB conversion is non-linear, alpha must be factored out of it
						ref_image[...,:3] = np.divide(ref_image[...,:3], ref_image[...,3:4], out=np.zeros_like(ref_image[...,:3]), where=ref_image[...,3:4] != 0)
						ref_image[...,:3] = linear_to_srgb(ref_image[...,:3])
						ref_image[...,:3] *= ref_image[...,3:4]
						ref_image += (1.0 - ref_image[...,3:4]) * testbed.background_color
						ref_image[...,:3] = srgb_to_linear(ref_image[...,:3])

					if i == 0:
						write_image("ref.png", ref_image)

					testbed.set_nerf_camera_matrix(np.matrix(frame["transform_matrix"])[:-1,:])
					# testbed.set_camera_to_training_view(i)
					image = testbed.render(ref_image.shape[1], ref_image.shape[0], spp, True)

					if i == 0:
						write_image("out.png", image)

					diffimg = np.absolute(image - ref_image)
					diffimg[...,3:4] = 1.0
					if i == 0:
						write_image("diff.png", diffimg)

					A = np.clip(linear_to_srgb(image[...,:3]), 0.0, 1.0)
					R = np.clip(linear_to_srgb(ref_image[...,:3]), 0.0, 1.0)
					mse = float(compute_error("MSE", A, R))
					ssim = float(compute_error("SSIM", A, R))
					totssim += ssim
					totmse += mse
					psnr = mse2psnr(mse)
					totpsnr += psnr
					minpsnr = psnr if psnr<minpsnr else minpsnr
					maxpsnr = psnr if psnr>maxpsnr else maxpsnr
					totcount = totcount+1
					t.set_postfix(psnr = totpsnr/(totcount or 1))
					# break

			psnr_avgmse = mse2psnr(totmse/(totcount or 1))
			psnr = totpsnr/(totcount or 1)
			ssim = totssim/(totcount or 1)
			print(f"PSNR={psnr} [min={minpsnr} max={maxpsnr}] SSIM={ssim}")

		if args.save_mesh_path:
			res = args.marching_cubes_res or 256
			print(f"Generating mesh via marching cubes and saving to {args.save_mesh_path}. Resolution=[{res},{res},{res}]")
			testbed.compute_and_save_marching_cubes_mesh(args.save_mesh_path, [res, res, res])

		if args.width:
			if ref_transforms:
				testbed.fov_axis = 0
				testbed.fov = ref_transforms["camera_angle_x"] * 180 / np.pi
				if not args.screenshot_frames:
					args.screenshot_frames = range(len(ref_transforms["frames"]))
				print(args.screenshot_frames)
				for idx in args.screenshot_frames:
					f = ref_transforms["frames"][int(idx)]
					cam_matrix = f["transform_matrix"]
					testbed.set_nerf_camera_matrix(np.matrix(cam_matrix)[:-1,:])
					outname = os.path.join(args.screenshot_dir, os.path.basename(f["file_path"]))

					# Some NeRF datasets lack the .png suffix in the dataset metadata
					if not os.path.splitext(outname)[1]:
						outname = outname + ".png"

					print(f"rendering {outname}")
					image = testbed.render(args.width or int(ref_transforms["w"]), args.height or int(ref_transforms["h"]), args.screenshot_spp, True)
					os.makedirs(os.path.dirname(outname), exist_ok=True)
					write_image(outname, image)
			elif args.screenshot_dir:
				outname = os.path.join(args.screenshot_dir, args.scene + "_" + network_stem)
				print(f"Rendering {outname}.png")
				image = testbed.render(args.width, args.height, args.screenshot_spp, True)
				if os.path.dirname(outname) != "":
					os.makedirs(os.path.dirname(outname), exist_ok=True)
				write_image(outname + ".png", image)



