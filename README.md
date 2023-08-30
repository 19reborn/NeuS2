# NeuS2: Fast Learning of Neural Implicit Surfaces for Multi-view Reconstruction
### [[Project]](https://vcai.mpi-inf.mpg.de/projects/NeuS2/)[ [Paper]](https://arxiv.org/abs/2212.05231)
<br/>

> NeuS2: Fast Learning of Neural Implicit Surfaces for Multi-view Reconstruction

> [Yiming Wang*](https://19reborn.github.io/), Qin Han*, [Marc Habermann](https://people.mpi-inf.mpg.de/~mhaberma/), [Kostas Daniilidis](https://www.cis.upenn.edu/~kostas/), [Christian Theobalt](http://people.mpi-inf.mpg.de/~theobalt/), [Lingjie Liu](https://lingjie0206.github.io/)

> ICCV 2023

<img src="docs/assets_readme/intro_1.gif" height="342"/> 
<img src="docs/assets_readme/intro_2.gif" height="342"/>

[NeuS2](https://vcai.mpi-inf.mpg.de/projects/NeuS2/) is a method for fast neural surface reconstruction, which achieves two orders of magnitude improvement in terms of acceleration without compromising reconstruction quality, compared to [NeuS](https://lingjie0206.github.io/papers/NeuS/). To accelerate the training process, we integrate multi-resolution hash encodings into a neural surface representation and implement our whole algorithm in CUDA. In addition, we extend our method for reconstructing dynamic scenes with an incremental training strategy.

This project is an extension of [Instant-NGP](https://github.com/NVlabs/instant-ngp) enabling it to model neural surface representation and dynmaic scenes. We extended:
- dependencies/[neus2_TCNN](https://github.com/19reborn/NeuS2_TCNN.git)
  - add second-order derivative backpropagation computation for MLP;
  - add progressive training for Grid Encoding.
- neural-graphics-primitives
  - extend NeRF mode for **NeuS**;
  - add support for dynamic scenes.

## Installation

**Please first see [Instant-NGP](https://github.com/NVlabs/instant-ngp#building-instant-ngp-windows--linux) for original requirements and compilation instructions. NeuS2 follows the installing steps of Instant-NGP.**

Clone this repository and all its submodules using the following command:
```
git clone --recursive https://github.com/19reborn/NeuS2
cd NeuS2
```

Then use CMake to build the project:

```
cmake . -B build
cmake --build build --config RelWithDebInfo -j 
```

For python useage, first install dependencies with conda and pip:
```
conda create -n neus2 python=3.9
conda activate neus2
pip install -r requirements.txt
```

Then check https://pytorch.org/ for pytorch installation, and https://github.com/facebookresearch/pytorch3d for pytorch3d installation.

If you meet problems of compiling, you may find solutions in https://github.com/NVlabs/instant-ngp#troubleshooting-compile-errors.

## Training

### Static Scene

You can specify a static scene by setting `--scene` to a `.json` file containing data descriptions.

The [DTU](https://roboimagedata.compute.dtu.dk/?page_id=36) Scan24 scene can be downloaded from [Google Drive](https://drive.google.com/file/d/1KkNkljeYNwg5dH_y080AlzslVl1RTnKy/view?usp=sharing):

```sh
./build/testbed --scene ${data_path}/transform.json
```
Or, you can run the experiment in an automated fashion through python bindings:

```sh
python scripts/run.py --scene ${data_path}/transform.json --name ${your_experiment_name} --network ${config_name} --n_steps ${training_steps}
```

For training on DTU dataset, `config_name` can be `dtu.json` and `training_steps` can be `15000`.

Also, you can use `scripts/run_dynamic.py` as:
```sh
python scripts/run_dynamic.py --scene ${data_path}/transform.json --name ${your_experiment_name} --network ${config_name}
```
, where the number of training iterations is specified in the config.

The outputs and logs of the experiment can be found at `output/${your_experiment_name}/`.

### Dynamic Scene

To specify a dynamic scene, you should set `--scene` to a directory containing `.json` files that describe training frames.

```sh
./build/testbed --scene ${data_dirname}
```

Or, run `scripts/run_dynamic.py` using python:

```sh
python scripts/run_dynamic.py --scene ${data_dirname} --name ${your_experiment_name} --network ${config_name}
```

There are some hyperparameters of the network configuration, such as `configs/nerf/base.json`, to control the dynamic training process:
- `first_frame_max_training_step`: determine the number of training iterations for the first frame, default `2000`.
- `next_frame_max_training_step`: determine the number of training iterations for subsequent frames, default `1300`, including global transformation prediction.
- `predict_global_movement`: set `true` if use global transformation prediction.
- `predict_global_movement_training_step`: determine the number of training iterations for global transformation prediction, default `300`. Only valid when `predict_global_movement` is `true`.

Also, we provide scripts to reconstruct dynamic scenes by reconstructing static scene frame by frame.

```sh
python scripts/run_per_frame.py --base_dir ${data_dirname} --output_dir ${output_path} --config ${config_name}
```

Dynamic scene examples can be downloaded from [Google Drive](https://drive.google.com/file/d/1hvqaupbufxuadVMP_2reTAqnaEZ4xvhj/view?usp=sharing).

### Unbounded Scenes (Unmasked scenes)
** This implementation is still testing **

We add an additional NeRF network to model the background for unbounded scenes or unmasked scenes. You can use the configuration `womask.json` for this mode. 

## Data Convention

**NeuS2 supports the data format provided by [Instant-NGP](https://github.com/NVlabs/instant-ngp).** Also, you can use NeuS2's data format (with `from_na=true`).

Our NeuS2 implementation expects initial camera parameters to be provided in a `transforms.json` file, organized as follows:
```
{
	"from_na": true, # if true, specify NeuS2's data format, which rotates the coordinate system by the x-axis for 180 degrees
	"w": 512, # image_width
	"h": 512, # image_height
	"aabb_scale": 1.0,
	"scale": 0.5,
	"offset": [
		0.5,
		0.5,
		0.5
	],
	"frames": [ # list of reference images & corresponding camera parameters
		{
			"file_path": "images/000000.png", # specify the image path (should be relative path)
			"transform_matrix": [ # specify extrinsic parameters of camera, a camera to world transform (shape: [4, 4])
				[
					0.9702627062797546,
					-0.01474287360906601,
					-0.2416049838066101,
					0.9490470290184021
				],
				[
					0.0074799139983952045,
					0.9994929432868958,
					-0.0309509988874197,
					0.052045613527297974
				],
				[
					0.2419387847185135,
					0.028223415836691856,
					0.9698809385299683,
					-2.6711924076080322
				],
				[
					0.0,
					0.0,
					0.0,
					1.0
				]
			],
			"intrinsic_matrix": [ # specify intrinsic parameters of camera (shape: [4, 4])
				[
					2892.330810546875,
					-0.00025863019982352853,
					823.2052612304688,
					0.0
				],
				[
					0.0,
					2883.175537109375,
					619.0709228515625,
					0.0
				],
				[
					0.0,
					0.0,
					1.0,
					0.0
				],
				[
					0.0,
					0.0,
					0.0,
					1.0
				]
			]
		},
		...
	]
}
```
Each `transforms.json` file contains data about a single frame, including camera parameters and image paths. You can specify specific transform files, such as `transforms_test.json` and `transforms_train.json`, to use for training and testing with data splitting.

For example, you can organize your dynamic scene data as:
```
<case_name>
|-- images
   |-- 000280 # target frame of the scene
      |-- image_c_000_f_000280.png
      |-- image_c_001_f_000280.png
      ...
   |-- 000281
      |-- image_c_000_f_000281.png
      |-- image_c_001_f_000281.png
      ...
   ...
|-- train
   |-- transform_000280.json
   |-- transform_000281.json
   ...
|-- test
   |-- transform_000280.json
   |-- transform_000281.json
   ...
```

Images are four-dimensional, with three channels for RGB and one channel for the mask.

We also provide a data conversion from [NeuS](https://lingjie0206.github.io/papers/NeuS/) to our data convention, which can be found in `tools/data_format_from_neus.py`.

## Citation

```bibtex
@inproceedings{neus2,
    title={NeuS2: Fast Learning of Neural Implicit Surfaces for Multi-view Reconstruction}, 
    author={Wang, Yiming and Han, Qin and Habermann, Marc and Daniilidis, Kostas and Theobalt, Christian and Liu, Lingjie},
    year={2023},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)}
}
```

