# NeuS2: Fast Learning of Neural Implicit Surfaces for Multi-view Reconstruction
### [[Project]](https://vcai.mpi-inf.mpg.de/projects/NeuS2/)[ [Paper]](https://arxiv.org/abs/2212.05231)
<br/>

> [Yiming Wang*](https://19reborn.github.io/), Qin Han*, [Marc Habermann](https://people.mpi-inf.mpg.de/~mhaberma/), [Kostas Daniilidis](https://www.cis.upenn.edu/~kostas/), [Christian Theobalt](http://people.mpi-inf.mpg.de/~theobalt/), [Lingjie Liu](https://lingjie0206.github.io/)

> ICCV 2023

<img src="docs/assets_readme/intro_2.gif" height="342"/>

[NeuS2](https://vcai.mpi-inf.mpg.de/projects/NeuS2/) is a method for fast neural surface reconstruction, which achieves two orders of magnitude improvement in terms of acceleration without compromising reconstruction quality, compared to [NeuS](https://lingjie0206.github.io/papers/NeuS/). To accelerate the training process, we integrate multi-resolution hash encodings into a neural surface representation and implement our whole algorithm in CUDA. In addition, we extend our method for reconstructing dynamic scenes with an incremental training strategy.

This project is an extension of [Instant-NGP](https://github.com/NVlabs/instant-ngp) enabling it to model neural surface representation and dynmaic scenes. We extended:
- dependencies/[neus2_TCNN](https://github.com/19reborn/NeuS2_TCNN.git)
  - add second-order derivative backpropagation computation for MLP;
  - add progressive training for Grid Encoding.
- neural-graphics-primitives
  - extend NeRF mode for **NeuS**;
  - add support for dynamic scenes.

### Updates

- [ ] [08/30/2023] Released NeuS2++ (See [neuspp](https://github.com/19reborn/NeuS2/tree/neuspp) branch). Now you can reconstruct unmasked/unbounded scenes in seconds!

- [ ] [08/15/2023] Released official codes!

## Table Of Contents

- [Gallery](#gallery)
- [Installation](#installation)
- [Training](#training)
- [Data](#data)
- [Data Convention](#data-convention)
- [Acknowledgements \& Citation](#acknowledgements--citation)


## Gallery

|                                                              |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Synthetic Scene **surface reconstruction** in 5 minutes <br />Input: multi-view images with mask <br />Output: freeview synthesis, mesh <br/> <video src="https://github.com/19reborn/NeuS2/assets/142323223/f702cebc-d36e-4958-a2e3-42647dddccf9"></video> | Synthetic Scene **surface reconstruction** in 5 minutes <br />Comparison between NeuS (8h), Instant-NGP (5min) and NeuS2 (5min) <br/> Download dataset from [Google Drive](https://drive.google.com/file/d/1KkNkljeYNwg5dH_y080AlzslVl1RTnKy/view?usp=sharing) <br /> <video src="https://github.com/19reborn/NeuS2/assets/142323223/2df342ca-e639-47c3-b3da-8bbdb37f593b"></video>
| Dynamic Scene **surface reconstruction** in 20s per frame<br />Input: a sequence of multi-view images with mask <br />Output: novel view synthesis, mesh <br/> <video src="https://github.com/19reborn/NeuS2/assets/142323223/28a26d52-e4cf-4e19-bba0-3261c02f7eca"></video> | Long sequence **surface reconstruction** with 2000 frames <br />  Input: a sequence of multi-view images with mask <br /> NeuS2 can handle long sequences input with large movements <br /> <video src="https://github.com/19reborn/NeuS2/assets/142323223/e0f2344b-7fa7-4d7e-9c90-e82664522267"></video> |


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

Then install [pytorch](https://pytorch.org/) and [pytorch3d](https://github.com/facebookresearch/pytorch3d).

If you meet problems of compiling, you may find solutions [here](https://github.com/NVlabs/instant-ngp#troubleshooting-compile-errors).

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

## Data

- Static scene example, [link](https://drive.google.com/file/d/1KkNkljeYNwg5dH_y080AlzslVl1RTnKy/view?usp=sharing).
- Dynamic scene example, [link](https://drive.google.com/file/d/1hvqaupbufxuadVMP_2reTAqnaEZ4xvhj/view?usp=sharing).
- Pretrained model and configuration files for DTU, [link](https://drive.google.com/file/d/1DKXLkOHml6s5IB5yzn_HdYNv-ykGUJxr/view?usp=drive_link).

## Data Convention

**NeuS2 supports the data format provided by [Instant-NGP](https://github.com/NVlabs/instant-ngp).** Also, you can use NeuS2's data format (with `from_na=true`), see [data convention](https://github.com/19reborn/NeuS2/blob/main/DATA_CONVENTION.md).

We also provide a data conversion from [NeuS](https://lingjie0206.github.io/papers/NeuS/) to our data convention, which can be found in `tools/data_format_from_neus.py`.

## Acknowledgements & Citation

- [NeuS2](https://vcai.mpi-inf.mpg.de/projects/NeuS2/)

```bibtex
@inproceedings{neus2,
    title={NeuS2: Fast Learning of Neural Implicit Surfaces for Multi-view Reconstruction}, 
    author={Wang, Yiming and Han, Qin and Habermann, Marc and Daniilidis, Kostas and Theobalt, Christian and Liu, Lingjie},
    year={2023},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)}
}
```
- [Instant-NGP](https://github.com/NVlabs/instant-ngp)

```bibtex
@article{mueller2022instant,
    author = {Thomas M\"uller and Alex Evans and Christoph Schied and Alexander Keller},
    title = {Instant Neural Graphics Primitives with a Multiresolution Hash Encoding},
    journal = {ACM Trans. Graph.},
    issue_date = {July 2022},
    volume = {41},
    number = {4},
    month = jul,
    year = {2022},
    pages = {102:1--102:15},
    articleno = {102},
    numpages = {15},
    url = {https://doi.org/10.1145/3528223.3530127},
    doi = {10.1145/3528223.3530127},
    publisher = {ACM},
    address = {New York, NY, USA},
}
```
- [NeuS](https://lingjie0206.github.io/papers/NeuS/)

```bibtex
@inproceedings{wang2021neus,
	title={NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction},
	author={Wang, Peng and Liu, Lingjie and Liu, Yuan and Theobalt, Christian and Komura, Taku and Wang, Wenping},
	booktitle={Proc. Advances in Neural Information Processing Systems (NeurIPS)},
	volume={34},
	pages={27171--27183},
	year={2021}
}
```
