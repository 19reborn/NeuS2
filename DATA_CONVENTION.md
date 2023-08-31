# Data Convention

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
			"transform_matrix": [ # specify a camera to world transform (i.e. the inverse of the extrinsic matrix), use meters as the unit (shape: [4, 4])
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
