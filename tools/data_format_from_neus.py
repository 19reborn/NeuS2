import json
from os.path import join
import numpy as np
import os
import cv2
import torch
from glob import glob
import shutil


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.conf = conf

        self.data_dir = conf['data_dir']
        self.render_cameras_name = conf['render_cameras_name']
        self.object_cameras_name = conf['object_cameras_name']

        self.camera_outside_sphere = True
        self.scale_mat_scale = 1.1

        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict
        self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.png')))
        self.n_images = len(self.images_lis)

        # world_mat is a projection matrix from world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.scale_mats_np = []

        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []

        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.intrinsics_all = torch.stack(self.intrinsics_all)   # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all)  # [n_images, 4, 4]

        print('Load data: End')


def generate(dataset_name, base_par_dir, copy_image=True, is_downsample=False, downsample_scale=1, fixed_camera=True, wrong_camera=[]):
    assert is_downsample == False, "Not implemented"

    base_dir = os.path.join(base_par_dir, dataset_name)
    output_dir = f'./data/neus/{dataset_name}'

    conf = {
        "data_dir": base_dir,
        "render_cameras_name": "cameras_sphere.npz",
        "object_cameras_name": "cameras_sphere.npz",
    }
    dataset = Dataset(conf)
    image_name = 'image'
    mask_name = 'mask'
    test_view = [8, 13, 16, 21, 26, 31, 34, 56]

    os.makedirs(output_dir, exist_ok=True)

    base_rgb_dir = join(base_dir,image_name)
    base_msk_dir = join(base_dir, mask_name)
    all_images = sorted(os.listdir(base_rgb_dir))
    all_masks = sorted(os.listdir(base_msk_dir))
    assert len(all_images) == len(all_masks)
    print("#images:", len(all_images))

    def copy_directories(root_src_dir, root_dst_dir):
        for src_dir, dirs, files in os.walk(root_src_dir):
            dst_dir = src_dir.replace(root_src_dir, root_dst_dir, 1)
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            for file_ in files:
                src_file = os.path.join(src_dir, file_)
                dst_file = os.path.join(dst_dir, file_)
                if os.path.exists(dst_file):
                    os.remove(dst_file)
                shutil.copy(src_file, dst_dir)

    H, W = 1200, 1600

    if copy_image:
        new_image_dir = join(output_dir, "images")
        os.makedirs(new_image_dir, exist_ok=True)
        for i in range(len(all_images)):
            img_name = all_images[i]
            msk_name = all_masks[i]
            img_path = join(base_rgb_dir, img_name)
            msk_path = join(base_msk_dir, msk_name)
            # print("copy", img_path, msk_path)
            img = cv2.imread(img_path)
            msk = cv2.imread(msk_path, 0)
            image = np.concatenate([img,msk[:,:,np.newaxis]],axis=-1)
            H , W = image.shape[0], image.shape[1]
            H , W = image.shape[0], image.shape[1]
            cv2.imwrite(join(new_image_dir, img_name), image)
        print("Copy images done")
    
    base_rgb_dir = "images"
    print("base_rgb_dir:", base_rgb_dir)

    output = {
        "w": W,
        "h": H,
        "aabb_scale": 1.0,
        "scale": 0.5,
        "offset": [ # neus: [-1,1] ngp[0,1]
            0.5,
            0.5,
            0.5
        ],
        "from_na": True,
    }
    
    for frame_i in range(1):
        output['frames'] = []
        all_rgb_dir = sorted(os.listdir(join(output_dir,base_rgb_dir)))
        rgb_num = len(all_rgb_dir)
        camera_num = dataset.intrinsics_all.shape[0]
        assert rgb_num == camera_num, "The number of cameras should be eqaul to the number of images!"
        for i in range(rgb_num):
            if i in wrong_camera: # this camera goes wrong
                continue
            rgb_dir = join(base_rgb_dir, all_rgb_dir[i])
            ixt = dataset.intrinsics_all[i]

            # add one_frame
            one_frame = {}
            one_frame["file_path"] = rgb_dir
            one_frame["transform_matrix"] = dataset.pose_all[i].tolist()

            one_frame["intrinsic_matrix"] = ixt.tolist()

            if i not in test_view:
                output['frames'].append(one_frame)

        file_dir = join(output_dir, f'transform_train.json')
        with open(file_dir,'w') as f:
            json.dump(output, f, indent=4)
            
    output_test = {
        "w": W,
        "h": H,
        "aabb_scale": 1.0,
        "scale": 0.5,
        "offset": [ # neus: [-1,1] ngp[0,1]
            0.5,
            0.5,
            0.5
        ],
        "from_na": True,
    }
    
    for frame_i in range(1):
        # init_params(output)
        output_test['frames'] = []
        all_rgb_dir = sorted(os.listdir(join(output_dir,base_rgb_dir)))
        rgb_num = len(all_rgb_dir)
        camera_num = dataset.intrinsics_all.shape[0]
        assert rgb_num == camera_num, "The number of cameras should be eqaul to the number of images!"
        for i in range(rgb_num):
            if i in wrong_camera: # this camera goes wrong
                continue
            rgb_dir = join(base_rgb_dir, all_rgb_dir[i])
            ixt = dataset.intrinsics_all[i]

            # add one_frame
            one_frame = {}
            one_frame["file_path"] = rgb_dir
            one_frame["transform_matrix"] = dataset.pose_all[i].tolist()

            one_frame["intrinsic_matrix"] = ixt.tolist()

            if i in test_view:
                output_test['frames'].append(one_frame)

        file_dir = join(output_dir, f'transform_test.json')
        with open(file_dir,'w') as f:
            json.dump(output_test, f, indent=4)


if __name__ == "__main__":
    base_par_dir = "/home/downloads/data_DTU/"
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_all', action="store_true")
    parser.add_argument('--dataset_name', type=str, default='dtu_scan97')
    parser.add_argument("--copy_image", action="store_true")
    args = parser.parse_args()
    
    if args.dataset_all:
        for dataset_name in os.listdir(base_par_dir):
            print("dataset_name:", dataset_name)
            generate(dataset_name, base_par_dir, args.copy_image)
    else:
        generate(args.dataset_name, base_par_dir, args.copy_image)
    
    
    
    
    