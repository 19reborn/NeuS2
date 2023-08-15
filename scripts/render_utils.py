import numpy as np
import json
import os
from tqdm import tqdm

from common import *
import pyngp as ngp # noqa
from os.path import join

renderer_resolution = 1024

def load_ref_images(args, transform_path):
    with open(transform_path) as f:
        test_transforms = json.load(f)
    if os.path.isfile(args.scene):
        data_dir=os.path.dirname(args.scene)
    else:
        data_dir=args.scene
    n_camera_views = len(test_transforms["frames"])
    ref_images = []
    for camera_view in tqdm(range(n_camera_views)):
        frame = test_transforms["frames"][camera_view]
        p = frame["file_path"]  
        if "." not in p:
            p = p + ".png"
        p = p.replace("\\", "/")
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
        ref_images.append(ref_image)
    return ref_images

def load_obj_mesh(mesh_file, with_normal=False, with_texture=False):
    vertex_data = []
    norm_data = []
    uv_data = []


    face_data = []
    face_norm_data = []
    face_uv_data = []

    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file
    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
        elif values[0] == 'vn':
            vn = list(map(float, values[1:4]))
            norm_data.append(vn)
        elif values[0] == 'vt':
            vt = list(map(float, values[1:3]))
            uv_data.append(vt)

        elif values[0] == 'f':
            # quad mesh
            if len(values) > 4:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
                f = list(map(lambda x: int(x.split('/')[0]), [values[3], values[4], values[1]]))
                face_data.append(f)
            # tri mesh
            else:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
            
            # deal with texture
            if len(values[1].split('/')) >= 2:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[1]), [values[3], values[4], values[1]]))
                    face_uv_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[1]) != 0:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
            # deal with normal
            if len(values[1].split('/')) == 3:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[2]), [values[3], values[4], values[1]]))
                    face_norm_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[2]) != 0:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)

    vertices = np.array(vertex_data)
    faces = np.array(face_data) - 1

    if with_texture and with_normal:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        norms = np.array(norm_data)
        if norms.shape[0] == 0:
            norms = compute_normal(vertices, faces)
            face_normals = faces
        else:
            norms = normalize_v3(norms)
            face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals, uvs, face_uvs

    if with_texture:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        return vertices, faces, uvs, face_uvs

    if with_normal:
        norms = np.array(norm_data)
        norms = normalize_v3(norms)
        face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals

    return vertices, faces

def cal_psnr(testbed, ref_images, skip=5, save_image=False, save_dir=None, spp = 8, white_bkgd = False):
    print("calculate psnr...")
    psnr_l = []
    log_ptr = {}
    if white_bkgd:
        testbed.background_color = [1.0, 1.0, 1.0, 1.0]
    else:
        testbed.background_color = [0.0, 0.0, 0.0, 1.0]
    
    # Prior nerf papers don't typically do multi-sample anti aliasing.
    # So snap all pixels to the pixel centers.
    testbed.snap_to_pixel_centers = True
    spp = spp

    testbed.nerf.rendering_min_transmittance = 1e-4

    if save_image:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
    for camera_view, ref_image in tqdm(enumerate(ref_images)):
        if camera_view % skip != 0:
            continue
        if testbed.color_space == ngp.ColorSpace.SRGB and ref_image.shape[2] == 4:
            # Since sRGB conversion is non-linear, alpha must be factored out of it
            ref_image[...,:3] = np.divide(ref_image[...,:3], ref_image[...,3:4], out=np.zeros_like(ref_image[...,:3]), where=ref_image[...,3:4] != 0)
            ref_image[...,:3] = linear_to_srgb(ref_image[...,:3])
            ref_image[...,:3] *= ref_image[...,3:4]
            ref_image += (1.0 - ref_image[...,3:4]) * testbed.background_color
            ref_image[...,:3] = srgb_to_linear(ref_image[...,:3])
        else:
            ref_image[...,:3] *= ref_image[...,3:4]
            ref_image += (1.0 - ref_image[...,3:4]) * testbed.background_color
        

        psnr = 0
        testbed.reset_camera()
        testbed.set_camera_to_training_view(camera_view)
        image = testbed.render(ref_image.shape[1], ref_image.shape[0], spp, True)

        
        A = np.clip(linear_to_srgb(image[...,:3]), 0.0, 1.0)
        R = np.clip(linear_to_srgb(ref_image[...,:3]), 0.0, 1.0)
        mse = float(compute_error("MSE", A, R))
        psnr = mse2psnr(mse)

        if save_image:
            write_image(os.path.join(save_dir, "psnr_{:04d}.png".format(camera_view)), image)
            write_image(os.path.join(save_dir, "gt_{:04d}.png".format(camera_view)), ref_image)
        psnr_l.append(psnr)
        log_ptr[f"camera_view_{camera_view}"] = psnr
    psnr_l = np.array(psnr_l)
    avg_psnr = np.average(psnr_l)
    log_ptr["avg_psnr"] = avg_psnr
    return log_ptr, avg_psnr

def cal_psnr_old(testbed, ref_images, skip=5, log_ptr=None, save_image=False, save_dir=None, white_bkgd=False):
    print("calculate psnr...")
    psnr_l = []
    
    # testbed.background_color = [0.0, 0.0, 0.0, 0.0]
    if white_bkgd:
        testbed.background_color = [1.0, 1.0, 1.0, 1.0]
    else:
        testbed.background_color = [0.0, 0.0, 0.0, 1.0]
    
    # Prior nerf papers don't typically do multi-sample anti aliasing.
    # So snap all pixels to the pixel centers.
    testbed.snap_to_pixel_centers = True
    spp = 8

    testbed.nerf.rendering_min_transmittance = 1e-4

    if save_image:
        save_dir = os.path.join(save_dir, "evaluation", "psnr")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    for camera_view, ref_image in tqdm(enumerate(ref_images)):
        if camera_view % skip != 0:
            continue
        if testbed.color_space == ngp.ColorSpace.SRGB and ref_image.shape[2] == 4:
            # Since sRGB conversion is non-linear, alpha must be factored out of it
            ref_image[...,:3] = np.divide(ref_image[...,:3], ref_image[...,3:4], out=np.zeros_like(ref_image[...,:3]), where=ref_image[...,3:4] != 0)
            ref_image[...,:3] = linear_to_srgb(ref_image[...,:3])
            ref_image[...,:3] *= ref_image[...,3:4]
            ref_image += (1.0 - ref_image[...,3:4]) * testbed.background_color
            ref_image[...,:3] = srgb_to_linear(ref_image[...,:3])
        else:
            ref_image[...,:3] *= ref_image[...,3:4]
            ref_image += (1.0 - ref_image[...,3:4]) * testbed.background_color

        testbed.set_camera_to_training_view(camera_view)
        image = testbed.render(ref_image.shape[1], ref_image.shape[0], spp, True)
        if save_image:
            write_image(os.path.join(save_dir, "psnr_{:04d}.png".format(camera_view)), image)
            write_image(os.path.join(save_dir, "gt_{:04d}.png".format(camera_view)), ref_image)

        A = np.clip(linear_to_srgb(image[...,:3]), 0.0, 1.0)
        R = np.clip(linear_to_srgb(ref_image[...,:3]), 0.0, 1.0)
        mse = float(compute_error("MSE", A, R))
        psnr = mse2psnr(mse)
        psnr_l.append(psnr)
        # if log_ptr is not None, write psnr and camera_view to log file
        if log_ptr is not None:
            log_ptr.write('camera_view: {} psnr: {}\n'.format(camera_view, psnr))
            log_ptr.flush()
    psnr_l = np.array(psnr_l)
    avg_psnr = np.average(psnr_l)
    if log_ptr is not None:
        log_ptr.write('mean psnr: {}\n'.format(avg_psnr))
        log_ptr.flush()
    return avg_psnr    

def render_img_training_view(args, testbed, log_ptr, image_dir, frame_time_id = 0, training_step = -1):
    eval_path = args.output_path
    os.makedirs(eval_path, exist_ok=True)
    img_path = os.path.join(eval_path, 'images',f'{args.test_camera_view:04}')
    os.makedirs(img_path, exist_ok=True)
    print("Evaluating test transforms from ", args.scene, file=log_ptr)
    log_ptr.flush()
    with open(image_dir) as f:
        test_transforms = json.load(f)
    if os.path.isfile(args.scene):
        data_dir=os.path.dirname(args.scene)
    else:
        data_dir=args.scene
    totmse = 0
    totpsnr = 0
    totssim = 0
    totcount = 0
    minpsnr = 1000
    maxpsnr = 0

    # Evaluate metrics on black background
    testbed.background_color = [0.0, 0.0, 0.0, 0.0]

    # Prior nerf papers don't typically do multi-sample anti aliasing.
    # So snap all pixels to the pixel centers.
    testbed.snap_to_pixel_centers = True
    spp = 8

    testbed.nerf.rendering_min_transmittance = 1e-4

    camera_view = args.test_camera_view
    frame = test_transforms["frames"][camera_view]
    p = frame["file_path"]
    if "." not in p:
        p = p + ".png"
    p = p.replace("\\", "/")
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
    
    if training_step < 0:
        write_image(join(img_path,f"frame_{frame_time_id:06}_gt.png"), ref_image)
    else:
        os.makedirs(os.path.join(img_path, f"frame_{frame_time_id:06}"), exist_ok=True)
        write_image(join(img_path,f"frame_{frame_time_id:06}",f"frame_{frame_time_id:06}_{training_step}_gt.png"), ref_image)

    H = ref_image.shape[0]
    W = ref_image.shape[1] # original H W
    if args.render_img_HW is not None:
        # resize ref_image
        ref_image = cv2.resize(ref_image, (args.render_img_HW, args.render_img_HW), interpolation=cv2.INTER_AREA)

    testbed.reset_camera()
    testbed.set_camera_to_training_view(camera_view)
    image = testbed.render(ref_image.shape[1], ref_image.shape[0], spp, True)

    if training_step < 0:
        write_image(join(img_path,f"frame_{frame_time_id:06}_pred.png"), image)
    else:
        write_image(join(img_path,f"frame_{frame_time_id:06}",f"frame_{frame_time_id:06}_{training_step}_pred.png"), image)
    
    diffimg = np.absolute(image - ref_image)
    diffimg[...,3:4] = 1.0

    if training_step < 0:
        write_image(join(img_path,f"frame_{frame_time_id:06}_diff.png"), diffimg)
    else:
        write_image(join(img_path,f"frame_{frame_time_id:06}",f"frame_{frame_time_id:06}_{training_step}_diff.png"), diffimg)
        return
    
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

    psnr_avgmse = mse2psnr(totmse/(totcount or 1))
    psnr = totpsnr/(totcount or 1)
    ssim = totssim/(totcount or 1)
    print(f"camera_view:{camera_view}, frame_time:{frame_time_id}: PSNR={psnr} SSIM={ssim}", file=log_ptr)
    log_ptr.flush() # write immediately to file

    normal_img = None
    if args.save_mesh:
        ## render mesh normal
        mesh = trimesh.load(args.save_mesh_path)
        vex = mesh.vertices
        faces = mesh.faces
        if "lego" in data_dir:
            pose = np.array(frame['transform_matrix']).astype('float32')
            new_pose = np.array([
                [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3]],
                [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3]],
                [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3]],
                [0, 0, 0, 1],
            ], dtype=np.float32)

            ext = torch.tensor(np.linalg.inv(new_pose))

        else:
            ext = torch.inverse(torch.tensor(frame['transform_matrix']))

        if 'intrinsic_matrix' in frame:
            ixt = (torch.tensor(np.array(frame['intrinsic_matrix'])))
        else:
            cx = (test_transforms['cx']) if 'cx' in test_transforms else (H / 2)
            cy = (test_transforms['cy']) if 'cy' in test_transforms else (W / 2)
            # load intrinsics
            if 'fl_x' in test_transforms or 'fl_y' in test_transforms:
                fl_x = (test_transforms['fl_x'] if 'fl_x' in test_transforms else test_transforms['fl_y']) / downscale
                fl_y = (test_transforms['fl_y'] if 'fl_y' in test_transforms else test_transforms['fl_x']) / downscale
            elif 'camera_angle_x' in test_transforms or 'camera_angle_y' in test_transforms:
                # blender, assert in radians. already downscaled since we use H/W
                fl_x = W / (2 * np.tan(test_transforms['camera_angle_x'] / 2)) if 'camera_angle_x' in test_transforms else None
                fl_y = H / (2 * np.tan(test_transforms['camera_angle_y'] / 2)) if 'camera_angle_y' in test_transforms else None
                if fl_x is None: fl_x = fl_y
                if fl_y is None: fl_y = fl_x
            else:
                raise RuntimeError('Failed to load focal length, please check the test_transformss.json!')

            ixt = torch.tensor([[fl_x, 0, cx , 0],
                                [0, fl_y, cy, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])

        scale_ratio = torch.tensor([renderer_resolution/W, renderer_resolution/H]) 

        if args.scene[-4:] == "json":
            base_dir = os.path.dirname(args.scene) 
        else:
            base_dir = args.scene

        ixt[:2,:] = ixt[:2,:] * scale_ratio[:,None]

        renderer = Render(size = renderer_resolution)  ## to do:: support different resolution

        normal_img = render_mesh(renderer, vex.astype(np.float32), faces, ixt, ext, shaded = args.shaded_mesh)
        cv2.imwrite(join(img_path,f"frame_{frame_time_id:06}_mesh.png"), normal_img)

    return load_image(image), load_image(ref_image), normal_img

from pytorch3d_utils import *
