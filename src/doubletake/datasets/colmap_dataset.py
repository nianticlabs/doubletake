import functools
import logging
import os

import numpy as np
import PIL.Image as pil
import torch
from torchvision import transforms

from doubletake.datasets.generic_mvs_dataset import GenericMVSDataset
from doubletake.utils.generic_utils import (
    crop_or_pad,
    fov_to_image_dimension,
    read_image_file,
)
from doubletake.utils.geometry_utils import qvec2rotmat, rotx

logger = logging.getLogger(__name__)


class ColmapDataset(GenericMVSDataset):
    """
    Reads COLMAP undistored images and poses from a text based sparse COLMAP
    reconstruction.

    self.capture_poses is a dictionary indexed with a scan's id and is populated
    with a scan's pose information when a frame is loaded from that scan.

    This class expects each scan's directory to have the following hierarchy:
    
    dataset_path:
        scans.txt (contains list of scans, you can define a different filepath)
        tuples (dir where you store tuples, you can define a different directory)
        scans:
            scan_1:
                images:
                    img1.jpg (undistored image from COLMAP)
                    img2.jpg
                    ...
                    imgN.jpg
                sparse:
                    0:
                        cameras.txt: SIMPLE_PINHOLE camera text file with intrinsics.
                        images.txt: text file output with image poses.
                valid_frames.txt (generated when you run tuple scripts)
                scale.txt a text file with the scale of the scene.
                
                
    The `scale.txt` file should contain a scale factor to go from COLMAP's coords 
    to real metric coords. You can get this by dividing by the real size of an item 
    by the measured size of that item in the COLMAP point cloud. 
    
    This class does not load depth, instead returns dummy data.
    
    Set `modify_to_fov` to True to crop images to an fov of [58.18, 45.12]. 

    Inherits from GenericMVSDataset and implements missing methods.
    """

    def __init__(
        self,
        dataset_path,
        split,
        mv_tuple_file_suffix,
        include_full_res_depth=False,
        limit_to_scan_id=None,
        num_images_in_tuple=None,
        color_transform=transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
        tuple_info_file_location=None,
        image_height=384,
        image_width=512,
        high_res_image_width=640,
        high_res_image_height=480,
        image_depth_ratio=2,
        shuffle_tuple=False,
        include_full_depth_K=False,
        include_high_res_color=False,
        pass_frame_id=False,
        skip_frames=None,
        skip_to_frame=None,
        verbose_init=True,
        native_depth_width=640,
        native_depth_height=480,
        fill_depth_hints=False,
        load_empty_hints=False,
        depth_hint_aug=0.0,
        depth_hint_dir=None,
        disable_flip=False,
        rotate_images=False,
        modify_to_fov=False,
    ):
        super().__init__(
            dataset_path=dataset_path,
            split=split,
            mv_tuple_file_suffix=mv_tuple_file_suffix,
            include_full_res_depth=include_full_res_depth,
            limit_to_scan_id=limit_to_scan_id,
            num_images_in_tuple=num_images_in_tuple,
            color_transform=color_transform,
            tuple_info_file_location=tuple_info_file_location,
            image_height=image_height,
            image_width=image_width,
            high_res_image_width=high_res_image_width,
            high_res_image_height=high_res_image_height,
            image_depth_ratio=image_depth_ratio,
            shuffle_tuple=shuffle_tuple,
            include_full_depth_K=include_full_depth_K,
            include_high_res_color=include_high_res_color,
            pass_frame_id=pass_frame_id,
            skip_frames=skip_frames,
            skip_to_frame=skip_to_frame,
            verbose_init=verbose_init,
            native_depth_width=native_depth_width,
            native_depth_height=native_depth_height,
            fill_depth_hints=fill_depth_hints,
            load_empty_hints=load_empty_hints,
            depth_hint_dir=depth_hint_dir,
            depth_hint_aug=depth_hint_aug,
            disable_flip=disable_flip,
            rotate_images=rotate_images,
        )

        """
        Args:
            dataset_path: base path to the dataaset directory.
            split: the dataset split.
            mv_tuple_file_suffix: a suffix for the tuple file's name. The 
                tuple filename searched for wil be 
                {split}{mv_tuple_file_suffix}.
            tuple_info_file_location: location to search for a tuple file, if 
                None provided, will search in the dataset directory under 
                'tuples'.
            limit_to_scan_id: limit loaded tuples to one scan's frames.
            num_images_in_tuple: optional integer to limit tuples to this number
                of images.
            image_height, image_width: size images should be loaded at/resized 
                to. 
            include_high_res_color: should the dataset pass back higher 
                resolution images.
            high_res_image_height, high_res_image_width: resolution images 
                should be resized if we're passing back higher resolution 
                images.
            image_depth_ratio: returned gt depth maps "depth_b1hw" will be of 
                size (image_height, image_width)/image_depth_ratio.
            include_full_res_depth: if true will return depth maps from the 
                dataset at the highest resolution available.
            color_transform: optional color transform that applies when split is
                "train".
            shuffle_tuple: by default source images will be ordered according to 
                overall pose distance to the reference image. When this flag is
                true, source images will be shuffled. Only used for ablation.
            pass_frame_id: if we should return the frame_id as part of the item 
                dict
            skip_frames: if not none, will stride the tuple list by this value.
                Useful for only fusing every 'skip_frames' frame when fusing 
                depth.
            verbose_init: if True will let the init print details on the 
                initialization.
            native_depth_width, native_depth_height: the defaults here match the 
                scan processing functions used in this module. We've used fixed
                hardcoded values for the native depth resolution for loading 
                correct intrinsics, but this can definitely be automated.
            image_resampling_mode: resampling method for resizing images.
        
        """

        self.capture_poses = {}
        
        self.modify_to_fov = modify_to_fov

        self.target_fov = None
        if self.modify_to_fov:
            self.target_fov = [58.18, 45.12]

        self.image_resampling_mode = pil.BILINEAR

    @staticmethod
    def get_sub_folder_dir(split):
        return "scans"

    def get_frame_id_string(self, frame_id):
        """Returns an id string for this frame_id that's unique to this frame
        within the scan.

        This string is what this dataset uses as a reference to store files
        on disk.
        """
        return frame_id

    def get_valid_frame_path(self, split, scan):
        """returns the filepath of a file that contains valid frame ids for a
        scan."""

        scan_dir = os.path.join(self.dataset_path, self.get_sub_folder_dir(split), scan)

        return os.path.join(scan_dir, "valid_frames.txt")

    def get_valid_frame_ids(self, split, scan, store_computed=True):
        """Either loads or computes the ids of valid frames in the dataset for
        a scan.

        A valid frame is one that has an existing RGB frame, an existing
        depth file, and existing pose file where the pose isn't inf, -inf,
        or nan.

        Args:
            split: the data split (train/val/test)
            scan: the name of the scan
            store_computed: store the valid_frame file where we'd expect to
            see the file in the scan folder. get_valid_frame_path defines
            where this file is expected to be. If the file can't be saved,
            a warning will be printed and the exception reason printed.

        Returns:
            valid_frames: a list of strings with info on valid frames.
            Each string is a concat of the scan_id and the frame_id.
        """
        scan = scan.rstrip("\n")
        valid_frame_path = self.get_valid_frame_path(split, scan)

        if os.path.exists(valid_frame_path):
            # valid frame file exists, read that to find the ids of frames with
            # valid poses.
            with open(valid_frame_path) as f:
                valid_frames = f.readlines()
        else:
            print(f"Compuiting valid frames for scene {scan}.")
            # find out which frames have valid poses

            # load capture poses for this scan
            self.load_capture_poses(scan)

            bad_file_count = 0
            dist_to_last_valid_frame = 0
            valid_frames = []
            for frame_id in sorted(self.capture_poses[scan]):
                world_T_cam_44, _ = self.load_pose(scan, frame_id)

                if (
                    np.isnan(np.sum(world_T_cam_44))
                    or np.isinf(np.sum(world_T_cam_44))
                    or np.isneginf(np.sum(world_T_cam_44))
                ):
                    bad_file_count += 1
                    dist_to_last_valid_frame += 1
                    continue

                valid_frames.append(f"{scan} {frame_id} {dist_to_last_valid_frame}")
                dist_to_last_valid_frame = 0

            print(
                f"Scene {scan} has {bad_file_count} bad frame files out of "
                f"{len(self.capture_poses[scan])}."
            )

            # store computed if we're being asked, but wrapped inside a try
            # incase this directory is read only.
            if store_computed:
                # store those files to valid_frames.txt
                try:
                    with open(valid_frame_path, "w") as f:
                        f.write("\n".join(valid_frames) + "\n")
                except Exception as e:
                    print(f"Couldn't save valid_frames at {valid_frame_path}, " f"cause:")
                    print(e)

        return valid_frames

    @functools.cache
    def load_pose(self, scan_id, frame_id):
        """Loads a frame's pose file.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            world_T_cam (numpy array): matrix for transforming from the
                camera to the world (pose).
            cam_T_world (numpy array): matrix for transforming from the
                world to the camera (extrinsics).

        """
        if scan_id not in self.capture_poses:
            self.load_capture_poses(scan_id)

        scene_path = os.path.join(
            self.dataset_path, self.get_sub_folder_dir(self.split), scan_id,
        )
        scale_file = os.path.join(scene_path, "scale.txt")
        with open(scale_file, "r") as f:
            scale = float(f.readline().strip())

        pose_info = self.capture_poses[scan_id][frame_id]
        world_T_cam = pose_info["world_T_cam"]

        rot_mat = world_T_cam[:3, :3]
        trans = world_T_cam[:3, 3]

        rot_mat = rotx(-np.pi / 2) @ rot_mat
        trans = rotx(-np.pi / 2) @ trans

        world_T_cam[:3, :3] = rot_mat
        world_T_cam[:3, 3] = trans * scale

        world_T_cam = world_T_cam
        cam_T_world = np.linalg.inv(world_T_cam)

    
        return world_T_cam, cam_T_world

    def get_target_fov_hw(self, scan_id):
        """
        Returns the target fov crop height and width for a scan.
        args:
            scan_id: the scan this file belongs to.
        returns:
            target_fov_crop_hw: tuple of the target fov crop height and width.
        """


        scene_path = os.path.join(
            self.dataset_path, self.get_sub_folder_dir(self.split), scan_id, "sparse", "0",
        )

        with open(os.path.join(scene_path, "cameras.txt"), "r") as f:
            for line in f:
                if line[0] == "#":
                    continue
                els = line.split(" ")
                w = float(els[2])
                h = float(els[3])
                fl_x = float(els[4])
                fl_y = float(els[4])
                k1 = 0
                k2 = 0
                p1 = 0
                p2 = 0
                cx = w / 2
                cy = h / 2
                if els[1] == "SIMPLE_PINHOLE":
                    cx = float(els[5])
                    cy = float(els[6])
                elif els[1] == "PINHOLE":
                    fl_y = float(els[5])
                    cx = float(els[6])
                    cy = float(els[7])
                elif els[1] == "SIMPLE_RADIAL":
                    cx = float(els[5])
                    cy = float(els[6])
                    k1 = float(els[7])
                elif els[1] == "RADIAL":
                    cx = float(els[5])
                    cy = float(els[6])
                    k1 = float(els[7])
                    k2 = float(els[8])
                elif els[1] == "OPENCV":
                    fl_y = float(els[5])
                    cx = float(els[6])
                    cy = float(els[7])
                    k1 = float(els[8])
                    k2 = float(els[9])
                    p1 = float(els[10])
                    p2 = float(els[11])
                else:
                    print("unknown camera model ", els[1])

        # images are assumed undistored, so using simple pinhole.
        fx = fl_x
        fy = fl_y

        new_width = int(np.round(fov_to_image_dimension(self.target_fov[0], fx)))
        new_height = int(np.round(fov_to_image_dimension(self.target_fov[1], fy)))
        target_fov_crop_hw = (new_height, new_width)

        return target_fov_crop_hw

    def load_intrinsics(self, scan_id, frame_id=None, flip=None):
        """Loads intrinsics, computes scaled intrinsics, and returns a dict
        with intrinsics matrices for a frame at multiple scales.

        This function assumes all images have the same intrinsics and
        doesn't handle per image intrinsics from COLMAP

        Images are assumed undistored, so using simple pinhole.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.
            flip: unused

        Returns:
            output_dict: A dict with
                - K_s{i}_b44 (intrinsics) and invK_s{i}_b44
                (backprojection) where i in [0,1,2,3,4]. i=0 provides
                intrinsics at the scale for depth_b1hw.
                - K_full_depth_b44 and invK_full_depth_b44 provides
                intrinsics for the maximum available depth resolution.
                Only provided when include_full_res_depth is true.

        """
        output_dict = {}

        scene_path = os.path.join(
            self.dataset_path, self.get_sub_folder_dir(self.split), scan_id, "sparse", "0",
        )

        with open(os.path.join(scene_path, "cameras.txt"), "r") as f:
            for line in f:
                if line[0] == "#":
                    continue
                els = line.split(" ")
                w = float(els[2])
                h = float(els[3])
                fl_x = float(els[4])
                fl_y = float(els[4])
                k1 = 0
                k2 = 0
                p1 = 0
                p2 = 0
                cx = w / 2
                cy = h / 2
                if els[1] == "SIMPLE_PINHOLE":
                    cx = float(els[5])
                    cy = float(els[6])
                elif els[1] == "PINHOLE":
                    fl_y = float(els[5])
                    cx = float(els[6])
                    cy = float(els[7])
                elif els[1] == "SIMPLE_RADIAL":
                    cx = float(els[5])
                    cy = float(els[6])
                    k1 = float(els[7])
                elif els[1] == "RADIAL":
                    cx = float(els[5])
                    cy = float(els[6])
                    k1 = float(els[7])
                    k2 = float(els[8])
                elif els[1] == "OPENCV":
                    fl_y = float(els[5])
                    cx = float(els[6])
                    cy = float(els[7])
                    k1 = float(els[8])
                    k2 = float(els[9])
                    p1 = float(els[10])
                    p2 = float(els[11])
                else:
                    print("unknown camera model ", els[1])

        # images are assumed undistored, so using simple pinhole.
        fx = fl_x
        fy = fl_y

        py = cy
        px = cx
        int_width = w
        int_height = h
        
        K = torch.eye(4, dtype=torch.float32)
        K[0, 0] = float(fx)
        K[1, 1] = float(fy)
        K[0, 2] = float(px)
        K[1, 2] = float(py)
        # print("K before crop: ", K)

        if self.modify_to_fov:
            target_fov_crop_hw = self.get_target_fov_hw(scan_id)

            # centre crop the intrinsics
            left = (float(int_width) - target_fov_crop_hw[1]) / 2
            top = (float(int_width) - target_fov_crop_hw[0]) / 2

            K[0, 2] -= left
            K[1, 2] -= top

            # normalize intrinsics to the dataset's configured color resolution, taking into account the target fov.
            K[0] /= target_fov_crop_hw[1]
            K[1] /= target_fov_crop_hw[0]
            K[0, 2] = 0.5
            K[1, 2] = 0.5
        else:
            K[0] /= int_width
            K[1] /= int_height
        # print("K after crop: ", K)
        

        if self.include_full_depth_K:
            K_full_res = K.clone()
            K_full_res[0] *= self.native_depth_width
            K_full_res[1] *= self.native_depth_height

            output_dict[f"K_full_depth_b44"] = K_full_res
            output_dict[f"invK_full_depth_b44"] = torch.linalg.inv(K_full_res)

        # scale intrinsics to the dataset's configured depth resolution.
        K[0] *= self.depth_width
        K[1] *= self.depth_height

        for i in range(5):
            K_scaled = K.clone()
            K_scaled[:2] /= 2**i
            invK_scaled = np.linalg.inv(K_scaled)
            output_dict[f"K_s{i}_b44"] = K_scaled
            output_dict[f"invK_s{i}_b44"] = invK_scaled

        return output_dict

    def load_capture_poses(self, scan_id):
        """Loads in poses for a scan in COLMAP format. Saves these to the
        self.capture_poses dictionary under the key scan_id

        Args:
            scan_id: the id of the scan whose poses will be loaded.
        """

        if scan_id in self.capture_poses:
            return

        scene_path = os.path.join(
            self.dataset_path, self.get_sub_folder_dir(self.split), scan_id, "sparse", "0",
        )

        self.capture_poses[scan_id] = {}

        with open(os.path.join(scene_path, "images.txt"), "r") as f:
            i = 0
            bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])

            for line in f:
                line = line.strip()
                if line[0] == "#":
                    continue
                i = i + 1

                if i % 2 == 1:
                    # 1-4 is quat, 5-7 is trans, 9ff is filename
                    # (9, if filename contains no spaces)
                    elems = line.split(" ")

                    # remove the extension
                    image_id = "".join(elems[9:]).strip().split(".")[0]

                    qvec = np.array(tuple(map(float, elems[1:5])))
                    tvec = np.array(tuple(map(float, elems[5:8])))
                    R = qvec2rotmat(-qvec)
                    t = tvec.reshape([3, 1])

                    m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
                    world_T_cam = np.linalg.inv(m)

                    cam_T_world = np.linalg.inv(world_T_cam)

                    poses = {}
                    poses["world_T_cam"] = world_T_cam
                    poses["cam_T_world"] = cam_T_world
                    self.capture_poses[scan_id][image_id] = poses

    def load_target_size_depth_and_mask(self, scan_id, frame_id):
        """Loads a depth map at the resolution the dataset is configured for.

        This function is not implemented for Scanniverse
        """
        depth = torch.ones((1, self.depth_height, self.depth_width), dtype=torch.float32)
        mask = torch.ones((1, self.depth_height, self.depth_width), dtype=torch.float32)
        mask_b = torch.ones((1, self.depth_height, self.depth_width), dtype=torch.bool)

        return depth, mask, mask_b

    def load_full_res_depth_and_mask(self, scan_id, frame_id):
        """Loads a depth map at the native resolution the dataset provides.

        This function is not implemented for Scanniverse
        """

        full_res_depth = torch.ones(
            (1, self.native_depth_height, self.native_depth_width), dtype=torch.float32
        )
        full_res_mask = torch.ones(
            (1, self.native_depth_height, self.native_depth_width), dtype=torch.float32
        )
        full_res_mask_b = torch.ones(
            (1, self.native_depth_height, self.native_depth_width), dtype=torch.bool
        )

        return full_res_depth, full_res_mask, full_res_mask_b

    def get_cached_depth_filepath(self, scan_id, frame_id):
        """returns the filepath for a frame's depth file at the dataset's
        configured depth resolution.

        This function is not implemented for Scanniverse

        """

        return ""

    def get_full_res_depth_filepath(self, scan_id, frame_id):
        """returns the filepath for a frame's depth file at the native
        resolution in the dataset.

        This function is not implemented for Scanniverse
        """
        return ""

    def get_color_filepath(self, scan_id, frame_id):
        """returns the filepath for a frame's color file at the dataset's
        configured RGB resolution.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            Either the filepath for a precached RGB file at the size
            required, or if that doesn't exist, the full size RGB frame
            from the dataset.

        """
        scene_path = os.path.join(
            self.dataset_path, self.get_sub_folder_dir(self.split), scan_id,
        )
        color_path = os.path.join(scene_path, "images", f"{frame_id}.JPG")

        if os.path.exists(color_path):
            return color_path

        # high res isn't available, so load in normal res.
        return os.path.join(scene_path, "images", f"{frame_id}.JPG")

    def get_high_res_color_filepath(self, scan_id, frame_id):
        """returns the filepath for a frame's higher res color file at the
        dataset's configured high RGB resolution.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            Either the filepath for a precached RGB file at the high res
            size required, or if that doesn't exist, the full size RGB frame
            from the dataset.

        """
        scene_path = os.path.join(
            self.dataset_path, self.get_sub_folder_dir(self.split), scan_id, 
        )
        return os.path.join(scene_path, "images", f"{frame_id}.JPG")

    def read_fov_cropped_image_file(
        self,
        color_filepath,
        target_fov_crop_h,
        target_fov_crop_w,
        target_h,
        target_w,
        resampling_mode,
        value_scale_factor=1.0,
        pad_mode="constant",
    ):
        """
        Reads the image file, resizes it to native_h, native_w, crops it to
        target_fov_crop_h, target_fov_crop_w, and then resizes it to
        target_h, target_w.
        args:
            color_filepath: the filepath for the color image.
            native_h, native_w: the height and width of the native image.
            target_fov_crop_h, target_fov_crop_w: the height and width of the
                cropped image.
            target_h, target_w: the height and width of the resized image.
            resampling_mode: the resampling mode for the image.
            value_scale_factor: the scale factor for the image values.
            pad_mode: the padding mode for the image.
        returns:
            image: the resized and cropped image.
        """

        image = read_image_file(
            color_filepath,
            height=None,
            width=None,
            resampling_mode=resampling_mode,
            value_scale_factor=value_scale_factor,
            disable_warning=True,
        )
        image_bchw = image.unsqueeze(0).numpy()
        # print(f"image shape: {image_bchw.shape}")
        cropped_image_bchw = crop_or_pad(
            image_bchw, target_fov_crop_h, target_fov_crop_w, pad_mode=pad_mode
        )
        # print(f"target_fov_crop_hw: {target_fov_crop_h, target_fov_crop_w}")
        # print(f"cropped image shape: {cropped_image_bchw.shape}")
        if resampling_mode == pil.NEAREST:
            mode = "nearest"
        elif resampling_mode == pil.BILINEAR:
            mode = "bilinear"
        else:
            raise ValueError(f"resampling mode {self.image_resampling_mode} not supported")
        image = torch.nn.functional.interpolate(
            torch.tensor(cropped_image_bchw), (target_h, target_w), mode=mode
        ).squeeze(0)
        # print(f"resized image shape: {image.shape}")
        return image

    @functools.cache
    def load_color(self, scan_id, frame_id):
        """Loads a frame's RGB file, resizes it to configured RGB size. Also,
        crops images to satisfy aspect ratio.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            iamge: tensor of the resized RGB image at self.image_height and
            self.image_width resolution.

        """

        color_filepath = self.get_color_filepath(scan_id, frame_id)

        if self.modify_to_fov:
            target_fov_crop_hw = self.get_target_fov_hw(scan_id)
            
            image = self.read_fov_cropped_image_file(
                color_filepath=color_filepath,
                target_fov_crop_h=target_fov_crop_hw[0],
                target_fov_crop_w=target_fov_crop_hw[1],
                target_h=self.image_height,
                target_w=self.image_width,
                resampling_mode=self.image_resampling_mode,
            )
        else:
            image = read_image_file(
                color_filepath,
                height=self.image_height,
                width=self.image_width,
                resampling_mode=self.image_resampling_mode,
                disable_warning=self.disable_resize_warning,
                target_aspect_ratio=4 / 3,
            )

        return image

    @functools.cache
    def load_high_res_color(self, scan_id, frame_id):
        """Loads a frame's RGB file at a high resolution as configured.

        NOTE: Usually images in COLMAP dataset world are very large, so this
        function will default to the standard sized images if available.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            iamge: tensor of the resized RGB image at
            self.high_res_image_height and self.high_res_image_width
            resolution.

        """
        # not get_high_res_color_filepath for speed
        return self.load_color(scan_id, frame_id)


    def load_depth_hint(self, scan_id, frame_id, flip=False, mark_all_empty=False):
        """Loads a depth hint for a frame if it exists.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.
            flip: if the hint should be flipped along x.
            mark_all_empty: if the hint should be marked as empty.

        Returns:
            depth_hint_dict: depth hint dict.
        """
        depth_hint_dict = {}

        depth_hint_1hw = torch.zeros(1, self.depth_height, self.depth_width)
        depth_hint_1hw[:] = torch.nan
        depth_hint_mask_1hw = torch.zeros_like(depth_hint_1hw)
        depth_hint_mask_b_1hw = torch.zeros_like(depth_hint_1hw).bool()
        sampled_weights_1hw = torch.zeros_like(depth_hint_1hw)

        depth_hint_dict["depth_hint_b1hw"] = depth_hint_1hw
        depth_hint_dict["depth_hint_mask_b1hw"] = depth_hint_mask_1hw
        depth_hint_dict["depth_hint_mask_b_b1hw"] = depth_hint_mask_b_1hw
        depth_hint_dict["sampled_weights_b1hw"] = sampled_weights_1hw

        return depth_hint_dict