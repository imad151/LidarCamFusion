import numpy as np
from scipy.spatial.transform import Rotation as R

from isaacsim.sensors.camera import Camera
from typing import Optional, Tuple


class TopCameraProcessing:
    def __init__(self, top_camera: Camera):
        self._top_camera = top_camera
        self._bbox_cam = None

    def _get_camera_intrinsics(self):
        return self._top_camera.get_intrinsics_matrix()
    
    def _get_camera_extrinsics(self):
        translate, orient = self._top_camera.get_world_pose("ros")
        qw, qx, qy, qz = orient
        rot_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()
        
        T_cw = np.eye(4)
        T_cw[:3, :3] = rot_matrix
        T_cw[:3, 3] = translate
        
        return T_cw
    
    def _get_bounding_boxes_from_frame(self, frame: Optional[dict] = None):
        if frame is None:
            frame = self.capture_image()
        
        bbox = frame["bounding_box_2d_tight"]["data"][0]
        _, xmin, ymin, xmax, ymax, _ = bbox
        return np.array([xmin, ymin, xmax, ymax])
    
    def _img_to_world_3d(self, arr: np.ndarray, z_world: float = 0.8):
        K = self._get_camera_intrinsics()
        T_cw = self._get_camera_extrinsics()
        
        K_inv = np.linalg.inv(K)
        rot_matrix = T_cw[:3, :3]
        cam_pos_world = T_cw[:3, 3]
        
        arr = np.atleast_2d(arr)
        if arr.shape[1] == 2:
            arr_h = np.hstack([arr, np.ones((arr.shape[0], 1))])
        else:
            arr_h = arr
        
        rays_cam = (K_inv @ arr_h.T).T
        rays_cam_normalized = rays_cam / np.linalg.norm(rays_cam, axis=1, keepdims=True)
        
        rays_world = (rot_matrix @ rays_cam_normalized.T).T
        
        world_points = []
        for i, ray in enumerate(rays_world):
            if np.abs(ray[2]) < 1e-6:
                continue
                
            t = (z_world - cam_pos_world[2]) / ray[2]
            point = cam_pos_world + t * ray
            world_points.append(point)
        
        return np.array(world_points)
    
    def capture_image(self):
        return self._top_camera.get_current_frame(clone=True)
    
    def get_scan_end_points(self, frame: Optional[dict] = None, z_world: float = 0.8, use_model: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Returns top-left, bottom-right, and center points in world coordinates.
        
        Returns:
            tuple: (top_left_3d, bottom_right_3d, center_3d) - each as (3,) numpy arrays
        '''
        if frame is None:
            frame = self.capture_image()

        if use_model:
            ...
        
        xmin, ymin, xmax, ymax = self._get_bounding_boxes_from_frame(frame)
        self._bbox_cam = np.array([xmin, ymin, xmax, ymax])
        
        top_left = np.array([xmin - 10, ymin - 10])
        bottom_right = np.array([xmax+ 10, ymax+ 10])
        center = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])
        
        points_2d = np.stack([top_left, bottom_right, center])
        
        points_3d = self._img_to_world_3d(points_2d, z_world)
        
        top_left_3d = points_3d[0]
        bottom_right_3d = points_3d[1]
        center_3d = points_3d[2]
        
        return top_left_3d, bottom_right_3d, center_3d
    

