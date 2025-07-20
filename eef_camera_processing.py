import numpy as np
import os
from isaacsim.sensors.camera import Camera


class EEfCameraProcessing:
    def __init__(self,
                eef_camera: Camera,
                save_path: str):
        self._camera = eef_camera
        self._save_path = save_path
        self._frame_id = 0

    def _capture_image(self):
        return self._camera.get_rgb()

    def save_camera_intrinsics(self):
        np.savez(os.path.join(self._save_path, "eef_cam_intrinsics"), K=self._camera.get_intrinsics_matrix())
    
    def _get_camera_extrinsics(self):
        return self._camera.get_world_pose()
    
    def save_image(self):
        frame = self._capture_image()

        curr_img_path = os.path.join(self._save_path, f"rgb_{self._frame_id}")

        pos, orient = self._get_camera_extrinsics()

        frame_dict = {
            "pos": pos,
            "orient": orient,
            "rgb": frame
        }

        np.savez(curr_img_path, **frame_dict)
        print("saved frame id", self._frame_id)

        self._frame_id += 1