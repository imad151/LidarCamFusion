import numpy as np
import os

from omni.timeline import Timeline
from isaacsim.sensors.physx import RotatingLidarPhysX

class LidarScanner:
    def __init__(self,
                lidar: RotatingLidarPhysX,
                timeline: Timeline,
                save_path: str = r"C:\Users\lab\Desktop\Personal Projects\UR10_PickPlace\_Update_pipeline\data"):
        self._lidar = lidar
        self._timeline = timeline
        self._savepath = save_path

        self._frame_id = 0

    def add_current_scan(self):
        self._timeline.pause()

        frame = self._lidar.get_current_frame()
        pc = frame["point_cloud"]

        curr_pos, curr_orient = self._get_lidar_extrinsics()

        data_dict = {
            "pos": curr_pos,
            "orient": curr_orient,
            "pointcloud": pc
        }

        os.makedirs(self._savepath, exist_ok=True)
        np.savez(os.path.join(self._savepath, f"pointcloud_{self._frame_id}.npz"), **data_dict)
        self._frame_id += 1
        self._timeline.play()

        return True
    
    def _get_lidar_extrinsics(self):
        return self._lidar.get_world_pose()