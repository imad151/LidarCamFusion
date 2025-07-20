from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": False})
import omni.usd
omni.usd.get_context().open_stage(r"C:\Users\lab\Desktop\Personal Projects\UR10_PickPlace\Stage\New_Stage.usd")

import os
from isaacsim.core.api import World
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.sensors.camera import Camera

from world_init import UR10World
from rmpflow_controller import UR10RmpFlow
from top_camera_processing import TopCameraProcessing
from lidar_scanner import LidarScanner
from eef_camera_processing import EEfCameraProcessing
from GeneratePointCloud import PointCloudProcessor


class MainLoop:
    def __init__(self):
        self._save_path = r"C:\Users\lab\Desktop\Personal Projects\UR10_PickPlace\_Update_pipeline\data"    # MUST CHANGE TO WHERE YOU WANT TO SAVE IT
        os.makedirs(self._save_path, exist_ok= True)

        self._point_cloud_processor = PointCloudProcessor(self._save_path)

        self._ur10_world = UR10World()
        self._ur10_world.post_init()

        self._my_world: World = self._ur10_world.my_world
        self._my_stage = self._ur10_world.my_stage
        self._my_manipulator: SingleManipulator = self._ur10_world.manipulator
        self._top_camera: Camera = self._ur10_world.top_camera

        self._ur10_rmpflow = UR10RmpFlow(robot_articulation=self._my_manipulator)
        self._top_camera_processing = TopCameraProcessing(top_camera=self._top_camera)
        self._eef_camera_processing = EEfCameraProcessing(eef_camera=self._ur10_world.eef_camera, save_path=self._save_path)

        self._arc_points = []
        self._arc_orient = []
        self._is_scanning = False

        self._physics_dt = self._my_world.get_physics_dt()
        self._t = 0
        self._events_dt = [0.01, 0.1, 0.005, 0.08, 0.0001]  # events loop
        self._current_state = 0

        self._lidar_scanner = LidarScanner(
            lidar=self._ur10_world.lidar,
            timeline=self._ur10_world.timeline,
            save_path=self._save_path 
        )

    def start_simulation(self):
        while sim_app.is_running():
            if self._my_world.is_playing():
                if self._my_world.current_time_step_index == 0:
                    self._my_world.reset()
                    break

                self._my_world.step(render=True)
                sim_app.update()
                self._ur10_rmpflow.update()
                
                if self._current_state == 0:
                    print("Init State Func")
                    if self._init_state_func():
                        self._current_state += 1
                        self._t = 0
                elif self._current_state == 1:
                    print("Top Cam Func")
                    if self._top_cam_state_func():
                        self._current_state += 1
                        self._t = 0
                elif self._current_state == 2:
                    self._move_to_obj_func()
                elif self._current_state == 3:
                    self._scan_1_state_func()
                elif self._current_state == 4:
                    self._make_pointcloud_func()
                    break

                self._t += self._events_dt[self._current_state]
                if self._t >= 1:
                    self._current_state += 1
                    self._t = 0
    
    def _init_state_func(self):
        for _ in range(30):
            self._my_world.step()

        self._eef_camera_processing.save_camera_intrinsics()
        
        return True
    
    def _top_cam_state_func(self):
        import numpy as np
        frame = self._top_camera_processing.capture_image()
        self.right, self.left, self.center = self._top_camera_processing.get_scan_end_points(frame)
        self._arc_points = self._ur10_rmpflow.get_arc_points(self.center, self.left, self.right)
        self._arc_orient = self._ur10_rmpflow.compute_look_at_quaternions(self._arc_points, self.center)
        
        return True

    def _move_to_obj_func(self):
        print("Move to obj State")
        self._ur10_rmpflow.update_target(
            target_end_effector_orientation=self._arc_orient[0],
            target_end_effector_position=self._arc_points[0]
        )
    
    def _scan_1_state_func(self):
        print("scanning state")
        
        if not hasattr(self, '_reverse_scan'):
            self._reverse_scan = False
        
        if self._reverse_scan:
            pos_orient_pairs = list(zip(reversed(self._arc_points), reversed(self._arc_orient)))
        else:
            pos_orient_pairs = list(zip(self._arc_points, self._arc_orient))
        
        for pos, orient in pos_orient_pairs:
            self._ur10_rmpflow.update_target(
                target_end_effector_position=pos,
                target_end_effector_orientation=orient
            )
            self._add_scan_func()
            self._add_img_func()
            
            self._ur10_rmpflow.update()
            self._my_world.step()
        
        self._reverse_scan = not self._reverse_scan
    
    def _make_pointcloud_func(self):
        print("generating pointcloud now")
        self._point_cloud_processor.save_pointcloud(output_path=os.path.join(self._save_path, "pointcloud.ply"))
        self._point_cloud_processor.process_and_save()
        self._point_cloud_processor.get_screws_locations()

    def _add_scan_func(self):
        self._lidar_scanner.add_current_scan()
    
    def _add_img_func(self):
        self._eef_camera_processing.save_image()

if __name__ == "__main__":
    main_loop = MainLoop()
    main_loop.start_simulation()
    sim_app.close()
                
