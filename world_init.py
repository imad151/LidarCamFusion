import omni.usd
import omni.timeline
from isaacsim.core.api import World
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.sensors.camera import Camera
from isaacsim.sensors.physx import RotatingLidarPhysX

import numpy as np

class UR10World:
    def __init__(self):
        self.my_stage = omni.usd.get_context().get_stage()
        self.my_world = World(stage_units_in_meters = 1.0)


        self.manipulator: SingleManipulator | None = None
        self.camera: Camera | None = None

        self._setup_robot()
        self._setup_camera()
        self._setup_lidar()

        self.timeline = omni.timeline.get_timeline_interface()

        
    def _setup_robot(self):
        gripper = ParallelGripper(
            end_effector_prim_path="/World/Robot/Robotiq_2F_140_physics_edit/_F_Body",
            joint_prim_names=["body_f1_l", "body_f1_r"],
            joint_opened_positions=np.array([0., 0.0]),
            joint_closed_positions=np.array([45.0, 45.0]),
            action_deltas=np.array([-0.45, -0.45])
        )

        self.manipulator = SingleManipulator(
            prim_path="/World/Robot/ur10/base_link",
            name="ur10_robot",
            end_effector_prim_name="_F_Body",
            gripper=gripper
        )
        
        self.my_world.scene.add(self.manipulator)
    
    def _setup_camera(self):
        self.top_camera = Camera("/World/TopCamera", frequency=20, resolution=(2000, 2000))
        self.eef_camera = Camera("/World/Robot/ur10/ee_link/Camera", name="eef_cam", frequency=20, resolution=(500, 500))
        self.my_world.scene.add(self.top_camera)
        self.my_world.scene.add(self.eef_camera)

    def _setup_lidar(self):
        self.lidar = RotatingLidarPhysX("/World/Robot/ur10/ee_link/Lidar", rotation_frequency=0, valid_range=[0.4, 5])
        self.my_world.scene.add(self.lidar)


    def post_init(self):
        self.my_world.reset()
        self.top_camera.initialize()
        self.eef_camera.initialize()
        self.top_camera.add_bounding_box_2d_tight_to_frame()
        self.lidar.add_point_cloud_data_to_frame()


