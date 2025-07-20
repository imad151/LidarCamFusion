from isaacsim.robot.manipulators import SingleManipulator
import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.prims import SingleXFormPrim

from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats

from typing import Optional
import numpy as np
from scipy.spatial.transform import Rotation as R


class UR10RmpFlow(mg.MotionPolicyController):
    def __init__(self,
                name: Optional[str] = "controller",
                robot_articulation: Optional[SingleManipulator] = None):
        
        self._articulation = robot_articulation
        
        self._rmpflow = mg.RmpFlow(
            robot_description_path=r"C:\Users\lab\Desktop\Personal Projects\UR10_PickPlace\Robot Description\ur10_robot_description.yaml",
            rmpflow_config_path=r"C:\Users\lab\Desktop\Personal Projects\UR10_PickPlace\Robot Description\ur10_rmpflow_config.yaml",
            urdf_path=r"C:\Users\lab\Desktop\Personal Projects\UR10_PickPlace\Robot Description\ur10_robot.urdf",
            end_effector_frame_name="tool0",
            maximum_substep_size=0.00334
        )

        self._articulation_rmpflow = mg.ArticulationMotionPolicy(
            robot_articulation=self._articulation,
            motion_policy=self._rmpflow,
            default_physics_dt=1.0 / 60.0
        )

        super().__init__(
            name=name,
            articulation_motion_policy=self._articulation_rmpflow
        )

        self.default_position, self.default_orient = (self._articulation.get_world_pose())

        add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", "/World/target")
        self._target = SingleXFormPrim("/World/target", scale=[0.1, 0.1, 0.1])
        self._target.set_world_pose(np.array([0.1,0.1,1.5]),euler_angles_to_quats([0,0,0]))
    
    def update_target(self, target_end_effector_position, target_end_effector_orientation = None):
        if self._target is None:
            return
        
        self._target.set_world_pose(target_end_effector_position, target_end_effector_orientation)
    
    def update(self, step: float = 1 / 60):
        pos, orient = self._target.get_world_pose()

        self._rmpflow.set_end_effector_target(pos, orient)
        self._rmpflow.update_world()
        action = self._articulation_rmpflow.get_next_articulation_action(step)

        self._articulation.apply_action(action)

    def reset(self):
        self._target.set_world_pose(np.array([0.1,0.1,1.5]),euler_angles_to_quats([0,0,0]))
        self.default_position, self.default_orient = (self._articulation.get_world_pose())

    
    def get_arc_points(self, center, left, right, radius_scaling=3.0, num_samples=50):
        left_xy = np.array([left[0], left[1], center[2]])
        right_xy = np.array([right[0], right[1], center[2]])
        
        midpoint_xy = (left_xy + right_xy) / 2
        
        base_dist = np.linalg.norm(midpoint_xy - center)
        
        arc_points = []
        
        for i in range(num_samples):
            t = i / (num_samples - 1)
            
            xy_point = (1 - t) * left_xy + t * right_xy
            height_factor = 4 * t * (1 - t)
            max_height = base_dist * radius_scaling
            z_offset = height_factor * max_height
            
            point = np.array([xy_point[0], xy_point[1], center[2] + z_offset])
            arc_points.append(point)

        return np.array(arc_points)

    def compute_look_at_quaternions(self, arc_points, center, up = np.array([0, 0, 1])):
        f = center - arc_points
        f_norm = np.linalg.norm(f, axis=1, keepdims=True)
        f = f / f_norm

        r = np.cross(np.broadcast_to(up, f.shape), f)
        r_norm = np.linalg.norm(r, axis=1, keepdims=True)
        r = r / r_norm

        u = np.cross(f, r)

        rot_mats = np.stack([f, r, -u], axis=2)

        rots = R.from_matrix(rot_mats)
        quats = rots.as_quat()

        # Roll quats to [w, x, y, z]
        quats = np.roll(quats, 1, axis=1)

        return quats