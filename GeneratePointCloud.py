import os
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


class PointCloudProcessor:
    """
    A class for processing LiDAR point clouds and coloring them using camera images.
    """
    
    def __init__(self, data_folder, voxel_size=0.01, default_intrinsics=None):
        """
        Initialize the PointCloudProcessor.
        
        Args:
            data_folder (str): Path to the folder containing LiDAR and RGB data
            voxel_size (float): Voxel size for downsampling (default: 0.01)
            default_intrinsics (np.ndarray): Default camera intrinsics matrix (3x3)
        """
        self.data_folder = data_folder
        self.voxel_size = voxel_size
        self.default_intrinsics = default_intrinsics or np.array([
            [400, 0, 250], 
            [0, 400, 250], 
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Data storage
        self.points = None
        self.colors = None
        self.trajectory = None
        self.camera_intrinsics = None
        
    def _load_camera_intrinsics(self):
        """Load camera intrinsics from file or use defaults."""
        intrinsics_file = os.path.join(self.data_folder, "eef_cam_intrinsics.npz")
        if os.path.exists(intrinsics_file):
            self.camera_intrinsics = np.load(intrinsics_file)['K']
            print(f"[INFO] Loaded camera intrinsics from {intrinsics_file}")
        else:
            self.camera_intrinsics = self.default_intrinsics
            print("[INFO] Using default camera intrinsics")
    
    def _spherical_to_cartesian(self, sph):
        """
        Convert spherical coords [r, theta, phi] to Cartesian [x, y, z].
        
        Args:
            sph (np.ndarray): Spherical coordinates with shape (..., 3)
            
        Returns:
            np.ndarray: Cartesian coordinates with shape (..., 3)
        """
        r = sph[..., 0]
        theta = sph[..., 1]
        phi = sph[..., 2]

        x = r * np.cos(phi) * np.cos(theta)
        y = r * np.cos(phi) * np.sin(theta)
        z = r * np.sin(phi)

        return np.stack((x, y, z), axis=-1)
    
    def _project_points_to_image(self, points, pos, orient, image_shape):
        """
        Project 3D points to image plane and get pixel coordinates.
        
        Args:
            points (np.ndarray): 3D points in world coordinates
            pos (np.ndarray): Camera position
            orient (np.ndarray): Camera orientation (WXYZ quaternion)
            image_shape (tuple): Image dimensions (height, width)
            
        Returns:
            tuple: (valid_indices, pixel_coordinates)
        """
        h, w = image_shape[:2]
        
        # Transform points to camera frame
        rotation = R.from_quat([orient[1], orient[2], orient[3], orient[0]])  # WXYZ -> XYZW
        cam_points = rotation.inv().apply(points - pos)
        
        # Keep points in front of camera
        valid_mask = cam_points[:, 0] > 0.000001 
        cam_points = cam_points[valid_mask]
        
        if len(cam_points) == 0:
            return np.array([]), np.array([])
        
        # Project to image plane
        fx, fy = self.camera_intrinsics[0, 0], self.camera_intrinsics[1, 1]
        cx, cy = self.camera_intrinsics[0, 2], self.camera_intrinsics[1, 2]
        
        u = (cam_points[:, 1] * fx / cam_points[:, 0] + cx).astype(int)
        v = (cam_points[:, 2] * fy / cam_points[:, 0] + cy).astype(int)
        
        # Keep points within image bounds
        valid_pixels = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        
        return np.where(valid_mask)[0][valid_pixels], np.column_stack((u[valid_pixels], v[valid_pixels]))
    
    def _load_rgb_data(self):
        """Load all RGB camera data."""
        rgb_files = sorted(
            [f for f in os.listdir(self.data_folder) 
             if f.startswith("rgb_") and f.endswith(".npz")],
            key=lambda x: int(x.split("_")[-1].split(".")[0])
        )
        
        rgb_data = {}
        for file in rgb_files:
            frame_id = int(file.split("_")[-1].split(".")[0])
            data = np.load(os.path.join(self.data_folder, file))
            rgb_data[frame_id] = {
                'rgb': data['rgb'],
                'pos': data['pos'],
                'orient': data['orient']
            }
        
        print(f"[INFO] Loaded {len(rgb_files)} RGB files")
        return rgb_data
    
    def _process_lidar_frame(self, file, frame_id, rgb_data):
        """Process a single LiDAR frame and return colored points."""
        lidar_data = np.load(os.path.join(self.data_folder, file))
        
        sph_points = lidar_data["pointcloud"].reshape(-1, 3)
        pos = lidar_data["pos"]
        orient = lidar_data["orient"]
        
        # Convert to world coordinates
        local_cart = self._spherical_to_cartesian(sph_points)
        rotation = R.from_quat([orient[1], orient[2], orient[3], orient[0]])
        world_points = rotation.apply(local_cart) + pos
        
        # Default color (gray)
        point_colors = np.tile([0.7, 0.7, 0.7], (len(world_points), 1))
        
        # Try to color points using camera data
        if frame_id in rgb_data:
            cam_data = rgb_data[frame_id]
            valid_indices, pixels = self._project_points_to_image(
                world_points, cam_data['pos'], cam_data['orient'], cam_data['rgb'].shape
            )
            
            if len(valid_indices) > 0:
                # Get colors from RGB image
                rgb_colors = cam_data['rgb'][pixels[:, 1], pixels[:, 0]] / 255.0
                point_colors[valid_indices] = rgb_colors
                print(f"[INFO] Colored {len(valid_indices)}/{len(world_points)} points in frame {frame_id}")
        
        return world_points, point_colors, pos
    
    def load_and_process_data(self):
        """Load and process all LiDAR and RGB data."""
        print("[INFO] Loading camera intrinsics...")
        self._load_camera_intrinsics()
        
        print("[INFO] Loading RGB data...")
        rgb_data = self._load_rgb_data()
        
        # Get LiDAR files
        lidar_files = sorted(
            [f for f in os.listdir(self.data_folder) 
             if f.startswith("pointcloud_") and f.endswith(".npz")],
            key=lambda x: int(x.split("_")[-1].split(".")[0])
        )
        
        print(f"[INFO] Processing {len(lidar_files)} LiDAR files...")
        
        all_points = []
        all_colors = []
        trajectory = []
        
        # Process each LiDAR frame
        for file in lidar_files:
            frame_id = int(file.split("_")[-1].split(".")[0])
            world_points, point_colors, pos = self._process_lidar_frame(file, frame_id, rgb_data)
            
            all_points.append(world_points)
            all_colors.append(point_colors)
            trajectory.append(pos)
        
        # Merge all data
        self.points = np.vstack(all_points)
        self.colors = np.vstack(all_colors)
        self.trajectory = np.array(trajectory)
        
        print(f"[INFO] Processed {len(self.points)} total points")
        return self.points, self.colors, self.trajectory
    
    def save_pointcloud(self, output_path=None, apply_voxel_downsampling=False):
        """
        Save the colored point cloud to a PLY file.
        
        Args:
            output_path (str): Output file path. If None, saves to data_folder/colored_scene.ply
            apply_voxel_downsampling (bool): Whether to apply voxel downsampling before saving
        """
        if self.points is None or self.colors is None:
            print("[WARNING] No data loaded. Call load_and_process_data() first.")
            return
        
        if output_path is None:
            output_path = os.path.join(self.data_folder, "colored_scene.ply")
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        pcd.colors = o3d.utility.Vector3dVector(self.colors)
        
        # Apply voxel downsampling if requested
        if apply_voxel_downsampling:
            original_count = len(pcd.points)
            pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
            print(f"[INFO] Downsampled from {original_count} to {len(pcd.points)} points")
        
        # Save to file
        success = o3d.io.write_point_cloud(output_path, pcd)
        if success:
            print(f"[INFO] Successfully saved colored point cloud to {output_path}")
        else:
            print(f"[ERROR] Failed to save point cloud to {output_path}")
    
    def visualize_pointcloud(self, pointcloud_path=None):
        """
        Visualize a saved point cloud using Open3D.
        
        Args:
            pointcloud_path (str): Path to the PLY file. If None, uses data_folder/colored_scene.ply
        """
        if pointcloud_path is None:
            pointcloud_path = os.path.join(self.data_folder, "colored_scene.ply")
        
        if not os.path.exists(pointcloud_path):
            print(f"[ERROR] Point cloud file not found: {pointcloud_path}")
            return
        
        try:
            pcd = o3d.io.read_point_cloud(pointcloud_path)
            if len(pcd.points) == 0:
                print(f"[ERROR] Empty point cloud loaded from {pointcloud_path}")
                return
            
            print(f"[INFO] Loaded point cloud with {len(pcd.points)} points")
            o3d.visualization.draw_geometries([pcd])
        except Exception as e:
            print(f"[ERROR] Failed to visualize point cloud: {e}")
    
    def get_screws_locations(self, height_threshold=0.05, cluster_tolerance=0.02, min_cluster_size=5):
        """
        Find screw locations by detecting high point clusters in the point cloud.
        
        Args:
            height_threshold (float): Height threshold below the highest point to consider as screws (default: 0.1m)
            cluster_tolerance (float): Maximum distance between points in the same cluster (default: 0.02m)
            min_cluster_size (int): Minimum number of points required to form a cluster (default: 5)
            
        Returns:
            np.ndarray: Array of screw locations (cluster centroids) with shape (n_screws, 3)
        """
        if self.points is None:
            print("[WARNING] No data loaded. Call load_and_process_data() first.")
            return np.array([])
        
        # Find the range of heights in the point cloud
        z_coords = self.points[:, 2]
        min_z = np.min(z_coords)
        max_z = np.max(z_coords)
        
        print(f"[INFO] Point cloud height range: {min_z:.3f}m to {max_z:.3f}m")
        
        # Define height threshold relative to the bottom-most point
        height_cutoff = max_z - height_threshold
        
        # Filter points that are high enough to be considered screws
        high_points_mask = z_coords >= height_cutoff
        high_points = self.points[high_points_mask]
        
        if len(high_points) == 0:
            print(f"[WARNING] No screws found")
            return np.array([])
        
        
        # Create point cloud for clustering
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(high_points)
        
        # Perform DBSCAN clustering
        labels = np.array(pcd.cluster_dbscan(eps=cluster_tolerance, min_points=min_cluster_size))
        
        # Find unique clusters (excluding noise points labeled as -1)
        unique_labels = np.unique(labels)
        valid_clusters = unique_labels[unique_labels != -1]
        
        if len(valid_clusters) == 0:
            print(f"[WARNING] No valid clusters found with current parameters")
            return np.array([])
        
        print(f"[INFO] Found {len(valid_clusters)} potential screw clusters")
        
        # Calculate centroid of each cluster
        screw_locations = []
        for cluster_id in valid_clusters:
            cluster_points = high_points[labels == cluster_id]
            centroid = np.mean(cluster_points, axis=0)
            screw_locations.append(centroid)
            print(f"[INFO] Cluster {cluster_id}: {len(cluster_points)} points, centroid at ({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})")
        
        screw_locations = np.array(screw_locations)
        print(f"[INFO] Detected {len(screw_locations)} screw locations")
        
        return screw_locations
    
    def process_and_save(self, output_path=None, apply_voxel_downsampling=True):
        """
        Convenience method to load, process, and save in one call.
        
        Args:
            output_path (str): Output file path. If None, saves to data_folder/colored_scene.ply
            apply_voxel_downsampling (bool): Whether to apply voxel downsampling before saving
        """
        print("[INFO] Starting point cloud processing pipeline...")
        self.load_and_process_data()
        self.save_pointcloud(output_path, apply_voxel_downsampling)
        print("[INFO] Point cloud processing complete!")


if __name__ == "__main__":
    # Simple usage
    data_folder = r"C:\Users\lab\Desktop\Personal Projects\UR10_PickPlace\_Update_pipeline\data"
    processor = PointCloudProcessor(data_folder, voxel_size=0.01)
    processor.visualize_pointcloud(r"C:\Users\lab\Desktop\Personal Projects\UR10_PickPlace\_Update_pipeline\pointcloud.ply")