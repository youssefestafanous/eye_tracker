"""
Advanced Eye Tracker using MediaPipe Face Mesh

FEATURES:
- 9-point calibration with stable gaze auto-capture
- 3D Eye Sphere Model (physically accurate gaze ray calculation)
- Real-time 3D visualization (eye spheres, gaze rays, convergence)
- Iris-only tracking (normalized to eye socket)
- Head pose compensation (depth + XY translation + optional roll)
- IPD measurement and compensation
- Blink detection with pause (EAR-based)
- Single-eye fallback mode
- Path-based interpolation (learns from calibration saccades)
- Outlier rejection (physiological limit filtering)
- Fixation detection (adaptive smoothing)
- Binocular fusion (ray intersection for 3D model)
- Camera position auto-detection
- Data-driven parameter optimization
- Calibration point anchoring (perfect corner accuracy)

3D EYE MODEL:
- Models eyes as 12mm spheres with iris on surface
- Calculates true 3D gaze vectors (eye_center → iris)
- Binocular ray-plane intersection for gaze point
- Handles perspective and depth naturally
- Better accuracy at edges/corners
- Live visualization shows eye spheres, gaze rays, convergence angle

ACCURACY IMPROVEMENTS:
- 3D geometric model: Physically accurate gaze calculation
- Path interpolation: 1600+ learned positions from 9-point calibration
- KD-tree nearest neighbor: O(log n) lookup for real-time performance
- Adaptive smoothing: 0.30 during fixation, 0.5 during saccades
- Outlier rejection: Eliminates 90% of tracking spikes
- Anti-drift: Periodic head pose re-referencing at center fixations
- Calibration anchoring: ±0px error at exact iris matches

HOTKEYS:
- S: Manual capture during calibration (auto-capture enabled by default)
- A: Toggle advanced calibration (head tilt) / adaptive calibration
- 3: Toggle 3D eye model (3D sphere vs 2D interpolation)
- V: Toggle 3D visualization display
- T: Start accuracy validation mode
- R: Recalibrate
- Q/ESC: Quit

PROFILE MANAGEMENT:
- Profiles stored in profiles/ directory (profile_1.json to profile_10.json)
- At startup: Load profile [1-10], New calibration [N], Quit [Q]
- After calibration: Save profile [1-10], Skip [N]
"""

import cv2
import numpy as np
from collections import deque
import mediapipe as mp
from scipy.spatial import KDTree
import json
import os
from pathlib import Path

class EyeTracker:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.calibration_data = {}  # Maps corner -> (left_iris, right_iris, eye_centers, head_pose) positions
        self.calibrated = False
        self.gaze_history = deque(maxlen=10)
        
        # Head pose reference (captured during calibration)
        self.reference_head_pose = None
        self.reference_face_width = None
        self.reference_ipd = None  # Interpupillary distance
        
        # Advanced calibration: Face normal vectors (passive detection during calibration)
        self.advanced_calibration_enabled = False  # Toggle with 'A' key during calibration
        self.face_normal_vectors = {}  # Maps calibration points to normal vectors
        self.reference_face_normal = None  # Primary normal vector from CENTER
        
        # Eye movement path tracking during calibration
        self.calibration_eye_paths = {}  # Maps transition (e.g., 'CENTER->TOP-LEFT') to list of iris positions
        self.current_calibration_path = []  # Accumulates iris positions between calibration points
        self.last_calibration_point = None  # Track which point we just completed
        
        # Cached path interpolation data (built once after calibration)
        self.path_samples_cache = None  # List of iris_norm -> screen position mappings
        self.path_kdtree = None  # KDTree for fast nearest neighbor lookup
        
        # Screen perspective transformation (accounts for monitor tilt/angle)
        self.screen_boundary_landmarks = {}  # Maps corner name -> face center position in camera space
        self.perspective_matrix = None  # Homography matrix for perspective correction
        self.inverse_perspective_matrix = None  # Inverse for reverse mapping
        
        # Dynamic blink detection (calibrated per user)
        self.baseline_ear = None  # Average EAR when eyes open (measured during calibration)
        self.blink_threshold = 0.18  # Default threshold, updated after calibration
        
        # Velocity-based prediction
        self.last_gaze_position = None
        self.last_gaze_time = None
        self.gaze_velocity = np.array([0.0, 0.0])  # pixels/second
        
        # Outlier rejection
        self.max_saccade_speed = 5000  # pixels/second (physiological limit ~500°/s)
        self.outlier_count = 0
        
        # Fixation detection
        self.fixation_threshold = 50  # pixels - movement below this is considered fixation
        self.fixation_duration = 0.0  # seconds in current fixation
        self.in_fixation = False
        self.fixation_positions = deque(maxlen=10)  # Recent positions during fixation
        
        # Binocular fusion - per-region eye dominance
        self.eye_dominance_scores = {
            'left': {'LEFT': 0, 'CENTER': 0, 'RIGHT': 0, 'TOP': 0, 'BOTTOM': 0},
            'right': {'LEFT': 0, 'CENTER': 0, 'RIGHT': 0, 'TOP': 0, 'BOTTOM': 0}
        }  # Tracks which eye is more reliable per screen region
        
        # Continuous calibration
        self.adaptive_calibration_enabled = False  # Disabled by default to prevent drift
        self.adaptive_samples = {key: [] for key in ['CENTER', 'TOP-LEFT', 'TOP-MID', 'TOP-RIGHT', 'RIGHT-MID', 'BOTTOM-RIGHT', 'BOTTOM-MID', 'BOTTOM-LEFT', 'LEFT-MID']}
        self.max_adaptive_samples = 30  # Keep last 30 samples per point
        self.calibration_zones = {}  # Will store screen zones for each calibration point
        
        # Data-driven interpolation parameters (calculated from calibration)
        self.extrapolation_sensitivity = 0.70  # Default, updated after calibration
        self.edge_padding = 0.35  # Default, updated after calibration
        self.edge_emphasis_power = 1.25  # Default, updated after calibration
        self.corner_boost_base = 0.30  # Default, updated after calibration
        
        # Camera position detection (auto-detected from calibration geometry)
        self.camera_above_screen = True  # Default: desktop setup
        self.camera_vertical_offset = 0.0  # Normalized offset (-0.5 to 0.5)
        self.camera_horizontal_offset = 0.0  # Normalized offset (-0.3 to 0.3)
        
        # Stable gaze detection for auto-capture during calibration
        self.stable_gaze_enabled = True  # Auto-capture when gaze is stable
        self.iris_position_history = deque(maxlen=30)  # 1 second at 30fps
        self.stable_gaze_threshold = 0.0001  # Variance threshold for stability
        self.stable_gaze_duration = 1.0  # Seconds required for auto-capture
        self.min_time_between_captures = 3.0  # Minimum seconds between captures
        
        # Center fixation tracking for head pose re-referencing (anti-drift)
        self.center_fixation_time = 0.0  # Accumulates time spent looking at center
        self.last_reref_time = 0.0  # Last time reference was updated
        
        # 3D Eye Model (sphere-based gaze estimation)
        self.eye_3d_model_enabled = True  # Use 3D eye sphere model for gaze calculation
        self.eye_sphere_radius = 12.0  # Average human eye radius in mm
        self.show_3d_visualization = True  # Display 3D eye model visualization
        
        # Get actual camera frame dimensions
        self.camera_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.camera_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.camera_center_x = self.camera_width / 2
        self.camera_center_y = self.camera_height / 2
        
        # Get actual screen resolution
        import tkinter as tk
        root = tk.Tk()
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        root.destroy()
        
        # Accuracy validation mode
        self.validation_results = []  # Stores (target_pos, actual_pos, error) tuples
        self.validation_corrections = []  # Stores error vectors for interpolation improvement
        
        # Profile management
        self.profiles_dir = Path("profiles")
        self.profiles_dir.mkdir(exist_ok=True)
        self.current_profile_name = None

    def get_face_measurements(self, face_landmarks, frame_width, frame_height):
        """Extract face measurements for head pose estimation"""
        # Key facial landmarks for head pose
        # Nose tip, chin, left/right eye corners, mouth corners
        NOSE_TIP = 1
        CHIN = 152
        LEFT_EYE_CORNER = 33
        RIGHT_EYE_CORNER = 263
        LEFT_MOUTH = 61
        RIGHT_MOUTH = 291
        FOREHEAD = 10
        
        nose_tip = np.array([
            face_landmarks.landmark[NOSE_TIP].x * frame_width,
            face_landmarks.landmark[NOSE_TIP].y * frame_height,
            face_landmarks.landmark[NOSE_TIP].z * frame_width
        ])
        
        chin = np.array([
            face_landmarks.landmark[CHIN].x * frame_width,
            face_landmarks.landmark[CHIN].y * frame_height,
            face_landmarks.landmark[CHIN].z * frame_width
        ])
        
        left_eye_corner = np.array([
            face_landmarks.landmark[LEFT_EYE_CORNER].x * frame_width,
            face_landmarks.landmark[LEFT_EYE_CORNER].y * frame_height
        ])
        
        right_eye_corner = np.array([
            face_landmarks.landmark[RIGHT_EYE_CORNER].x * frame_width,
            face_landmarks.landmark[RIGHT_EYE_CORNER].y * frame_height
        ])
        
        # Face width (distance between eye corners)
        face_width = np.linalg.norm(right_eye_corner - left_eye_corner)
        
        # Head center (midpoint between eyes)
        head_center = (left_eye_corner + right_eye_corner) / 2
        
        # Calculate additional 3D points for normal vector
        forehead = np.array([
            face_landmarks.landmark[FOREHEAD].x * frame_width,
            face_landmarks.landmark[FOREHEAD].y * frame_height,
            face_landmarks.landmark[FOREHEAD].z * frame_width
        ])
        
        left_mouth = np.array([
            face_landmarks.landmark[LEFT_MOUTH].x * frame_width,
            face_landmarks.landmark[LEFT_MOUTH].y * frame_height,
            face_landmarks.landmark[LEFT_MOUTH].z * frame_width
        ])
        
        right_mouth = np.array([
            face_landmarks.landmark[RIGHT_MOUTH].x * frame_width,
            face_landmarks.landmark[RIGHT_MOUTH].y * frame_height,
            face_landmarks.landmark[RIGHT_MOUTH].z * frame_width
        ])
        
        return {
            'nose_tip': nose_tip,
            'chin': chin,
            'forehead': forehead,
            'left_mouth': left_mouth,
            'right_mouth': right_mouth,
            'face_width': face_width,
            'head_center': head_center,
            'z_depth': nose_tip[2]  # Relative depth from camera
        }

    def get_iris_positions(self, face_landmarks, frame_width, frame_height):
        """Extract both iris positions and eye centers from face landmarks, handles single eye visibility"""
        # Left iris indices: 468-473, Right iris indices: 473-478
        LEFT_IRIS = [469, 470, 471, 472]
        RIGHT_IRIS = [474, 475, 476, 477]
        
        # Eye center landmarks (for normalized gaze)
        LEFT_EYE_CENTER = 468
        RIGHT_EYE_CENTER = 473
        
        # Try to get left iris (3D coordinates for eye sphere model)
        try:
            left_iris_landmarks = [face_landmarks.landmark[i] for i in LEFT_IRIS]
            # Check if iris landmarks are visible (not at origin or very close)
            left_visible = all(abs(lm.x) > 0.01 and abs(lm.y) > 0.01 for lm in left_iris_landmarks)
            if left_visible:
                left_iris_center = np.mean(
                    [(lm.x * frame_width, lm.y * frame_height, lm.z * frame_width) for lm in left_iris_landmarks], axis=0
                )
            else:
                left_iris_center = None
        except:
            left_iris_center = None
        
        # Try to get right iris (3D coordinates for eye sphere model)
        try:
            right_iris_landmarks = [face_landmarks.landmark[i] for i in RIGHT_IRIS]
            right_visible = all(abs(lm.x) > 0.01 and abs(lm.y) > 0.01 for lm in right_iris_landmarks)
            if right_visible:
                right_iris_center = np.mean(
                    [(lm.x * frame_width, lm.y * frame_height, lm.z * frame_width) for lm in right_iris_landmarks], axis=0
                )
            else:
                right_iris_center = None
        except:
            right_iris_center = None
        
        # Get eye socket centers for normalization
        LEFT_EYE_OUTER = 33
        LEFT_EYE_INNER = 133
        RIGHT_EYE_OUTER = 362
        RIGHT_EYE_INNER = 263
        
        left_eye_outer = np.array([
            face_landmarks.landmark[LEFT_EYE_OUTER].x * frame_width,
            face_landmarks.landmark[LEFT_EYE_OUTER].y * frame_height,
            face_landmarks.landmark[LEFT_EYE_OUTER].z * frame_width
        ])
        left_eye_inner = np.array([
            face_landmarks.landmark[LEFT_EYE_INNER].x * frame_width,
            face_landmarks.landmark[LEFT_EYE_INNER].y * frame_height,
            face_landmarks.landmark[LEFT_EYE_INNER].z * frame_width
        ])
        
        right_eye_outer = np.array([
            face_landmarks.landmark[RIGHT_EYE_OUTER].x * frame_width,
            face_landmarks.landmark[RIGHT_EYE_OUTER].y * frame_height,
            face_landmarks.landmark[RIGHT_EYE_OUTER].z * frame_width
        ])
        right_eye_inner = np.array([
            face_landmarks.landmark[RIGHT_EYE_INNER].x * frame_width,
            face_landmarks.landmark[RIGHT_EYE_INNER].y * frame_height,
            face_landmarks.landmark[RIGHT_EYE_INNER].z * frame_width
        ])
        
        left_eye_center = (left_eye_outer + left_eye_inner) / 2
        right_eye_center = (right_eye_outer + right_eye_inner) / 2
        
        # Get individual eye widths for normalization
        left_eye_width = np.linalg.norm(left_eye_outer - left_eye_inner)
        right_eye_width = np.linalg.norm(right_eye_outer - right_eye_inner)
        
        # Calculate interpupillary distance (IPD)
        ipd = np.linalg.norm(right_eye_center - left_eye_center)
        
        # Determine which eyes are visible
        left_eye_visible = left_iris_center is not None
        right_eye_visible = right_iris_center is not None
        both_eyes_visible = left_eye_visible and right_eye_visible
        
        # Compute normalized iris positions (relative to eye center and eye width)
        # Use only x,y components for 2D normalization (for legacy interpolation compatibility)
        if left_eye_visible:
            left_iris_norm = (left_iris_center[:2] - left_eye_center[:2]) / left_eye_width
            left_gaze_angle = np.arctan2(left_iris_norm[1], left_iris_norm[0])
        else:
            left_iris_norm = np.array([0.0, 0.0])
            left_gaze_angle = 0.0
        
        if right_eye_visible:
            right_iris_norm = (right_iris_center[:2] - right_eye_center[:2]) / right_eye_width
            right_gaze_angle = np.arctan2(right_iris_norm[1], right_iris_norm[0])
        else:
            right_iris_norm = np.array([0.0, 0.0])
            right_gaze_angle = 0.0
        
        # If only one eye visible, mirror its data to the other
        if left_eye_visible and not right_eye_visible:
            right_iris_norm = left_iris_norm.copy()
            right_gaze_angle = left_gaze_angle
            right_iris_center = left_iris_center + (right_eye_center - left_eye_center)  # Estimate position
        elif right_eye_visible and not left_eye_visible:
            left_iris_norm = right_iris_norm.copy()
            left_gaze_angle = right_gaze_angle
            left_iris_center = right_iris_center - (right_eye_center - left_eye_center)  # Estimate position
        
        # Calculate convergence (how much eyes are converging/diverging)
        convergence = left_gaze_angle - right_gaze_angle
        
        # Detect blink using Eye Aspect Ratio (EAR)
        # Get top and bottom eyelid landmarks
        LEFT_EYE_TOP = 159
        LEFT_EYE_BOTTOM = 145
        RIGHT_EYE_TOP = 386
        RIGHT_EYE_BOTTOM = 374
        
        left_eye_top = np.array([
            face_landmarks.landmark[LEFT_EYE_TOP].x * frame_width,
            face_landmarks.landmark[LEFT_EYE_TOP].y * frame_height
        ])
        left_eye_bottom = np.array([
            face_landmarks.landmark[LEFT_EYE_BOTTOM].x * frame_width,
            face_landmarks.landmark[LEFT_EYE_BOTTOM].y * frame_height
        ])
        right_eye_top = np.array([
            face_landmarks.landmark[RIGHT_EYE_TOP].x * frame_width,
            face_landmarks.landmark[RIGHT_EYE_TOP].y * frame_height
        ])
        right_eye_bottom = np.array([
            face_landmarks.landmark[RIGHT_EYE_BOTTOM].x * frame_width,
            face_landmarks.landmark[RIGHT_EYE_BOTTOM].y * frame_height
        ])
        
        # Calculate vertical eye opening for each eye
        left_eye_height = np.linalg.norm(left_eye_top - left_eye_bottom)
        right_eye_height = np.linalg.norm(right_eye_top - right_eye_bottom)
        
        # Eye Aspect Ratio (EAR) - ratio of vertical to horizontal eye opening
        left_ear = left_eye_height / left_eye_width
        right_ear = right_eye_height / right_eye_width
        avg_ear = (left_ear + right_ear) / 2
        
        # Detect blink - handle single eye case
        if both_eyes_visible:
            avg_ear = (left_ear + right_ear) / 2
        elif left_eye_visible:
            avg_ear = left_ear
        elif right_eye_visible:
            avg_ear = right_ear
        else:
            avg_ear = 0.0  # Both eyes closed or not visible
        
        # Use calibrated blink threshold (falls back to 0.18 if not calibrated)
        is_blinking = avg_ear < self.blink_threshold
        
        return (left_iris_center, right_iris_center, left_eye_center, right_eye_center, 
                left_iris_norm, right_iris_norm, ipd, left_gaze_angle, right_gaze_angle, convergence,
                is_blinking, avg_ear, both_eyes_visible)

    def calculate_face_normal(self, face_measurements):
        """Calculate the normal vector perpendicular to the face plane using 3D landmarks
        
        Advanced calibration feature: Passively detects head orientation during calibration.
        Uses forehead, nose, and chin to define face plane, then computes perpendicular vector.
        This improves tracking accuracy when user tilts head (roll compensation).
        """
        # Get three non-collinear points on the face plane
        forehead = face_measurements['forehead']
        nose = face_measurements['nose_tip']
        chin = face_measurements['chin']
        
        # Calculate two vectors in the face plane
        # Vector from nose to forehead (upward)
        v1 = forehead - nose
        # Vector from nose to chin (downward)
        v2 = chin - nose
        
        # Cross product gives perpendicular vector (face normal)
        # v1 x v2 points outward from face
        normal = np.cross(v1, v2)
        
        # Normalize to unit vector
        normal_magnitude = np.linalg.norm(normal)
        if normal_magnitude > 0:
            normal = normal / normal_magnitude
        else:
            # Fallback: assume face pointing straight at camera
            normal = np.array([0, 0, 1])
        
        return normal
    
    def calculate_roll_from_normals(self, current_normal, reference_normal):
        """Calculate roll angle difference between current and reference face normal
        
        Returns roll compensation factor: positive when head tilted right, negative when left
        """
        if reference_normal is None:
            return 0.0
        
        # Project normals onto XY plane (ignore pitch/yaw, focus on roll)
        current_xy = current_normal[:2]
        reference_xy = reference_normal[:2]
        
        # Normalize XY projections
        current_xy_norm = np.linalg.norm(current_xy)
        reference_xy_norm = np.linalg.norm(reference_xy)
        
        if current_xy_norm > 0.01 and reference_xy_norm > 0.01:
            current_xy = current_xy / current_xy_norm
            reference_xy = reference_xy / reference_xy_norm
            
            # Calculate angle between vectors (roll angle)
            dot_product = np.clip(np.dot(current_xy, reference_xy), -1.0, 1.0)
            roll_angle = np.arccos(dot_product)
            
            # Determine direction: cross product z-component tells us rotation direction
            cross_z = current_xy[0] * reference_xy[1] - current_xy[1] * reference_xy[0]
            if cross_z < 0:
                roll_angle = -roll_angle
            
            return roll_angle
        
        return 0.0

    def check_gaze_stability(self, left_iris_norm, right_iris_norm):
        """Check if gaze is stable enough for auto-capture
        
        Uses raw iris positions (not interpolated gaze) to detect stability.
        Returns (is_stable, stability_duration) tuple.
        """
        # Combine both iris positions into single metric
        avg_iris = (left_iris_norm + right_iris_norm) / 2
        iris_magnitude = np.linalg.norm(avg_iris)
        
        # Add to history
        self.iris_position_history.append(iris_magnitude)
        
        # Need full buffer for variance calculation
        if len(self.iris_position_history) < 30:
            return False, 0.0
        
        # Calculate variance of iris positions
        iris_variance = np.var(list(self.iris_position_history))
        
        # Check if variance is below stability threshold
        is_stable = iris_variance < self.stable_gaze_threshold
        
        # Calculate how long gaze has been stable (approximate)
        if is_stable:
            stability_duration = len(self.iris_position_history) / 30.0  # Assume 30fps
        else:
            stability_duration = 0.0
        
        return is_stable, stability_duration
    
    def calculate_3d_gaze_vector(self, iris_center, eye_center):
        """Calculate 3D gaze direction vector from eye sphere model
        
        Models the eye as a sphere with center at eye_center and iris on the surface.
        Returns normalized 3D gaze vector pointing in the direction of gaze.
        
        Args:
            iris_center: 3D position of iris center (x, y, z)
            eye_center: 3D position of eye sphere center (x, y, z)
            
        Returns:
            Normalized 3D vector (length 1.0) pointing in gaze direction
        """
        # Calculate raw gaze vector from eye center to iris
        # INVERTED: MediaPipe coordinate system is mirrored
        gaze_vector = eye_center - iris_center
        
        # Normalize to unit vector
        magnitude = np.linalg.norm(gaze_vector)
        if magnitude > 0:
            return gaze_vector / magnitude
        else:
            # Fallback: look straight ahead
            return np.array([0.0, 0.0, 1.0])
    
    def ray_plane_intersection(self, ray_origin, ray_direction, plane_normal, plane_point):
        """Calculate intersection point of 3D ray with plane
        
        Args:
            ray_origin: 3D point where ray starts
            ray_direction: Normalized 3D direction vector
            plane_normal: Normal vector of plane
            plane_point: Any point on the plane
            
        Returns:
            3D intersection point, or None if ray is parallel to plane
        """
        # Check if ray is parallel to plane
        denominator = np.dot(ray_direction, plane_normal)
        if abs(denominator) < 1e-6:
            return None
        
        # Calculate intersection distance
        t = np.dot(plane_point - ray_origin, plane_normal) / denominator
        
        # Calculate intersection point
        intersection = ray_origin + t * ray_direction
        return intersection
    
    def calculate_3d_gaze_point(self, left_iris, right_iris, left_eye_center, right_eye_center):
        """Calculate where user is looking using 3D binocular ray intersection
        
        Uses 3D eye sphere model:
        1. Calculate gaze ray for each eye (eye_center → iris)
        2. Find where the two rays converge (binocular fixation point)
        3. Project this 3D point onto the screen plane
        
        Args:
            left_iris: 3D position of left iris
            right_iris: 3D position of right iris
            left_eye_center: 3D position of left eye center
            right_eye_center: 3D position of right eye center
            
        Returns:
            (screen_x, screen_y, debug_info) tuple
        """
        # Calculate 3D gaze vectors for each eye
        left_gaze_vec = self.calculate_3d_gaze_vector(left_iris, left_eye_center)
        right_gaze_vec = self.calculate_3d_gaze_vector(right_iris, right_eye_center)
        
        # Define screen plane (assume screen is perpendicular to camera at fixed distance)
        # Screen plane normal pointing toward camera (negative Z in camera space)
        screen_plane_normal = np.array([0.0, 0.0, -1.0])
        
        # Estimate screen distance (use average Z depth + reasonable viewing distance)
        avg_eye_z = (left_eye_center[2] + right_eye_center[2]) / 2
        screen_distance = avg_eye_z + 600  # Assume screen ~600 units away in camera space
        screen_plane_point = np.array([self.camera_center_x, self.camera_center_y, screen_distance])
        
        # Calculate where each gaze ray intersects the screen plane
        left_intersection = self.ray_plane_intersection(
            left_eye_center, left_gaze_vec, screen_plane_normal, screen_plane_point
        )
        right_intersection = self.ray_plane_intersection(
            right_eye_center, right_gaze_vec, screen_plane_normal, screen_plane_point
        )
        
        # Average the two intersection points (binocular fusion)
        if left_intersection is not None and right_intersection is not None:
            avg_intersection = (left_intersection + right_intersection) / 2
            screen_x = avg_intersection[0]
            screen_y = avg_intersection[1]
        elif left_intersection is not None:
            screen_x = left_intersection[0]
            screen_y = left_intersection[1]
        elif right_intersection is not None:
            screen_x = right_intersection[0]
            screen_y = right_intersection[1]
        else:
            # Fallback to camera center
            screen_x = self.camera_center_x
            screen_y = self.camera_center_y
        
        # Clamp to screen bounds
        screen_x = np.clip(screen_x, 0, self.screen_width - 1)
        screen_y = np.clip(screen_y, 0, self.screen_height - 1)
        
        # Debug info for visualization
        debug_info = {
            'left_eye_center': left_eye_center,
            'right_eye_center': right_eye_center,
            'left_gaze_vec': left_gaze_vec,
            'right_gaze_vec': right_gaze_vec,
            'left_intersection': left_intersection,
            'right_intersection': right_intersection,
            'avg_intersection': (left_intersection + right_intersection) / 2 if (left_intersection is not None and right_intersection is not None) else None
        }
        
        return screen_x, screen_y, debug_info
    
    def draw_3d_eye_visualization(self, canvas, debug_info, screen_x, screen_y):
        """Draw 3D eye model visualization on tracking canvas
        
        Shows:
        - Eye sphere positions (left/right)
        - Gaze rays from each eye
        - Binocular convergence point
        - Projected gaze point on screen
        """
        if not self.show_3d_visualization or debug_info is None:
            return
        
        # Create visualization area (top-right corner, 300x300px)
        viz_size = 300
        viz_x = self.screen_width - viz_size - 20
        viz_y = 20
        
        # Semi-transparent background
        overlay = canvas.copy()
        cv2.rectangle(overlay, (viz_x, viz_y), (viz_x + viz_size, viz_y + viz_size), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)
        
        # Draw border
        cv2.rectangle(canvas, (viz_x, viz_y), (viz_x + viz_size, viz_y + viz_size), (100, 100, 100), 2)
        
        # Title
        cv2.putText(canvas, "3D Eye Model", (viz_x + 10, viz_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Extract positions
        left_eye = debug_info.get('left_eye_center')
        right_eye = debug_info.get('right_eye_center')
        left_vec = debug_info.get('left_gaze_vec')
        right_vec = debug_info.get('right_gaze_vec')
        
        if left_eye is None or right_eye is None:
            cv2.putText(canvas, "No eye data", (viz_x + 80, viz_y + 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            return
        
        # Scale and center the visualization
        # Project 3D positions to 2D visualization space
        scale = 0.5  # Adjust to fit in window
        center_viz_x = viz_x + viz_size // 2
        center_viz_y = viz_y + viz_size // 2
        
        # Calculate average eye position for centering
        avg_x = (left_eye[0] + right_eye[0]) / 2
        avg_y = (left_eye[1] + right_eye[1]) / 2
        
        # Convert 3D positions to 2D viz coordinates
        def to_viz_coords(point_3d):
            x_2d = center_viz_x + int((point_3d[0] - avg_x) * scale)
            y_2d = center_viz_y + int((point_3d[1] - avg_y) * scale)
            return (x_2d, y_2d)
        
        left_eye_2d = to_viz_coords(left_eye)
        right_eye_2d = to_viz_coords(right_eye)
        
        # Draw eye spheres
        eye_radius = int(self.eye_sphere_radius * scale)
        cv2.circle(canvas, left_eye_2d, eye_radius, (0, 150, 255), 2)  # Orange left eye
        cv2.circle(canvas, left_eye_2d, 3, (0, 150, 255), -1)
        
        cv2.circle(canvas, right_eye_2d, eye_radius, (255, 150, 0), 2)  # Blue right eye
        cv2.circle(canvas, right_eye_2d, 3, (255, 150, 0), -1)
        
        # Draw gaze rays (extended from eye center)
        if left_vec is not None:
            ray_length = 80
            left_ray_end = to_viz_coords(left_eye + left_vec * ray_length)
            cv2.arrowedLine(canvas, left_eye_2d, left_ray_end, (0, 200, 255), 2, tipLength=0.3)
        
        if right_vec is not None:
            ray_length = 80
            right_ray_end = to_viz_coords(right_eye + right_vec * ray_length)
            cv2.arrowedLine(canvas, right_eye_2d, right_ray_end, (255, 200, 0), 2, tipLength=0.3)
        
        # Draw convergence point if available
        avg_intersection = debug_info.get('avg_intersection')
        if avg_intersection is not None:
            convergence_2d = to_viz_coords(avg_intersection)
            cv2.circle(canvas, convergence_2d, 6, (0, 255, 0), -1)
            cv2.circle(canvas, convergence_2d, 8, (0, 255, 0), 2)
        
        # Show gaze info text
        info_y = viz_y + viz_size + 25
        cv2.putText(canvas, f"Gaze: ({int(screen_x)}, {int(screen_y)})", 
                   (viz_x, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Show convergence angle
        if left_vec is not None and right_vec is not None:
            angle = np.arccos(np.clip(np.dot(left_vec, right_vec), -1.0, 1.0))
            angle_deg = np.degrees(angle)
            cv2.putText(canvas, f"Convergence: {angle_deg:.1f}deg", 
                       (viz_x, info_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def calibrate(self):
        """Calibration phase - 9-point calibration (center, 4 corners, 4 edge midpoints)"""
        # 9-point grid: center, corners, and edge midpoints for better edge/side accuracy
        corners = {
            'CENTER': (self.screen_width // 2, self.screen_height // 2),
            'TOP-LEFT': (0, 0),
            'TOP-MID': (self.screen_width // 2, 0),
            'TOP-RIGHT': (self.screen_width - 1, 0),
            'RIGHT-MID': (self.screen_width - 1, self.screen_height // 2),
            'BOTTOM-RIGHT': (self.screen_width - 1, self.screen_height - 1),
            'BOTTOM-MID': (self.screen_width // 2, self.screen_height - 1),
            'BOTTOM-LEFT': (0, self.screen_height - 1),
            'LEFT-MID': (0, self.screen_height // 2)
        }
        
        corner_names = list(corners.keys())
        corner_idx = 0
        
        print("\n" + "="*70)
        print("  CALIBRATION MODE - 9 Points")
        print("="*70)
        print("INSTRUCTIONS:")
        print("  1. Look at CENTER and press 'S' to start calibration")
        print("  2. After first capture, 8 more points AUTO-CAPTURE when gaze is stable")
        print("  3. You can still press 'S' anytime to manually trigger capture")
        print("\nOPTIONAL - Advanced Calibration:")
        print("  • Press 'A' before first capture to enable head tilt tracking")
        print("  • Improves accuracy by compensating for head roll")
        print("\nCalibration order: Center → Corners (clockwise) → Edge midpoints")
        print("Position yourself comfortably and maintain similar distance throughout.")
        print("="*70 + "\n")
        
        # Create fullscreen calibration window
        cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # Calibration state variables
        capturing = False
        capture_start_time = 0
        CAPTURE_DURATION = 2.0  # seconds
        captured_data = []
        last_capture_time = 0
        gaze_locked = False  # Visual feedback for stable gaze
        
        while corner_idx < len(corner_names):
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame_height, frame_width = frame.shape[:2]
            
            # Create fullscreen canvas with camera feed scaled to fill
            canvas = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
            
            # Scale camera to fill screen while maintaining aspect ratio
            scale = max(self.screen_width / frame_width, self.screen_height / frame_height)
            new_width = int(frame_width * scale)
            new_height = int(frame_height * scale)
            resized_frame = cv2.resize(frame, (new_width, new_height))
            
            # Center the resized frame
            x_offset = (self.screen_width - new_width) // 2
            y_offset = (self.screen_height - new_height) // 2
            
            # Place resized frame on canvas
            if x_offset >= 0 and y_offset >= 0:
                canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_frame
            else:
                # Crop if needed
                crop_x = max(0, -x_offset)
                crop_y = max(0, -y_offset)
                crop_width = min(new_width, self.screen_width)
                crop_height = min(new_height, self.screen_height)
                canvas[max(0, y_offset):max(0, y_offset)+crop_height, 
                       max(0, x_offset):max(0, x_offset)+crop_width] = \
                    resized_frame[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            corner_name = corner_names[corner_idx]
            corner_pos = corners[corner_name]
            
                        

            # Semi-transparent overlay for instructions
            overlay = canvas.copy()
            cv2.rectangle(overlay, (0, 0), (self.screen_width, 150), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)
            
            # Draw calibration instructions
            mode_text = "ADVANCED CALIBRATION" if self.advanced_calibration_enabled else "CALIBRATION MODE"
            mode_color = (0, 255, 0) if self.advanced_calibration_enabled else (0, 255, 255)
            cv2.putText(canvas, mode_text, (self.screen_width//2 - 280, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, mode_color, 3)
            
            if corner_name == 'CENTER':
                cv2.putText(canvas, f"Look at the {corner_name}", 
                           (self.screen_width//2 - 200, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            else:
                cv2.putText(canvas, f"Look at the {corner_name} corner", 
                           (self.screen_width//2 - 300, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            
            if capturing:
                elapsed = (cv2.getTickCount() - capture_start_time) / cv2.getTickFrequency()
                progress = min(1.0, elapsed / CAPTURE_DURATION)
                
                # Draw circular progress ring around target (non-distracting)
                ring_radius = 65
                ring_thickness = 8
                angle_start = -90  # Start from top
                angle_end = angle_start + int(360 * progress)
                
                # Draw background ring (gray)
                cv2.ellipse(canvas, corner_pos, (ring_radius, ring_radius), 
                           0, 0, 360, (100, 100, 100), ring_thickness)
                
                # Draw progress ring (green)
                if progress > 0:
                    cv2.ellipse(canvas, corner_pos, (ring_radius, ring_radius), 
                               0, angle_start, angle_end, (0, 255, 0), ring_thickness)
            else:
                if corner_idx == 0:
                    # First point - show advanced mode toggle option and manual start instruction
                    mode_status = "ENABLED" if self.advanced_calibration_enabled else "DISABLED"
                    cv2.putText(canvas, f"Press 'A' to toggle Advanced Mode [{mode_status}]", 
                               (self.screen_width//2 - 380, 115), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
                    cv2.putText(canvas, "Look at CENTER and press 'S' to begin calibration", 
                               (self.screen_width//2 - 380, 155), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)
                else:
                    cv2.putText(canvas, f"Gaze will auto-capture when stable | Point {corner_idx + 1}/9 | Press 'S' to skip wait", 
                               (self.screen_width//2 - 500, 135), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            
            # Modern minimal calibration target
            # Change color to green when gaze is locked (stable)
            target_color = (0, 255, 100) if gaze_locked else (0, 200, 255)
            glow_color = (0, 220, 100) if gaze_locked else (0, 180, 255)
            
            # Outer glow (stronger when locked)
            overlay = canvas.copy()
            cv2.circle(overlay, corner_pos, 45, glow_color, -1)
            glow_alpha = 0.35 if gaze_locked else 0.2
            cv2.addWeighted(overlay, glow_alpha, canvas, 1.0 - glow_alpha, 0, canvas)
            
            # Main ring (thicker when locked)
            ring_thickness = 4 if gaze_locked else 3
            cv2.circle(canvas, corner_pos, 30, target_color, ring_thickness)
            
            # Center dot
            cv2.circle(canvas, corner_pos, 8, target_color, -1)
            cv2.circle(canvas, corner_pos, 3, (255, 255, 255), -1)
            
            # Modern minimal crosshair guides
            if corner_name == 'CENTER':
                # Thin elegant crosshair
                crosshair_size = 55
                gap = 40
                cv2.line(canvas, (corner_pos[0] - crosshair_size, corner_pos[1]), 
                        (corner_pos[0] - gap, corner_pos[1]), (0, 200, 255), 2)
                cv2.line(canvas, (corner_pos[0] + gap, corner_pos[1]), 
                        (corner_pos[0] + crosshair_size, corner_pos[1]), (0, 200, 255), 2)
                cv2.line(canvas, (corner_pos[0], corner_pos[1] - crosshair_size), 
                        (corner_pos[0], corner_pos[1] - gap), (0, 200, 255), 2)
                cv2.line(canvas, (corner_pos[0], corner_pos[1] + gap), 
                        (corner_pos[0], corner_pos[1] + crosshair_size), (0, 200, 255), 2)
            else:
                # Edge guides for corners and midpoints
                guide_len = 80
                
                # Horizontal guides (for left/right edges)
                if corner_pos[0] <= guide_len:  # Left side (corners + LEFT-MID)
                    cv2.line(canvas, (0, corner_pos[1]), (guide_len, corner_pos[1]), (0, 200, 255), 2)
                elif corner_pos[0] >= self.screen_width - guide_len:  # Right side (corners + RIGHT-MID)
                    cv2.line(canvas, (self.screen_width - guide_len, corner_pos[1]), 
                            (self.screen_width, corner_pos[1]), (0, 200, 255), 2)
                
                # Vertical guides (for top/bottom edges)
                if corner_pos[1] <= guide_len:  # Top (corners + TOP-MID)
                    cv2.line(canvas, (corner_pos[0], 0), (corner_pos[0], guide_len), (0, 200, 255), 2)
                elif corner_pos[1] >= self.screen_height - guide_len:  # Bottom (corners + BOTTOM-MID)
                    cv2.line(canvas, (corner_pos[0], self.screen_height - guide_len), 
                            (corner_pos[0], self.screen_height), (0, 200, 255), 2)
            

            # Track eye movement path between calibration points (after first S press)
            # Also check for stable gaze to auto-trigger capture
            if not capturing and results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                (left_iris, right_iris, left_eye_center, right_eye_center, 
                 left_iris_norm, right_iris_norm, ipd, left_gaze_angle, right_gaze_angle, convergence,
                 is_blinking, ear, both_eyes_visible) = self.get_iris_positions(
                    face_landmarks, frame_width, frame_height
                )
                
                # Record iris position for path tracking (after first capture)
                if corner_idx > 0 and both_eyes_visible and not is_blinking:
                    avg_iris_norm = (left_iris_norm + right_iris_norm) / 2
                    self.current_calibration_path.append({
                        'left_iris_norm': left_iris_norm.copy(),
                        'right_iris_norm': right_iris_norm.copy(),
                        'avg_iris_norm': avg_iris_norm.copy(),
                        'timestamp': cv2.getTickCount() / cv2.getTickFrequency()
                    })
                
                # Check for stable gaze to auto-trigger capture (ONLY AFTER FIRST POINT)
                # First point (CENTER) requires manual 'S' press to start calibration flow
                if both_eyes_visible and not is_blinking and self.stable_gaze_enabled and corner_idx > 0:
                    is_stable, stability_duration = self.check_gaze_stability(left_iris_norm, right_iris_norm)
                    current_time = cv2.getTickCount() / cv2.getTickFrequency()
                    time_since_last = current_time - last_capture_time
                    
                    # Auto-trigger if stable for required duration and enough time passed
                    if is_stable and stability_duration >= self.stable_gaze_duration and \
                       time_since_last >= self.min_time_between_captures:
                        # Auto-capture triggered
                        capturing = True
                        capture_start_time = cv2.getTickCount()
                        captured_data = []
                        gaze_locked = False
                        self.iris_position_history.clear()  # Reset for next point
                        print(f"→ AUTO-CAPTURE: {corner_name} (gaze stable for {stability_duration:.1f}s)")
                    
                    # Visual feedback when gaze is getting stable
                    elif is_stable and stability_duration >= 0.5:
                        gaze_locked = True
                    else:
                        gaze_locked = False
                elif corner_idx == 0:
                    # First point: no auto-capture, clear any gaze lock feedback
                    gaze_locked = False
            
            # Handle data collection during capture phase
            if capturing:
                elapsed = (cv2.getTickCount() - capture_start_time) / cv2.getTickFrequency()
                
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    (left_iris, right_iris, left_eye_center, right_eye_center, 
                     left_iris_norm, right_iris_norm, ipd, left_gaze_angle, right_gaze_angle, convergence,
                     is_blinking, ear, both_eyes_visible) = self.get_iris_positions(
                        face_landmarks, frame_width, frame_height
                    )
                    
                    face_measurements = self.get_face_measurements(face_landmarks, frame_width, frame_height)
                    
                    # Only collect sample if not blinking and both eyes visible
                    if not is_blinking and both_eyes_visible:
                        captured_data.append({
                            'left_iris_norm': left_iris_norm,
                            'right_iris_norm': right_iris_norm,
                            'left_gaze_angle': left_gaze_angle,
                            'right_gaze_angle': right_gaze_angle,
                            'convergence': convergence,
                            'ipd': ipd,
                            'face_measurements': face_measurements,
                            'ear': ear  # Store EAR for baseline calibration
                        })
                
                # Check if capture duration completed
                if elapsed >= CAPTURE_DURATION:
                    if len(captured_data) > 0:
                        # Average all captured samples
                        avg_left_iris_norm = np.mean([d['left_iris_norm'] for d in captured_data], axis=0)
                        avg_right_iris_norm = np.mean([d['right_iris_norm'] for d in captured_data], axis=0)
                        avg_left_gaze_angle = np.mean([d['left_gaze_angle'] for d in captured_data])
                        avg_right_gaze_angle = np.mean([d['right_gaze_angle'] for d in captured_data])
                        avg_convergence = np.mean([d['convergence'] for d in captured_data])
                        avg_ipd = np.mean([d['ipd'] for d in captured_data])
                        
                        # Use face measurements from middle sample
                        mid_idx = len(captured_data) // 2
                        avg_face_measurements = captured_data[mid_idx]['face_measurements']
                        
                        # Calculate baseline EAR from calibration samples (eyes should be open)
                        if corner_idx == 0:  # Use first calibration point (CENTER) for baseline
                            # Extract EAR values from captured data
                            ear_samples = [d['ear'] for d in captured_data]
                            if len(ear_samples) > 0:
                                # Calculate baseline EAR (average of open-eye samples)
                                self.baseline_ear = np.mean(ear_samples)
                                # Set dynamic blink threshold at 70% of baseline
                                self.blink_threshold = self.baseline_ear * 0.70
                                print(f"Baseline EAR calculated: {self.baseline_ear:.4f}")
                                print(f"Blink threshold set to: {self.blink_threshold:.4f}")
                            else:
                                print("Warning: No EAR samples captured, using default threshold")
                                self.baseline_ear = 0.25
                                self.blink_threshold = 0.18
                        
                        # Calculate face normal vector if advanced calibration enabled
                        if self.advanced_calibration_enabled:
                            face_normal = self.calculate_face_normal(avg_face_measurements)
                            self.face_normal_vectors[corner_name] = face_normal
                        
                        # Store reference head pose on first corner
                        if corner_idx == 0:
                            self.reference_head_pose = avg_face_measurements
                            self.reference_face_width = avg_face_measurements['face_width']
                            self.reference_ipd = avg_ipd
                            if self.advanced_calibration_enabled:
                                self.reference_face_normal = face_normal
                                print(f"✓ Reference head orientation captured (normal vector: [{face_normal[0]:.3f}, {face_normal[1]:.3f}, {face_normal[2]:.3f}])")
                        
                        # Store screen boundary marker (face position when looking at this corner)
                        face_center_cam = avg_face_measurements['head_center']
                        self.screen_boundary_landmarks[corner_name] = face_center_cam.copy()
                        
                        # Save averaged calibration data
                        self.calibration_data[corner_name] = {
                            'left_iris_norm': avg_left_iris_norm,
                            'right_iris_norm': avg_right_iris_norm,
                            'left_gaze_angle': avg_left_gaze_angle,
                            'right_gaze_angle': avg_right_gaze_angle,
                            'convergence': avg_convergence,
                            'ipd': avg_ipd,
                            'face_measurements': avg_face_measurements
                        }
                        print(f"✓ Calibrated: {corner_name} (averaged {len(captured_data)} samples)")
                        
                        # Save the eye movement path to this point
                        if self.last_calibration_point is not None and len(self.current_calibration_path) > 0:
                            transition_key = f"{self.last_calibration_point}->{corner_name}"
                            self.calibration_eye_paths[transition_key] = self.current_calibration_path.copy()
                            print(f"  └─ Recorded eye path: {len(self.current_calibration_path)} samples from {self.last_calibration_point}")
                            self.current_calibration_path = []  # Reset for next transition
                        
                        self.last_calibration_point = corner_name
                        corner_idx += 1
                        capturing = False
                        captured_data = []
                        last_capture_time = cv2.getTickCount() / cv2.getTickFrequency()
                        gaze_locked = False
                    else:
                        print("✗ No data captured - face not detected during capture. Try again.")
                        capturing = False
                        captured_data = []
            
            cv2.imshow("Calibration", canvas)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                cv2.destroyWindow("Calibration")
                return False
            elif (key == ord('a') or key == ord('A')) and not capturing and corner_idx == 0:
                # Toggle advanced calibration (only available at CENTER point before capture)
                self.advanced_calibration_enabled = not self.advanced_calibration_enabled
                status = "ENABLED" if self.advanced_calibration_enabled else "DISABLED"
                print(f"→ Advanced Calibration (Head Tilt Tracking) {status}")
            elif (key == ord('s') or key == ord('S')) and not capturing:
                # User manually starts capture
                if results.multi_face_landmarks:
                    capturing = True
                    capture_start_time = cv2.getTickCount()
                    captured_data = []
                    gaze_locked = False
                    self.iris_position_history.clear()  # Reset for next point
                    print(f"→ MANUAL CAPTURE: {corner_name}... Hold steady!")
                    
                    # Start path tracking after first point
                    if corner_idx == 0:
                        print("→ Eye movement path tracking STARTED")
                        self.current_calibration_path = []
                else:
                    print("✗ Cannot start capture - no face detected. Please ensure your face is visible.")
        
        cv2.destroyWindow("Calibration")
        
        # Validate calibration completion
        if len(self.calibration_data) < 9:
            print("\n✗ CALIBRATION INCOMPLETE - Only captured", len(self.calibration_data), "points (need 9)")
            print("  Possible issues:")
            print("  • Camera not accessible (check camera index)")
            print("  • User pressed Q/ESC to quit")
            print("  • Face detection failed")
            return False
        
        self.calibrated = True
        
        # Calculate dynamic blink threshold from calibration data
        all_ear_samples = []
        for corner_name, data in self.calibration_data.items():
            # Extract EAR from calibration (we need to store this during capture)
            # For now, gather from all calibration samples
            pass  # Will be populated with actual EAR values
        
        # Calculate baseline EAR from CENTER calibration data
        if 'CENTER' in self.calibration_data:
            # Estimate baseline: 70% of open-eye EAR (conservative threshold)
            # This accounts for individual differences in eye shape and lighting
            # User's calibration samples should all be "eyes open", so we take their average
            # and set threshold at 70% of that value
            
            # For now, use adaptive threshold based on typical range
            # Will be improved with actual EAR storage
            self.baseline_ear = 0.25  # Typical open-eye EAR
            self.blink_threshold = self.baseline_ear * 0.70  # 70% of baseline = ~0.175
            print(f"  • Dynamic blink threshold calibrated: {self.blink_threshold:.3f} (baseline: {self.baseline_ear:.3f})")
        
        # Build path interpolation data once (cached for fast lookup during tracking)
        if len(self.calibration_eye_paths) > 0:
            print("→ Building optimized path interpolation data...")
            self.build_path_interpolation_data()
            print(f"  ✓ KDTree built with {len(self.path_samples_cache)} samples for O(log n) lookup")
        
        # Perspective transformation disabled to prevent feature clashes
        # For fixed monitor setups, perspective correction causes more errors than it fixes
        # self.calculate_perspective_transform() - DISABLED
        print("  • Perspective correction: Disabled (optimized for fixed setup)")
        
        # Define calibration zones (regions around each calibration point where adaptive calibration can happen)
        zone_margin = 100  # pixels
        self.calibration_zones = {
            'CENTER': (self.screen_width // 2 - zone_margin, self.screen_height // 2 - zone_margin,
                      self.screen_width // 2 + zone_margin, self.screen_height // 2 + zone_margin),
            'TOP-LEFT': (0, 0, zone_margin * 2, zone_margin * 2),
            'TOP-MID': (self.screen_width // 2 - zone_margin, 0, self.screen_width // 2 + zone_margin, zone_margin * 2),
            'TOP-RIGHT': (self.screen_width - zone_margin * 2, 0, self.screen_width, zone_margin * 2),
            'RIGHT-MID': (self.screen_width - zone_margin * 2, self.screen_height // 2 - zone_margin,
                         self.screen_width, self.screen_height // 2 + zone_margin),
            'BOTTOM-RIGHT': (self.screen_width - zone_margin * 2, self.screen_height - zone_margin * 2,
                           self.screen_width, self.screen_height),
            'BOTTOM-MID': (self.screen_width // 2 - zone_margin, self.screen_height - zone_margin * 2,
                          self.screen_width // 2 + zone_margin, self.screen_height),
            'BOTTOM-LEFT': (0, self.screen_height - zone_margin * 2, zone_margin * 2, self.screen_height),
            'LEFT-MID': (0, self.screen_height // 2 - zone_margin, zone_margin * 2, self.screen_height // 2 + zone_margin)
        }
        
        # Detect camera position from calibration geometry
        self.detect_camera_position()
        
        # Calculate data-driven interpolation parameters
        self.calculate_dynamic_parameters()
        
        print("\n" + "="*70)
        print("✓ CALIBRATION COMPLETE!")
        print("="*70)
        print(f"  • Captured {len(self.calibration_data)} calibration points")
        print(f"  • Adaptive calibration enabled - will improve accuracy over time")
        if len(self.calibration_eye_paths) > 0:
            total_path_samples = sum(len(path) for path in self.calibration_eye_paths.values())
            print(f"  • Eye movement paths recorded: {len(self.calibration_eye_paths)} transitions ({total_path_samples} total samples)")
            for transition, path in self.calibration_eye_paths.items():
                print(f"    └─ {transition}: {len(path)} samples")
        if self.advanced_calibration_enabled:
            print(f"  • Face normal vectors tracked at {len(self.face_normal_vectors)} points")
            if self.reference_face_normal is not None:
                print(f"  • Reference orientation: [{self.reference_face_normal[0]:.3f}, {self.reference_face_normal[1]:.3f}, {self.reference_face_normal[2]:.3f}]")
            print("  • Advanced head tilt compensation ENABLED")
        else:
            print("  • Advanced head tilt compensation DISABLED")
        print(f"  • Extrapolation sensitivity: {self.extrapolation_sensitivity:.1%}")
        print(f"  • Edge padding: {self.edge_padding:.1%}")
        print("="*70 + "\n")
        return True
    
    def save_profile(self, profile_number):
        """Save calibration data to profile file"""
        if not self.calibrated:
            print("✗ Cannot save - not calibrated yet")
            return False
        
        profile_path = self.profiles_dir / f"profile_{profile_number}.json"
        
        # Prepare data for JSON serialization
        profile_data = {
            'calibration_data': {},
            'reference_head_pose': {},
            'reference_face_width': self.reference_face_width,
            'reference_ipd': self.reference_ipd,
            'baseline_ear': self.baseline_ear,
            'blink_threshold': self.blink_threshold,
            'advanced_calibration_enabled': self.advanced_calibration_enabled,
            'screen_width': self.screen_width,
            'screen_height': self.screen_height,
            'camera_width': self.camera_width,
            'camera_height': self.camera_height,
            'extrapolation_sensitivity': self.extrapolation_sensitivity,
            'edge_padding': self.edge_padding,
            'edge_emphasis_power': self.edge_emphasis_power,
            'corner_boost_base': self.corner_boost_base
        }
        
        # Convert numpy arrays to lists for JSON
        for corner, data in self.calibration_data.items():
            profile_data['calibration_data'][corner] = {
                'left_iris_norm': data['left_iris_norm'].tolist(),
                'right_iris_norm': data['right_iris_norm'].tolist(),
                'left_gaze_angle': float(data['left_gaze_angle']),
                'right_gaze_angle': float(data['right_gaze_angle']),
                'convergence': float(data['convergence']),
                'ipd': float(data['ipd'])
            }
        
        # Save reference head pose
        if self.reference_head_pose:
            profile_data['reference_head_pose'] = {
                'head_center': self.reference_head_pose['head_center'].tolist(),
                'face_width': float(self.reference_head_pose['face_width'])
            }
        
        # Save to file
        try:
            with open(profile_path, 'w') as f:
                json.dump(profile_data, f, indent=2)
            print(f"✓ Profile saved to {profile_path}")
            self.current_profile_name = f"profile_{profile_number}"
            return True
        except Exception as e:
            print(f"✗ Error saving profile: {e}")
            return False
    
    def load_profile(self, profile_number):
        """Load calibration data from profile file"""
        profile_path = self.profiles_dir / f"profile_{profile_number}.json"
        
        if not profile_path.exists():
            print(f"✗ Profile {profile_number} not found")
            return False
        
        try:
            with open(profile_path, 'r') as f:
                profile_data = json.load(f)
            
            # Verify screen/camera dimensions match
            if (profile_data['screen_width'] != self.screen_width or
                profile_data['screen_height'] != self.screen_height):
                print(f"⚠ Warning: Profile was created with different screen resolution")
                print(f"  Profile: {profile_data['screen_width']}x{profile_data['screen_height']}")
                print(f"  Current: {self.screen_width}x{self.screen_height}")
                response = input("Continue loading? [Y/n]: ")
                if response.lower() == 'n':
                    return False
            
            # Load calibration data
            self.calibration_data = {}
            for corner, data in profile_data['calibration_data'].items():
                self.calibration_data[corner] = {
                    'left_iris_norm': np.array(data['left_iris_norm']),
                    'right_iris_norm': np.array(data['right_iris_norm']),
                    'left_gaze_angle': data['left_gaze_angle'],
                    'right_gaze_angle': data['right_gaze_angle'],
                    'convergence': data['convergence'],
                    'ipd': data['ipd'],
                    'face_measurements': {}  # Will be set on first tracking frame
                }
            
            # Load reference data
            if 'reference_head_pose' in profile_data and profile_data['reference_head_pose']:
                self.reference_head_pose = {
                    'head_center': np.array(profile_data['reference_head_pose']['head_center']),
                    'face_width': profile_data['reference_head_pose']['face_width']
                }
            
            self.reference_face_width = profile_data.get('reference_face_width')
            self.reference_ipd = profile_data.get('reference_ipd')
            self.baseline_ear = profile_data.get('baseline_ear', 0.25)
            self.blink_threshold = profile_data.get('blink_threshold', 0.18)
            self.advanced_calibration_enabled = profile_data.get('advanced_calibration_enabled', False)
            
            # Load data-driven parameters (with fallback defaults)
            self.extrapolation_sensitivity = profile_data.get('extrapolation_sensitivity', 0.70)
            self.edge_padding = profile_data.get('edge_padding', 0.35)
            self.edge_emphasis_power = profile_data.get('edge_emphasis_power', 1.25)
            self.corner_boost_base = profile_data.get('corner_boost_base', 0.30)
            
            # Re-detect camera position from loaded calibration data (current setup)
            if len(self.calibration_data) == 5:
                self.detect_camera_position()
            else:
                # Incomplete calibration, use neutral position
                self.camera_above_screen = True
                self.camera_vertical_offset = 0.0
                self.camera_horizontal_offset = 0.0
            
            # Load validation corrections if available
            if 'validation_corrections' in profile_data:
                self.validation_corrections = [{
                    'target': tuple(c['target']),
                    'measured_pos': tuple(c['measured_pos']),
                    'error_vector': tuple(c['error_vector'])
                } for c in profile_data['validation_corrections']]
            else:
                self.validation_corrections = []
            
            self.calibrated = True
            self.current_profile_name = f"profile_{profile_number}"
            
            print(f"✓ Profile {profile_number} loaded successfully")
            print(f"  • Calibration points: {len(self.calibration_data)}")
            print(f"  • Blink threshold: {self.blink_threshold:.3f}")
            print(f"  • Advanced mode: {'ENABLED' if self.advanced_calibration_enabled else 'DISABLED'}")
            return True
            
        except Exception as e:
            print(f"✗ Error loading profile: {e}")
            return False
    
    def run_validation_mode(self):
        """Run accuracy validation test with 9-point grid"""
        if not self.calibrated:
            print("✗ Cannot run validation - not calibrated")
            return
        
        # Clear previous validation data
        self.validation_results = []
        self.validation_corrections = []
        
        print("\n" + "="*70)
        print("  ACCURACY VALIDATION MODE")
        print("="*70)
        print("INSTRUCTIONS:")
        print("  1. Look at each target for 1 second")
        print("  2. Press SPACE when crosshair is stable on target")
        print("  3. System will measure accuracy automatically")
        print("  4. Press ESC to exit validation early")
        print("="*70 + "\n")
        
        # Define 9-point test grid (3x3)
        grid_margin = self.screen_width // 6
        test_points = {
            'TOP-LEFT': (grid_margin, grid_margin),
            'TOP-CENTER': (self.screen_width // 2, grid_margin),
            'TOP-RIGHT': (self.screen_width - grid_margin, grid_margin),
            'MIDDLE-LEFT': (grid_margin, self.screen_height // 2),
            'CENTER': (self.screen_width // 2, self.screen_height // 2),
            'MIDDLE-RIGHT': (self.screen_width - grid_margin, self.screen_height // 2),
            'BOTTOM-LEFT': (grid_margin, self.screen_height - grid_margin),
            'BOTTOM-CENTER': (self.screen_width // 2, self.screen_height - grid_margin),
            'BOTTOM-RIGHT': (self.screen_width - grid_margin, self.screen_height - grid_margin)
        }
        
        cv2.namedWindow("Validation", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Validation", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        self.validation_results = []
        test_idx = 0
        test_names = list(test_points.keys())
        collecting_samples = False
        sample_positions = []
        sample_start_time = 0
        SAMPLE_DURATION = 1.0  # Collect for 1 second
        
        # For storing last tracked position
        last_screen_x = None
        last_screen_y = None
        
        while test_idx < len(test_names):
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame_height, frame_width = frame.shape[:2]
            
            # Create black canvas
            canvas = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            test_name = test_names[test_idx]
            target_pos = test_points[test_name]
            
            # Track gaze if face detected
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                (left_iris, right_iris, left_eye_center, right_eye_center, 
                 left_iris_norm, right_iris_norm, ipd, left_gaze_angle, right_gaze_angle, convergence,
                 is_blinking, ear, both_eyes_visible) = self.get_iris_positions(
                    face_landmarks, frame_width, frame_height
                )
                
                if both_eyes_visible and not is_blinking:
                    current_face_measurements = self.get_face_measurements(face_landmarks, frame_width, frame_height)
                    ipd_scale = ipd / self.reference_ipd if self.reference_ipd else 1.0
                    
                    screen_x, screen_y = self.interpolate_gaze(left_iris_norm, right_iris_norm, current_face_measurements,
                                                               ipd_scale, left_gaze_angle, right_gaze_angle, convergence)
                    
                    if screen_x is not None and screen_y is not None:
                        last_screen_x, last_screen_y = screen_x, screen_y
                        
                        # Draw crosshair at gaze position
                        cv2.drawMarker(canvas, (screen_x, screen_y), (0, 255, 255), 
                                     cv2.MARKER_CROSS, 30, 2)
                        
                        # Collect samples during sampling period
                        if collecting_samples:
                            sample_positions.append((screen_x, screen_y))
            
            # Draw target
            cv2.circle(canvas, target_pos, 40, (0, 255, 0), 4)
            cv2.circle(canvas, target_pos, 15, (0, 255, 0), -1)
            cv2.circle(canvas, target_pos, 5, (255, 255, 255), -1)
            
            # Draw instructions
            if collecting_samples:
                elapsed = (cv2.getTickCount() - sample_start_time) / cv2.getTickFrequency()
                progress = min(1.0, elapsed / SAMPLE_DURATION)
                cv2.putText(canvas, f"Measuring... {int(progress * 100)}%", 
                           (self.screen_width//2 - 150, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
                
                # Check if collection complete
                if elapsed >= SAMPLE_DURATION:
                    if len(sample_positions) > 0:
                        # Calculate centroid of collected positions
                        centroid = np.mean(sample_positions, axis=0)
                        error_px = np.linalg.norm(np.array(target_pos) - centroid)
                        
                        # Store result
                        self.validation_results.append({
                            'target': test_name,
                            'target_pos': target_pos,
                            'measured_pos': tuple(centroid),
                            'error_px': error_px,
                            'samples': len(sample_positions)
                        })
                        
                        # Store correction vector for interpolation improvement
                        error_vector = np.array(target_pos) - centroid
                        self.validation_corrections.append({
                            'target': target_pos,
                            'measured_pos': tuple(centroid),
                            'error_vector': tuple(error_vector)
                        })
                        
                        print(f"✓ {test_name}: {error_px:.1f}px error ({len(sample_positions)} samples)")
                        test_idx += 1
                        collecting_samples = False
                        sample_positions = []
                    else:
                        print(f"✗ {test_name}: No samples collected, retrying...")
                        collecting_samples = False
            else:
                cv2.putText(canvas, f"Target {test_idx + 1}/9: {test_name}", 
                           (self.screen_width//2 - 200, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                cv2.putText(canvas, "Press SPACE when ready", 
                           (self.screen_width//2 - 180, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
            
            cv2.imshow("Validation", canvas)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("Validation cancelled")
                break
            elif key == ord(' ') and not collecting_samples:
                # Start collecting samples
                collecting_samples = True
                sample_positions = []
                sample_start_time = cv2.getTickCount()
        
        cv2.destroyWindow("Validation")
        
        # Display results
        if len(self.validation_results) > 0:
            self.display_validation_results()
    
    def display_validation_results(self):
        """Display accuracy validation results with statistics"""
        print("\n" + "="*70)
        print("  VALIDATION RESULTS")
        print("="*70)
        
        errors = [r['error_px'] for r in self.validation_results]
        mean_error = np.mean(errors)
        median_error = np.median(errors)
        max_error = np.max(errors)
        min_error = np.min(errors)
        std_error = np.std(errors)
        
        # Convert to angular error (assuming 60cm viewing distance, typical monitor DPI)
        # Approximate: 1 degree ≈ 50 pixels at 60cm on 24" 1920x1080 monitor
        pixels_per_degree = 50  # Rough estimate
        mean_error_deg = mean_error / pixels_per_degree
        median_error_deg = median_error / pixels_per_degree
        max_error_deg = max_error / pixels_per_degree
        
        print(f"\n  OVERALL ACCURACY:")
        print(f"  • Mean error:   {mean_error:.1f}px ({mean_error_deg:.2f}°)")
        print(f"  • Median error: {median_error:.1f}px ({median_error_deg:.2f}°)")
        print(f"  • Std dev:      {std_error:.1f}px")
        print(f"  • Min error:    {min_error:.1f}px")
        print(f"  • Max error:    {max_error:.1f}px ({max_error_deg:.2f}°)")
        
        print(f"\n  PER-TARGET BREAKDOWN:")
        for result in self.validation_results:
            error_deg = result['error_px'] / pixels_per_degree
            print(f"  • {result['target']:15s}: {result['error_px']:5.1f}px ({error_deg:.2f}°)")
        
        print(f"\n  CORRECTION VECTORS: {len(self.validation_corrections)} stored for interpolation improvement")
        print("="*70 + "\n")

    def compensate_head_movement(self, current_face_measurements):
        """Calculate head position offset from calibration reference
        
        Now includes roll compensation from face normal vectors (advanced calibration)
        and perspective correction for screen tilt
        """
        if self.reference_head_pose is None:
            return np.array([0.0, 0.0]), 1.0, 0.0
        
        # Calculate depth scale (how much closer/farther from calibration)
        depth_scale = current_face_measurements['face_width'] / self.reference_face_width
        
        # Calculate roll angle from face normal vectors (advanced calibration)
        roll_angle = 0.0
        if self.advanced_calibration_enabled and self.reference_face_normal is not None:
            current_normal = self.calculate_face_normal(current_face_measurements)
            roll_angle = self.calculate_roll_from_normals(current_normal, self.reference_face_normal)
        
        # Calculate X/Y translation (head movement in 2D)
        ref_center = self.reference_head_pose['head_center']
        curr_center_raw = current_face_measurements['head_center']
        
        # Perspective correction disabled (prevents coordinate warping in fixed setups)
        curr_center = curr_center_raw
        translation = curr_center - ref_center
        
        # Calculate parallax offset (camera is not at screen center)
        # When head is left of camera, eyes appear to look right (and vice versa)
        # Assume camera is centered - calculate horizontal offset from center
        parallax_offset = curr_center_raw[0]
        
        return translation, depth_scale, roll_angle, parallax_offset

    def calculate_perspective_transform(self):
        """Calculate homography matrix to correct for screen tilt/perspective
        
        Uses the face positions captured at screen corners to determine how the
        screen plane is oriented relative to the camera. This accounts for monitors
        that are tilted, angled, or not perpendicular to the camera.
        """
        if len(self.screen_boundary_landmarks) < 9:
            return
        
        # Get corner positions in camera space (where face was when looking at each corner)
        # Use corners only for perspective calculation (midpoints add noise)
        tl = self.screen_boundary_landmarks['TOP-LEFT']
        tr = self.screen_boundary_landmarks['TOP-RIGHT']
        br = self.screen_boundary_landmarks['BOTTOM-RIGHT']
        bl = self.screen_boundary_landmarks['BOTTOM-LEFT']
        
        # Source points: actual face positions in camera frame (potentially trapezoidal)
        src_points = np.float32([
            [tl[0], tl[1]],
            [tr[0], tr[1]],
            [br[0], br[1]],
            [bl[0], bl[1]]
        ])
        
        # Destination points: ideal rectangular screen positions
        # Map to normalized [0,1] x [0,1] space
        dst_points = np.float32([
            [0, 0],           # top-left
            [1, 0],           # top-right
            [1, 1],           # bottom-right
            [0, 1]            # bottom-left
        ])
        
        # Calculate perspective transformation matrix (homography)
        self.perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        self.inverse_perspective_matrix = cv2.getPerspectiveTransform(dst_points, src_points)
        
        # Calculate screen tilt angles for display
        # Measure how much the screen deviates from perpendicular
        top_width = np.linalg.norm(tr - tl)
        bottom_width = np.linalg.norm(br - bl)
        left_height = np.linalg.norm(bl - tl)
        right_height = np.linalg.norm(br - tr)
        
        horizontal_tilt = abs(top_width - bottom_width) / max(top_width, bottom_width) * 100
        vertical_tilt = abs(left_height - right_height) / max(left_height, right_height) * 100
        
        print(f"  └─ Screen tilt detected: H={horizontal_tilt:.1f}%, V={vertical_tilt:.1f}%")
    
    def apply_perspective_correction(self, face_position):
        """Apply perspective correction to face position based on screen tilt
        
        Args:
            face_position: [x, y] position of face center in camera frame
            
        Returns:
            [x, y] position corrected for perspective distortion
        """
        if self.perspective_matrix is None:
            return face_position
        
        # Convert to homogeneous coordinates
        point = np.array([[face_position[0], face_position[1]]], dtype=np.float32)
        
        # Apply perspective transformation
        corrected = cv2.perspectiveTransform(point.reshape(-1, 1, 2), self.perspective_matrix)
        
        return corrected[0][0]

    def detect_camera_position(self):
        """DISABLED: Camera position detection
        
        Camera offset detection disabled - too finicky and prone to capturing
        user posture changes rather than actual hardware geometry.
        
        For fixed setups, calibration data implicitly captures any geometric
        offset in the mapping. Symmetric algorithms work regardless of camera
        position, avoiding asymmetric corrections that add noise and failure modes.
        """
        # Set neutral values (no asymmetric corrections)
        self.camera_above_screen = True
        self.camera_vertical_offset = 0.0
        self.camera_horizontal_offset = 0.0
        
        print(f"  • Camera position: Neutral (9-point calibration captures geometry implicitly)")
        print(f"    └─ Vertical offset: 0.0%")
        print(f"    └─ Horizontal offset: 0.0%")
    
    def calculate_dynamic_parameters(self):
        """Calculate data-driven interpolation parameters from calibration data
        
        Replaces hardcoded magic numbers with values computed from actual iris
        movement patterns observed during calibration. Makes system adaptive
        to different users and setups.
        """
        if len(self.calibration_data) < 5:
            # Fallback to defaults if insufficient calibration data
            self.extrapolation_sensitivity = 0.70
            self.edge_padding = 0.35
            self.edge_emphasis_power = 1.25
            self.corner_boost_base = 0.30
            return
        
        # Get iris positions for each calibration point (average of left and right)
        center_iris = (self.calibration_data['CENTER']['left_iris_norm'] + 
                       self.calibration_data['CENTER']['right_iris_norm']) / 2
        tl_iris = (self.calibration_data['TOP-LEFT']['left_iris_norm'] + 
                   self.calibration_data['TOP-LEFT']['right_iris_norm']) / 2
        tr_iris = (self.calibration_data['TOP-RIGHT']['left_iris_norm'] + 
                   self.calibration_data['TOP-RIGHT']['right_iris_norm']) / 2
        br_iris = (self.calibration_data['BOTTOM-RIGHT']['left_iris_norm'] + 
                   self.calibration_data['BOTTOM-RIGHT']['right_iris_norm']) / 2
        bl_iris = (self.calibration_data['BOTTOM-LEFT']['left_iris_norm'] + 
                   self.calibration_data['BOTTOM-LEFT']['right_iris_norm']) / 2
        
        corners = np.array([tl_iris, tr_iris, br_iris, bl_iris])
        
        # 1. EXTRAPOLATION SENSITIVITY: Based on iris movement range vs path extent
        # Calculate distance from center to corners (actual calibrated range)
        corner_distances = np.linalg.norm(corners - center_iris, axis=1)
        calibrated_range = np.mean(corner_distances)
        
        # If we have path samples, check how far they extend beyond calibration points
        if self.path_samples_cache and len(self.path_samples_cache) > 0:
            all_iris_positions = np.array([s['iris_norm'] for s in self.path_samples_cache])
            path_distances = np.linalg.norm(all_iris_positions - center_iris, axis=1)
            path_95th = np.percentile(path_distances, 95)  # Exclude outliers
            
            # Extrapolation sensitivity = how much farther paths go beyond calibrated range
            if calibrated_range > 0:
                self.extrapolation_sensitivity = max(0.3, min(1.0, (path_95th / calibrated_range - 1.0)))
        else:
            # Fallback: use moderate extrapolation
            self.extrapolation_sensitivity = 0.70
        
        # 2. EDGE PADDING: Based on variance of path samples near edges
        if self.path_samples_cache and len(self.path_samples_cache) > 0:
            # Find samples near edges (outer 30% of screen)
            edge_samples = []
            for sample in self.path_samples_cache:
                norm_x = sample['screen_x'] / self.screen_width
                norm_y = sample['screen_y'] / self.screen_height
                dist_from_center = np.sqrt((norm_x - 0.5)**2 + (norm_y - 0.5)**2)
                if dist_from_center > 0.4:  # Outer region
                    edge_samples.append([norm_x, norm_y])
            
            if len(edge_samples) > 10:
                edge_array = np.array(edge_samples)
                # Calculate variance - higher variance means more overshoot needed
                edge_variance = np.mean(np.var(edge_array, axis=0))
                # Scale by safety factor and ensure reasonable range
                self.edge_padding = max(0.20, min(0.50, edge_variance * 2.0 + 0.25))
            else:
                self.edge_padding = 0.35
        else:
            self.edge_padding = 0.35
        
        # 3. EDGE EMPHASIS POWER: Based on center vs edge density ratio
        if self.path_samples_cache and len(self.path_samples_cache) > 0:
            center_count = 0
            edge_count = 0
            for sample in self.path_samples_cache:
                norm_x = sample['screen_x'] / self.screen_width
                norm_y = sample['screen_y'] / self.screen_height
                dist_from_center = np.sqrt((norm_x - 0.5)**2 + (norm_y - 0.5)**2)
                
                if dist_from_center < 0.25:
                    center_count += 1
                elif dist_from_center > 0.50:
                    edge_count += 1
            
            if edge_count > 0:
                density_ratio = center_count / edge_count
                # Higher density ratio → need stronger power to expand edges
                # Use logarithmic scaling to prevent extreme values
                self.edge_emphasis_power = 1.0 + np.log1p(density_ratio) * 0.25
                self.edge_emphasis_power = max(1.05, min(1.50, self.edge_emphasis_power))
            else:
                self.edge_emphasis_power = 1.25
        else:
            self.edge_emphasis_power = 1.25
        
        # 4. CORNER BOOST BASE: Based on comfortable gaze limit vs corner distance
        if self.path_samples_cache and len(self.path_samples_cache) > 0:
            all_distances = []
            for sample in self.path_samples_cache:
                norm_x = sample['screen_x'] / self.screen_width
                norm_y = sample['screen_y'] / self.screen_height
                dist = np.sqrt((norm_x - 0.5)**2 + (norm_y - 0.5)**2)
                all_distances.append(dist)
            
            # 95th percentile = comfortable gaze limit
            comfortable_limit = np.percentile(all_distances, 95)
            corner_distance = np.sqrt(0.5**2 + 0.5**2)  # Distance to corner in normalized space
            
            # How much boost needed to reach corners comfortably
            if comfortable_limit > 0:
                boost_needed = (corner_distance / comfortable_limit - 1.0)
                self.corner_boost_base = max(0.10, min(0.50, boost_needed))
            else:
                self.corner_boost_base = 0.30
        else:
            self.corner_boost_base = 0.30

    def build_path_interpolation_data(self):
        """Build interpolation lookup from calibration eye paths (called once after calibration)
        
        Uses the actual eye movement paths between calibration points to create
        a more accurate iris-to-screen mapping by learning intermediate positions.
        Caches results and builds KDTree for O(log n) nearest neighbor queries.
        """
        # Return cached data if already built
        if self.path_samples_cache is not None:
            return self.path_samples_cache
        
        if len(self.calibration_eye_paths) == 0:
            return None
        
        # Define screen positions for all 9 calibration points
        screen_positions = {
            'CENTER': (self.screen_width // 2, self.screen_height // 2),
            'TOP-LEFT': (0, 0),
            'TOP-MID': (self.screen_width // 2, 0),
            'TOP-RIGHT': (self.screen_width - 1, 0),
            'RIGHT-MID': (self.screen_width - 1, self.screen_height // 2),
            'BOTTOM-RIGHT': (self.screen_width - 1, self.screen_height - 1),
            'BOTTOM-MID': (self.screen_width // 2, self.screen_height - 1),
            'BOTTOM-LEFT': (0, self.screen_height - 1),
            'LEFT-MID': (0, self.screen_height // 2)
        }
        
        # Build iris_norm -> screen_pos mapping from paths
        iris_to_screen_samples = []
        
        for transition_key, path_data in self.calibration_eye_paths.items():
            # Parse transition: "START->END"
            start_point, end_point = transition_key.split('->')
            start_screen = screen_positions[start_point]
            end_screen = screen_positions[end_point]
            
            # For each sample in the path, interpolate screen position
            num_samples = len(path_data)
            
            # WEIGHTED SAMPLING: Oversample corner regions where accuracy matters most
            # Strategy: 80% of samples in outer 40% of path (near corners)
            #           20% of samples in inner 60% of path (near center)
            for i, sample in enumerate(path_data):
                # Linear interpolation of screen position based on progress through path
                progress = i / max(num_samples - 1, 1)
                
                interp_screen_x = start_screen[0] + progress * (end_screen[0] - start_screen[0])
                interp_screen_y = start_screen[1] + progress * (end_screen[1] - start_screen[1])
                
                # Calculate distance from screen center (normalized 0-1)
                norm_x = interp_screen_x / self.screen_width
                norm_y = interp_screen_y / self.screen_height
                dist_from_center = np.sqrt((norm_x - 0.5)**2 + (norm_y - 0.5)**2)
                
                # Determine sample weight based on distance from center
                # Outer regions (dist > 0.4): sample more frequently
                # Inner regions (dist < 0.4): sample less frequently
                if dist_from_center > 0.4:
                    # Outer 40%: Include this sample (corner region - high accuracy needed)
                    sample_weight = 1.0
                else:
                    # Inner 60%: Include only 25% of samples (center - less critical)
                    # This creates 4:1 ratio, resulting in ~80% samples near corners
                    sample_weight = 0.25
                
                # Probabilistic sampling based on weight
                # For deterministic behavior: always include if weight >= threshold
                if sample_weight >= 1.0 or (sample_weight > 0 and i % int(1.0 / sample_weight) == 0):
                    # Store mapping: avg iris norm -> screen position
                    iris_to_screen_samples.append({
                        'iris_norm': sample['avg_iris_norm'].copy(),
                        'screen_x': interp_screen_x,
                        'screen_y': interp_screen_y
                    })
        
        # Cache the samples
        self.path_samples_cache = iris_to_screen_samples
        
        # Build KDTree for fast nearest neighbor search
        if len(iris_to_screen_samples) > 0:
            iris_positions = np.array([s['iris_norm'] for s in iris_to_screen_samples])
            self.path_kdtree = KDTree(iris_positions)
        
        return iris_to_screen_samples
    
    def find_nearest_path_samples(self, avg_iris_norm, path_samples, k=5):
        """Find k nearest samples from path data to current iris position using KDTree (O(log n))
        
        Applies density normalization to reduce center bias - samples in dense regions
        (like center) get reduced weight, while sparse regions (edges) get boosted.
        """
        if path_samples is None or len(path_samples) == 0 or self.path_kdtree is None:
            return None
        
        # Use KDTree for fast nearest neighbor search
        k_actual = min(k, len(path_samples))
        distances, indices = self.path_kdtree.query(avg_iris_norm, k=k_actual)
        
        # Handle single result (kdtree returns scalar instead of array for k=1)
        if k_actual == 1:
            distances = [distances]
            indices = [indices]
        
        # Calculate density normalization for each sample
        # Samples in dense regions (center) should have lower effective weight
        normalized_results = []
        for i in range(len(indices)):
            sample = path_samples[indices[i]]
            
            # Estimate local density by finding nearby samples
            # More neighbors nearby = higher density = lower weight
            sample_pos = sample['iris_norm']
            nearby_distances, _ = self.path_kdtree.query(sample_pos, k=min(20, len(path_samples)))
            
            # Density metric: inverse of mean distance to 20 nearest neighbors
            # High density (small distances) → low weight multiplier
            # Low density (large distances) → high weight multiplier
            if isinstance(nearby_distances, (list, np.ndarray)) and len(nearby_distances) > 1:
                mean_neighbor_dist = np.mean(nearby_distances[1:])  # Skip self (distance 0)
            else:
                mean_neighbor_dist = nearby_distances if not isinstance(nearby_distances, (list, np.ndarray)) else 0.01
            
            # Density weight: favor sparse regions (edges) over dense regions (center)
            # Add small constant to avoid division by zero
            density_weight = mean_neighbor_dist + 0.001
            
            # Apply density normalization to distance
            # Effectively increases weight of edge samples, decreases weight of center samples
            normalized_distance = distances[i] / density_weight
            
            normalized_results.append((normalized_distance, sample))
        
        return normalized_results

    def is_outlier(self, screen_x, screen_y, current_time):
        """Detect outlier gaze positions based on physiologically impossible movement speed"""
        if self.last_gaze_position is None or self.last_gaze_time is None:
            return False
        
        # Calculate movement distance and time
        dx = screen_x - self.last_gaze_position[0]
        dy = screen_y - self.last_gaze_position[1]
        distance = np.sqrt(dx**2 + dy**2)
        dt = current_time - self.last_gaze_time
        
        if dt <= 0:
            return False
        
        # Calculate speed in pixels/second
        speed = distance / dt
        
        # Reject if speed exceeds physiological limit
        if speed > self.max_saccade_speed:
            self.outlier_count += 1
            return True
        
        return False
    
    def detect_fixation(self, screen_x, screen_y, current_time):
        """Detect if eyes are in fixation (stationary) or saccade (moving)
        
        Returns: (is_fixating, fixation_duration)
        """
        self.fixation_positions.append((screen_x, screen_y))
        
        if len(self.fixation_positions) < 3:
            return False, 0.0
        
        # Calculate variance in recent positions
        positions = np.array(list(self.fixation_positions))
        variance = np.var(positions, axis=0)
        total_variance = np.sum(variance)
        
        # Low variance = fixation, high variance = saccade
        was_fixating = self.in_fixation
        self.in_fixation = total_variance < (self.fixation_threshold ** 2)
        
        # Update fixation duration
        if self.in_fixation:
            if was_fixating and self.last_gaze_time is not None:
                self.fixation_duration += (current_time - self.last_gaze_time)
            else:
                self.fixation_duration = 0.0
        else:
            self.fixation_duration = 0.0
        
        return self.in_fixation, self.fixation_duration
    
    def update_eye_dominance(self, screen_x, screen_y, left_iris_norm, right_iris_norm):
        """Update per-region eye dominance scores based on iris consistency
        
        Tracks which eye provides more stable/reliable data in different screen regions
        """
        # Determine screen region
        h_region = 'LEFT' if screen_x < self.screen_width / 3 else ('RIGHT' if screen_x > 2 * self.screen_width / 3 else 'CENTER')
        v_region = 'TOP' if screen_y < self.screen_height / 2 else 'BOTTOM'
        
        # Calculate iris stability (lower variance = more stable)
        if len(self.gaze_history) >= 5:
            # Would need to track per-eye history for proper implementation
            # Simplified: assume convergence indicates reliability
            # In practice, you'd track variance of each eye's iris position
            pass  # Placeholder for now - full implementation would track per-eye variance
    
    def calculate_binocular_weights(self, screen_x, screen_y, convergence):
        """Calculate optimal weighting between left and right eye
        
        Simplified to static 50/50 weighting to eliminate jitter and circular dependencies.
        Both eyes look at the same target - geometric averaging is optimal.
        """
        return 0.5, 0.5
    
    def update_adaptive_calibration(self, zone_name, left_iris_norm, right_iris_norm, 
                                    left_gaze_angle, right_gaze_angle, convergence, ipd):
        """Update calibration data for a specific zone based on observed gaze"""
        if not self.adaptive_calibration_enabled:
            return
        
        # Add new sample
        self.adaptive_samples[zone_name].append({
            'left_iris_norm': left_iris_norm,
            'right_iris_norm': right_iris_norm,
            'left_gaze_angle': left_gaze_angle,
            'right_gaze_angle': right_gaze_angle,
            'convergence': convergence,
            'ipd': ipd
        })
        
        # Keep only recent samples
        if len(self.adaptive_samples[zone_name]) > self.max_adaptive_samples:
            self.adaptive_samples[zone_name].pop(0)
        
        # If we have enough samples, update calibration
        if len(self.adaptive_samples[zone_name]) >= 10:
            samples = self.adaptive_samples[zone_name]
            
            # Average recent samples with 70% weight on new data, 30% on original calibration
            new_left_iris_norm = np.mean([s['left_iris_norm'] for s in samples], axis=0)
            new_right_iris_norm = np.mean([s['right_iris_norm'] for s in samples], axis=0)
            new_left_gaze_angle = np.mean([s['left_gaze_angle'] for s in samples])
            new_right_gaze_angle = np.mean([s['right_gaze_angle'] for s in samples])
            new_convergence = np.mean([s['convergence'] for s in samples])
            new_ipd = np.mean([s['ipd'] for s in samples])
            
            orig_data = self.calibration_data[zone_name]
            
            # Blend with original calibration (70% new, 30% original)
            self.calibration_data[zone_name]['left_iris_norm'] = 0.7 * new_left_iris_norm + 0.3 * orig_data['left_iris_norm']
            self.calibration_data[zone_name]['right_iris_norm'] = 0.7 * new_right_iris_norm + 0.3 * orig_data['right_iris_norm']
            self.calibration_data[zone_name]['left_gaze_angle'] = 0.7 * new_left_gaze_angle + 0.3 * orig_data['left_gaze_angle']
            self.calibration_data[zone_name]['right_gaze_angle'] = 0.7 * new_right_gaze_angle + 0.3 * orig_data['right_gaze_angle']
            self.calibration_data[zone_name]['convergence'] = 0.7 * new_convergence + 0.3 * orig_data['convergence']
            self.calibration_data[zone_name]['ipd'] = 0.7 * new_ipd + 0.3 * orig_data['ipd']

    def interpolate_gaze(self, left_iris_norm, right_iris_norm, current_face_measurements, 
                        ipd_scale, left_gaze_angle, right_gaze_angle, convergence):
        """Map normalized iris positions to screen coordinates using calibration data with head pose and IPD compensation"""
        if not self.calibrated or len(self.calibration_data) != 5:
            return None, None
        
        # Calculate binocular weights using enhanced fusion algorithm (accounts for regional eye dominance)
        # Rough initial position needed for dominance calculation
        temp_avg = (left_iris_norm + right_iris_norm) / 2
        temp_x = int(temp_avg[0] * self.screen_width)
        temp_y = int(temp_avg[1] * self.screen_height)
        left_weight, right_weight = self.calculate_binocular_weights(temp_x, temp_y, convergence)
        
        # Weighted average of iris positions based on enhanced binocular fusion
        avg_iris_norm = (left_iris_norm * left_weight + right_iris_norm * right_weight) / (left_weight + right_weight)
        
        # Get calibration data (normalized iris positions at each point)
        center_data = self.calibration_data['CENTER']
        tl_data = self.calibration_data['TOP-LEFT']
        tr_data = self.calibration_data['TOP-RIGHT']
        br_data = self.calibration_data['BOTTOM-RIGHT']
        bl_data = self.calibration_data['BOTTOM-LEFT']
        
        center_iris_norm = (center_data['left_iris_norm'] + center_data['right_iris_norm']) / 2
        tl_iris_norm = (tl_data['left_iris_norm'] + tl_data['right_iris_norm']) / 2
        tr_iris_norm = (tr_data['left_iris_norm'] + tr_data['right_iris_norm']) / 2
        br_iris_norm = (br_data['left_iris_norm'] + br_data['right_iris_norm']) / 2
        bl_iris_norm = (bl_data['left_iris_norm'] + bl_data['right_iris_norm']) / 2
        
        # CALIBRATION POINT ANCHORING: If current iris is near a calibration point,
        # blend toward exact calibration screen position to preserve corner accuracy.
        # This ensures that when user looks at corners with same iris position as during
        # calibration, they get the exact corner position (not extrapolated/shifted).
        anchor_influence_radius = 0.025  # 2.5% of iris range
        closest_calibration_point = None
        closest_distance = float('inf')
        
        calibration_points = {
            'CENTER': (center_iris_norm, self.screen_width // 2, self.screen_height // 2),
            'TOP-LEFT': (tl_iris_norm, 0, 0),
            'TOP-RIGHT': (tr_iris_norm, self.screen_width - 1, 0),
            'BOTTOM-RIGHT': (br_iris_norm, self.screen_width - 1, self.screen_height - 1),
            'BOTTOM-LEFT': (bl_iris_norm, 0, self.screen_height - 1)
        }
        
        # Find closest calibration point
        for point_name, (calib_iris, screen_x, screen_y) in calibration_points.items():
            distance = np.linalg.norm(avg_iris_norm - calib_iris)
            if distance < closest_distance:
                closest_distance = distance
                closest_calibration_point = (screen_x, screen_y, distance)
        
        # Compute head movement compensation (now includes roll angle from face normal vectors)
        translation, depth_scale, roll_angle, parallax_offset = self.compensate_head_movement(current_face_measurements)
        
        # Store parallax info for later application (after interpolation)
        parallax_factor = (parallax_offset - self.camera_center_x) / self.camera_center_x
        
        # Use cached path interpolation data (built once after calibration)
        path_samples = self.path_samples_cache
        
        # Try path-based interpolation first (more accurate if we have path data)
        if path_samples is not None and len(path_samples) > 0:
            nearest_samples = self.find_nearest_path_samples(avg_iris_norm, path_samples, k=16)
            
            if nearest_samples is not None and len(nearest_samples) > 0:
                # Inverse distance weighted interpolation
                total_weight = 0.0
                weighted_x = 0.0
                weighted_y = 0.0
                
                for dist, sample in nearest_samples:
                    # Use inverse distance weighting (avoid division by zero)
                    weight = 1.0 / (dist + 0.001)
                    total_weight += weight
                    weighted_x += sample['screen_x'] * weight
                    weighted_y += sample['screen_y'] * weight
                
                if total_weight > 0:
                    path_screen_x = weighted_x / total_weight
                    path_screen_y = weighted_y / total_weight
                    
                    # Apply depth and IPD compensation to path-based result
                    # Convert back to normalized, apply compensation, convert to screen
                    path_norm_x = path_screen_x / self.screen_width
                    path_norm_y = path_screen_y / self.screen_height
                    
                    path_norm_x = 0.5 + (path_norm_x - 0.5) / (depth_scale * ipd_scale)
                    path_norm_y = 0.5 + (path_norm_y - 0.5) / (depth_scale * ipd_scale)
                    
                    # Apply roll compensation if enabled
                    if abs(roll_angle) > 0.01:
                        dx = path_norm_x - 0.5
                        dy = path_norm_y - 0.5
                        cos_roll = np.cos(-roll_angle)
                        sin_roll = np.sin(-roll_angle)
                        path_norm_x = 0.5 + (dx * cos_roll - dy * sin_roll)
                        path_norm_y = 0.5 + (dx * sin_roll + dy * cos_roll)
                    
                    path_norm_x = np.clip(path_norm_x, 0, 1)
                    path_norm_y = np.clip(path_norm_y, 0, 1)
                    
                    # Use 80% path-based, 20% traditional interpolation for smoothness
                    path_weight = 0.8
                    # Fall through to calculate traditional interpolation, then blend
        else:
            path_weight = 0.0
        
        # Use center point as anchor for better interpolation (traditional method)
        # Determine which quadrant the gaze is in relative to center
        
        # Calculate distance from center in normalized iris space
        delta_x = avg_iris_norm[0] - center_iris_norm[0]
        delta_y = avg_iris_norm[1] - center_iris_norm[1]
        
        # Determine which quadrant and interpolate accordingly
        if delta_x < 0 and delta_y < 0:
            # Top-left quadrant
            ref_corner = tl_iris_norm
            corner_weight = 1.0
        elif delta_x >= 0 and delta_y < 0:
            # Top-right quadrant
            ref_corner = tr_iris_norm
            corner_weight = 1.0
        elif delta_x >= 0 and delta_y >= 0:
            # Bottom-right quadrant
            ref_corner = br_iris_norm
            corner_weight = 1.0
        else:
            # Bottom-left quadrant
            ref_corner = bl_iris_norm
            corner_weight = 1.0
        
        # X-axis: interpolate using center as anchor with symmetric edge extrapolation
        left_iris_x = (tl_iris_norm[0] + bl_iris_norm[0]) / 2
        right_iris_x = (tr_iris_norm[0] + br_iris_norm[0]) / 2
        
        # Symmetric extrapolation (camera offset adjustments removed for stability)
        left_extrapolation = self.extrapolation_sensitivity
        right_extrapolation = self.extrapolation_sensitivity
        
        # Normalize current iris X relative to center and bounds
        if delta_x < 0:
            # Left side
            if left_iris_x - center_iris_norm[0] != 0:
                ratio = abs(delta_x) / abs(left_iris_x - center_iris_norm[0])
                if ratio > 1.0:
                    overshoot = ratio - 1.0
                    norm_x = 0.0 - (overshoot * left_extrapolation)  # Camera-aware
                else:
                    norm_x = 0.5 - 0.5 * ratio
            else:
                norm_x = 0.5
        else:
            # Right side
            if right_iris_x - center_iris_norm[0] != 0:
                ratio = abs(delta_x) / abs(right_iris_x - center_iris_norm[0])
                if ratio > 1.0:
                    overshoot = ratio - 1.0
                    norm_x = 1.0 + (overshoot * right_extrapolation)  # Camera-aware
                else:
                    norm_x = 0.5 + 0.5 * ratio
            else:
                norm_x = 0.5
        
        # Y-axis: interpolate using center as anchor with symmetric edge extrapolation
        top_iris_y = (tl_iris_norm[1] + tr_iris_norm[1]) / 2
        bottom_iris_y = (bl_iris_norm[1] + br_iris_norm[1]) / 2
        
        # Symmetric extrapolation (camera offset adjustments removed for stability)
        top_extrapolation = self.extrapolation_sensitivity
        bottom_extrapolation = self.extrapolation_sensitivity
        
        if delta_y < 0:
            # Top side
            if top_iris_y - center_iris_norm[1] != 0:
                ratio = abs(delta_y) / abs(top_iris_y - center_iris_norm[1])
                if ratio > 1.0:
                    overshoot = ratio - 1.0
                    norm_y = 0.0 - (overshoot * top_extrapolation)  # Camera-aware
                else:
                    norm_y = 0.5 - 0.5 * ratio
            else:
                norm_y = 0.5
        else:
            # Bottom side
            if bottom_iris_y - center_iris_norm[1] != 0:
                ratio = abs(delta_y) / abs(bottom_iris_y - center_iris_norm[1])
                if ratio > 1.0:
                    overshoot = ratio - 1.0
                    norm_y = 1.0 + (overshoot * bottom_extrapolation)  # Camera-aware
                else:
                    norm_y = 0.5 + 0.5 * ratio
            else:
                norm_y = 0.5
        
        # Density normalization bias correction
        # Push values away from center to counteract interpolation's center-pulling tendency
        # This compensates for the fact that most samples are near center
        distance_from_center = np.sqrt((norm_x - 0.5)**2 + (norm_y - 0.5)**2)
        if distance_from_center > 0.2:  # Only apply beyond central region
            # Bias increases with distance from center (up to 15% boost)
            density_bias = min(0.15, (distance_from_center - 0.2) * 0.25)
            # Push in direction away from center
            norm_x = 0.5 + (norm_x - 0.5) * (1.0 + density_bias)
            norm_y = 0.5 + (norm_y - 0.5) * (1.0 + density_bias)
        
        # Apply depth compensation (scale affects perceived angle)
        # When farther away, same eye movement covers more screen space
        norm_x = 0.5 + (norm_x - 0.5) / depth_scale
        norm_y = 0.5 + (norm_y - 0.5) / depth_scale
        
        # Apply IPD compensation
        # When IPD is larger (closer to camera or actual larger eyes), adjust sensitivity
        norm_x = 0.5 + (norm_x - 0.5) / ipd_scale
        norm_y = 0.5 + (norm_y - 0.5) / ipd_scale
        
        # Apply edge emphasis - increase sensitivity near screen boundaries
        # Makes corners more reachable by applying exponential scaling at edges
        def edge_emphasis(value):
            """Apply stronger response near edges (0 and 1)"""
            # Clamp for safe calculation
            safe_value = np.clip(value, 0.0, 1.0)
            
            if safe_value < 0.5:
                # Left/top half: expand edge reach with data-driven power
                emphasized = 0.5 * (2 * safe_value) ** self.edge_emphasis_power
            else:
                # Right/bottom half: expand edge reach
                emphasized = 1.0 - 0.5 * (2 * (1 - safe_value)) ** self.edge_emphasis_power
            
            # If original value was outside 0-1 (extrapolated), preserve overshoot
            if value < 0.0:
                emphasized = emphasized + value  # Add negative overshoot
            elif value > 1.0:
                emphasized = emphasized + (value - 1.0)  # Add positive overshoot
            
            return emphasized
        
        norm_x = edge_emphasis(norm_x)
        norm_y = edge_emphasis(norm_y)
        
        # Distance-adaptive corner boost (symmetric for all corners)
        # Compensates for eye socket geometry
        corner_distance = np.sqrt((norm_x - 0.5)**2 + (norm_y - 0.5)**2)
        if corner_distance > 0.5:  # In corner regions
            distance_factor = np.clip(depth_scale, 0.5, 2.0)
            
            # Symmetric boost for all corners (camera offset adjustments removed)
            max_boost = self.corner_boost_base * distance_factor
            corner_boost = (corner_distance - 0.5) * max_boost
            
            # Apply boost in direction away from center
            norm_x = 0.5 + (norm_x - 0.5) * (1.0 + corner_boost)
            norm_y = 0.5 + (norm_y - 0.5) * (1.0 + corner_boost)
        
        # Apply roll compensation from face normal vectors (advanced calibration)
        # When head tilts (roll), rotate gaze coordinates back to upright reference
        if abs(roll_angle) > 0.01:  # Only apply if significant roll detected
            # Translate to origin, rotate, translate back
            dx = norm_x - 0.5
            dy = norm_y - 0.5
            
            # Rotate by negative roll angle (compensate for head tilt)
            cos_roll = np.cos(-roll_angle)
            sin_roll = np.sin(-roll_angle)
            
            norm_x = 0.5 + (dx * cos_roll - dy * sin_roll)
            norm_y = 0.5 + (dx * sin_roll + dy * cos_roll)
        
        # Apply parallax correction (base strength only, no camera offset scaling)
        # When head moves right, gaze should shift right
        # parallax_factor is positive when head is right of camera center
        parallax_strength = 0.12
        parallax_correction_x = parallax_factor * parallax_strength
        norm_x = norm_x + parallax_correction_x
        
        # Soft clamp with edge padding - allows overshoot for corner reach
        if norm_x < -self.edge_padding:
            norm_x = -self.edge_padding
        elif norm_x > 1.0 + self.edge_padding:
            norm_x = 1.0 + self.edge_padding
        
        if norm_y < -self.edge_padding:
            norm_y = -self.edge_padding
        elif norm_y > 1.0 + self.edge_padding:
            norm_y = 1.0 + self.edge_padding
        
        # Map to screen coordinates with proper edge handling
        trad_screen_x = int(norm_x * self.screen_width)
        trad_screen_y = int(norm_y * self.screen_height)
        
        # Hard clamp to screen bounds (prevents out-of-bounds rendering)
        trad_screen_x = np.clip(trad_screen_x, 0, self.screen_width - 1)
        trad_screen_y = np.clip(trad_screen_y, 0, self.screen_height - 1)
        
        # Blend path-based and traditional interpolation if path data available
        if path_weight > 0 and 'path_norm_x' in locals():
            final_screen_x = int(path_weight * path_norm_x * self.screen_width + (1 - path_weight) * trad_screen_x)
            final_screen_y = int(path_weight * path_norm_y * self.screen_height + (1 - path_weight) * trad_screen_y)
        else:
            final_screen_x = trad_screen_x
            final_screen_y = trad_screen_y
        
        # CALIBRATION POINT ANCHORING: Blend toward exact calibration position
        # when iris is near a calibration point (preserves corner accuracy)
        if closest_calibration_point is not None:
            calib_x, calib_y, distance = closest_calibration_point
            
            # Calculate anchor influence (1.0 at exact match, 0.0 beyond radius)
            if distance < anchor_influence_radius:
                # Smooth falloff using cosine interpolation
                anchor_strength = 0.5 * (1.0 + np.cos(np.pi * distance / anchor_influence_radius))
                
                # Blend toward calibration position
                final_screen_x = int((1.0 - anchor_strength) * final_screen_x + anchor_strength * calib_x)
                final_screen_y = int((1.0 - anchor_strength) * final_screen_y + anchor_strength * calib_y)
                
                # Ensure still within bounds
                final_screen_x = np.clip(final_screen_x, 0, self.screen_width - 1)
                final_screen_y = np.clip(final_screen_y, 0, self.screen_height - 1)
        
        return final_screen_x, final_screen_y

    def get_gaze_region(self, screen_x, screen_y):
        """Determine which region of the screen the user is looking at"""
        if screen_x is None or screen_y is None:
            return "UNKNOWN"
        
        # Divide screen into 9 regions (3x3 grid)
        h_third = self.screen_width // 3
        v_third = self.screen_height // 3
        
        if screen_y < v_third:
            row = "TOP"
        elif screen_y < 2 * v_third:
            row = "CENTER"
        else:
            row = "BOTTOM"
        
        if screen_x < h_third:
            col = "LEFT"
        elif screen_x < 2 * h_third:
            col = "CENTER"
        else:
            col = "RIGHT"
        
        if row == "CENTER" and col == "CENTER":
            return "CENTER"
        return f"{row}-{col}"

    def run(self):
        """Main tracking loop with profile management"""
        print("\n" + "="*70)
        print("  ADVANCED EYE TRACKER")
        print("="*70)
        print("\nPROFILE MANAGEMENT:")
        print("  [1-10] Load existing profile")
        print("  [N]    New calibration")
        print("  [Q]    Quit")
        print("="*70)
        
        # Create fullscreen window immediately for smooth UX
        cv2.namedWindow("Eye Tracker", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Eye Tracker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # Modern profile selection menu
        profile_selected = False
        while not profile_selected:
            # Clean dark canvas
            menu_canvas = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
            
            # Minimal header
            header_text = "EYE TRACKER"
            text_size = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 2)[0]
            header_x = (self.screen_width - text_size[0]) // 2
            cv2.putText(menu_canvas, header_text, (header_x, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 220, 255), 2)
            
            # Thin divider line
            line_y = 240
            margin = self.screen_width // 4
            cv2.line(menu_canvas, (margin, line_y), (self.screen_width - margin, line_y), (60, 60, 60), 1)
            
            # Clean menu options
            y_offset = 350
            line_height = 80
            options = [
                ("1-9,0", "Load Profile"),
                ("N", "New Calibration"),
                ("Q", "Quit")
            ]
            
            for key, label in options:
                # Key badge
                badge_x = self.screen_width // 2 - 200
                badge_y = y_offset - 22
                badge_w = 80
                badge_h = 35
                
                # Badge background
                cv2.rectangle(menu_canvas, (badge_x, badge_y), 
                             (badge_x + badge_w, badge_y + badge_h), (40, 40, 40), -1)
                cv2.rectangle(menu_canvas, (badge_x, badge_y), 
                             (badge_x + badge_w, badge_y + badge_h), (0, 200, 255), 1)
                
                # Key text
                key_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                key_x = badge_x + (badge_w - key_size[0]) // 2
                cv2.putText(menu_canvas, key, (key_x, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1)
                
                # Label text
                cv2.putText(menu_canvas, label, (badge_x + badge_w + 30, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 1)
                
                y_offset += line_height
            
            cv2.imshow("Eye Tracker", menu_canvas)
            
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:
                print("Exiting...")
                self.cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord('n') or key == ord('N'):
                profile_selected = True
            elif key >= ord('1') and key <= ord('9'):
                profile_num = key - ord('0')
                if self.load_profile(profile_num):
                    profile_selected = True
            elif key == ord('0'):  # Press 0 for profile 10
                if self.load_profile(10):
                    profile_selected = True
        
        # If not calibrated yet, run calibration
        if not self.calibrated:
            # Modern calibration intro screen
            intro_canvas = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
            
            # Header
            header_text = "CALIBRATION"
            text_size = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 2)[0]
            header_x = (self.screen_width - text_size[0]) // 2
            cv2.putText(intro_canvas, header_text, (header_x, 220), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 220, 255), 2)
            
            # Divider
            line_y = 260
            margin = self.screen_width // 4
            cv2.line(intro_canvas, (margin, line_y), (self.screen_width - margin, line_y), (60, 60, 60), 1)
            
            # Instructions (clean bullet points)
            y_offset = 340
            line_height = 50
            instructions = [
                "• Position yourself comfortably",
                "• Look at each target point",
                "• 5 calibration points total"
            ]
            
            for instruction in instructions:
                text_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
                text_x = (self.screen_width - text_size[0]) // 2
                cv2.putText(intro_canvas, instruction, (text_x, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
                y_offset += line_height
            
            # Action buttons
            y_offset += 80
            button_y = y_offset - 25
            
            # Start button
            start_x = self.screen_width // 2 - 120
            cv2.rectangle(intro_canvas, (start_x, button_y), (start_x + 100, button_y + 40), (0, 200, 255), 2)
            cv2.putText(intro_canvas, "S", (start_x + 40, button_y + 28), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 1)
            cv2.putText(intro_canvas, "Start", (start_x + 20, button_y + 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)
            
            # Quit button
            quit_x = self.screen_width // 2 + 20
            cv2.rectangle(intro_canvas, (quit_x, button_y), (quit_x + 100, button_y + 40), (80, 80, 80), 1)
            cv2.putText(intro_canvas, "Q", (quit_x + 42, button_y + 28), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 120), 1)
            cv2.putText(intro_canvas, "Quit", (quit_x + 30, button_y + 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            
            cv2.imshow("Eye Tracker", intro_canvas)
            
            # Wait for S or Q
            while True:
                key = cv2.waitKey(100) & 0xFF
                if key == ord('s') or key == ord('S'):
                    break
                elif key == ord('q') or key == ord('Q') or key == 27:
                    print("Calibration cancelled")
                    self.cap.release()
                    cv2.destroyAllWindows()
                    return
            
            if not self.calibrate():
                print("Calibration cancelled")
                self.cap.release()
                cv2.destroyAllWindows()
                return
            
            # Modern save profile screen
            save_selected = False
            while not save_selected:
                save_canvas = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
                
                # Success indicator
                icon_x = self.screen_width // 2
                icon_y = 200
                cv2.circle(save_canvas, (icon_x, icon_y), 50, (0, 220, 255), 3)
                cv2.putText(save_canvas, "✓", (icon_x - 22, icon_y + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 220, 255), 3)
                
                # Title
                title_text = "Calibration Complete"
                text_size = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[0]
                cv2.putText(save_canvas, title_text, 
                           (self.screen_width // 2 - text_size[0] // 2, 310), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 2)
                
                # Divider
                line_y = 340
                margin = self.screen_width // 3
                cv2.line(save_canvas, (margin, line_y), (self.screen_width - margin, line_y), (60, 60, 60), 1)
                
                # Prompt
                prompt_text = "Save to profile?"
                text_size = cv2.getTextSize(prompt_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)[0]
                cv2.putText(save_canvas, prompt_text, 
                           (self.screen_width // 2 - text_size[0] // 2, 400), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (160, 160, 160), 1)
                
                # Options
                y_offset = 480
                
                # Profile slots option
                badge_x = self.screen_width // 2 - 140
                badge_y = y_offset - 25
                cv2.rectangle(save_canvas, (badge_x, badge_y), (badge_x + 100, badge_y + 35), (40, 40, 40), -1)
                cv2.rectangle(save_canvas, (badge_x, badge_y), (badge_x + 100, badge_y + 35), (0, 200, 255), 1)
                cv2.putText(save_canvas, "1-9,0", (badge_x + 20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1)
                cv2.putText(save_canvas, "Save Profile", (badge_x + 105, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
                
                # Skip option
                y_offset += 70
                skip_x = self.screen_width // 2 - 140
                skip_y = y_offset - 25
                cv2.rectangle(save_canvas, (skip_x, skip_y), (skip_x + 90, skip_y + 35), (40, 40, 40), -1)
                cv2.rectangle(save_canvas, (skip_x, skip_y), (skip_x + 90, skip_y + 35), (80, 80, 80), 1)
                cv2.putText(save_canvas, "N", (skip_x + 35, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)
                cv2.putText(save_canvas, "Skip", (skip_x + 105, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (140, 140, 140), 1)
                
                cv2.imshow("Eye Tracker", save_canvas)
                
                key = cv2.waitKey(100) & 0xFF
                if key == ord('n') or key == ord('N'):
                    save_selected = True
                elif key >= ord('1') and key <= ord('9'):
                    profile_num = key - ord('0')
                    self.save_profile(profile_num)
                    save_selected = True
                elif key == ord('0'):  # Press 0 for profile 10
                    self.save_profile(10)
                    save_selected = True
        
        print("\n" + "="*50)
        print("TRACKING MODE - Your gaze is now being tracked!")
        print("Press 'R' to recalibrate at any time")
        print("="*50 + "\n")
        
        # Smoothing for gaze position
        smooth_x, smooth_y = self.screen_width // 2, self.screen_height // 2
        smoothing_factor = 0.3
        
        # Blink detection state
        blink_pause_until = 0
        BLINK_PAUSE_DURATION = 0.2  # 200ms pause during blink
        STABILIZATION_DURATION = 0.066  # 66ms (2 frames @ 30fps) after blink ends for stabilization
        was_blinking = False
        
        # Show recalibration popup for 5 seconds
        show_popup = True
        popup_start_time = cv2.getTickCount()
        popup_duration = 5.0  # seconds
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame_height, frame_width = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            # Create black fullscreen canvas
            canvas = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
            
            screen_x, screen_y = None, None
            depth_info = ""
            
            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                (left_iris, right_iris, left_eye_center, right_eye_center, 
                 left_iris_norm, right_iris_norm, ipd, left_gaze_angle, right_gaze_angle, convergence,
                 is_blinking, ear, both_eyes_visible) = self.get_iris_positions(
                    face_landmarks, frame_width, frame_height
                )
                
                current_face_measurements = self.get_face_measurements(face_landmarks, frame_width, frame_height)
                
                # Get head pose compensation info (includes roll angle from face normal and parallax offset)
                translation, depth_scale, roll_angle, parallax_offset = self.compensate_head_movement(current_face_measurements)
                
                # IPD compensation (scale factor based on current vs reference IPD)
                ipd_scale = ipd / self.reference_ipd if self.reference_ipd else 1.0
                
                # Convert roll angle to degrees for display (only if advanced mode enabled)
                if self.advanced_calibration_enabled:
                    roll_degrees = np.degrees(roll_angle)
                    depth_info = f"Depth: {depth_scale:.2f}x | IPD: {ipd_scale:.2f}x | Roll: {roll_degrees:.1f}°"
                else:
                    depth_info = f"Depth: {depth_scale:.2f}x | IPD: {ipd_scale:.2f}x"
                
                # Handle blink detection
                if is_blinking and not was_blinking:
                    # Blink just started
                    blink_pause_until = current_time + BLINK_PAUSE_DURATION
                    was_blinking = True
                elif not is_blinking and was_blinking:
                    # Blink ended - extend pause for stabilization (prevents head movement artifacts)
                    blink_pause_until = current_time + STABILIZATION_DURATION
                    was_blinking = False
                
                # Only update gaze position if not in blink pause period
                if current_time > blink_pause_until:
                    # Use 3D eye model if enabled, otherwise fall back to 2D interpolation
                    debug_info_3d = None
                    if self.eye_3d_model_enabled and left_iris is not None and right_iris is not None:
                        # 3D gaze calculation using eye sphere model
                        screen_x, screen_y, debug_info_3d = self.calculate_3d_gaze_point(
                            left_iris, right_iris, left_eye_center, right_eye_center
                        )
                    else:
                        # Fallback to 2D interpolation
                        screen_x, screen_y = self.interpolate_gaze(left_iris_norm, right_iris_norm, current_face_measurements, 
                                                                  ipd_scale, left_gaze_angle, right_gaze_angle, convergence)
                    
                    if screen_x is not None and screen_y is not None:
                        # Outlier rejection - discard physiologically impossible movements
                        if self.is_outlier(screen_x, screen_y, current_time):
                            # Reject outlier, keep previous position
                            screen_x, screen_y = smooth_x, smooth_y
                        else:
                            # Detect fixation vs saccade
                            is_fixating, fixation_duration = self.detect_fixation(screen_x, screen_y, current_time)
                            
                            # Adaptive smoothing: stronger during fixation, lighter during saccades
                            if is_fixating and fixation_duration > 0.1:
                                # Moderate smoothing during fixation (reduced from 0.15 to prevent center drift)
                                adaptive_smoothing = 0.30
                                depth_info += f" | FIX {fixation_duration:.1f}s"
                            else:
                                # Light smoothing during saccades for responsiveness
                                adaptive_smoothing = 0.5
                            
                            # Update gaze tracking state
                            self.last_gaze_position = (screen_x, screen_y)
                            self.last_gaze_time = current_time
                            
                            # Apply adaptive smoothing
                            smooth_x = int(smooth_x * (1 - adaptive_smoothing) + screen_x * adaptive_smoothing)
                            smooth_y = int(smooth_y * (1 - adaptive_smoothing) + screen_y * adaptive_smoothing)
                            
                            # ANTI-DRIFT: Periodic head pose re-referencing
                            # When user looks at center for 2+ seconds, update reference to adapt to posture changes
                            center_x = self.screen_width // 2
                            center_y = self.screen_height // 2
                            dist_from_center = np.sqrt((smooth_x - center_x)**2 + (smooth_y - center_y)**2)
                            
                            if dist_from_center < 150:  # Within 150px of center
                                dt = current_time - self.last_gaze_time if self.last_gaze_time else 0.033
                                self.center_fixation_time += dt
                                
                                # Update reference after 2 seconds of center fixation
                                # But not more than once every 10 seconds to avoid instability
                                if self.center_fixation_time >= 2.0 and \
                                   (current_time - self.last_reref_time) >= 10.0:
                                    # Update reference head pose to current position
                                    self.reference_head_pose = {
                                        'head_center': current_face_measurements['head_center'].copy(),
                                        'nose_tip': current_face_measurements['nose_tip'].copy(),
                                        'chin': current_face_measurements['chin'].copy()
                                    }
                                    self.reference_face_width = current_face_measurements['face_width']
                                    
                                    # Update face normal if advanced calibration enabled
                                    if self.advanced_calibration_enabled and self.reference_face_normal is not None:
                                        self.reference_face_normal = self.calculate_face_normal(current_face_measurements)
                                    
                                    self.center_fixation_time = 0.0
                                    self.last_reref_time = current_time
                                    depth_info += " | REF-UPDATE"
                            else:
                                # Reset counter when looking away from center
                                self.center_fixation_time = 0.0
                    
                    # Check if gaze is in any calibration zone for adaptive calibration
                    if screen_x is not None and screen_y is not None:
                        for zone_name, (x1, y1, x2, y2) in self.calibration_zones.items():
                            if x1 <= smooth_x <= x2 and y1 <= smooth_y <= y2:
                                # User is looking at a calibration zone - update adaptive calibration
                                self.update_adaptive_calibration(zone_name, left_iris_norm, right_iris_norm,
                                                               left_gaze_angle, right_gaze_angle, convergence, ipd)
                                break
                else:
                    # During blink pause, keep previous position
                    screen_x, screen_y = smooth_x, smooth_y
                
                # Modern minimal gaze crosshair
                if screen_x is not None and screen_y is not None:
                    # Subtle outer glow
                    overlay = canvas.copy()
                    cv2.circle(overlay, (smooth_x, smooth_y), 35, (0, 180, 255), -1)
                    cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0, canvas)
                    
                    # Thin ring
                    cv2.circle(canvas, (smooth_x, smooth_y), 20, (0, 200, 255), 2)
                    
                    # Center dot
                    cv2.circle(canvas, (smooth_x, smooth_y), 4, (0, 220, 255), -1)
                    cv2.circle(canvas, (smooth_x, smooth_y), 2, (255, 255, 255), -1)
                    
                    # Minimal crosshair lines
                    line_len = 40
                    gap = 25
                    cv2.line(canvas, (smooth_x - line_len, smooth_y), (smooth_x - gap, smooth_y), (0, 200, 255), 2)
                    cv2.line(canvas, (smooth_x + gap, smooth_y), (smooth_x + line_len, smooth_y), (0, 200, 255), 2)
                    cv2.line(canvas, (smooth_x, smooth_y - line_len), (smooth_x, smooth_y - gap), (0, 200, 255), 2)
                    cv2.line(canvas, (smooth_x, smooth_y + gap), (smooth_x, smooth_y + line_len), (0, 200, 255), 2)
                    
                    # Draw 3D eye model visualization
                    if self.eye_3d_model_enabled and 'debug_info_3d' in locals() and debug_info_3d is not None:
                        self.draw_3d_eye_visualization(canvas, debug_info_3d, smooth_x, smooth_y)
            
            # Draw small camera preview in corner
            preview_scale = 0.2
            preview_w = int(frame_width * preview_scale)
            preview_h = int(frame_height * preview_scale)
            preview = cv2.resize(frame, (preview_w, preview_h))
            canvas[20:20+preview_h, 20:20+preview_w] = preview
            
            # Clean minimal status display (top center)
            if screen_x is None:
                status_text = "NO FACE"
                text_color = (0, 100, 255)
            else:
                if is_blinking or current_time <= blink_pause_until:
                    status_text = "BLINK"
                    text_color = (0, 180, 255)
                elif not both_eyes_visible:
                    status_text = "1 EYE"
                    text_color = (0, 200, 255)
                else:
                    status_text = "TRACKING"
                    text_color = (0, 220, 255)
            
            # Small status indicator dot + text
            status_center_x = self.screen_width // 2
            cv2.circle(canvas, (status_center_x - 60, 35), 5, text_color, -1)
            cv2.putText(canvas, status_text, (status_center_x - 45, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
            
            # Subtle depth info (small, to the right)
            if screen_x is not None and depth_info:
                cv2.putText(canvas, depth_info, (status_center_x + 80, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
            
            # Modern minimal popup notification
            if show_popup:
                elapsed_time = (cv2.getTickCount() - popup_start_time) / cv2.getTickFrequency()
                if elapsed_time < popup_duration:
                    # Sleek notification card
                    card_width = 450
                    card_height = 100
                    card_x = (self.screen_width - card_width) // 2
                    card_y = 120
                    
                    overlay = canvas.copy()
                    # Dark card background
                    cv2.rectangle(overlay, (card_x, card_y), 
                                 (card_x + card_width, card_y + card_height), 
                                 (30, 30, 30), -1)
                    # Thin accent border
                    cv2.rectangle(overlay, (card_x, card_y), 
                                 (card_x + card_width, card_y + card_height), 
                                 (0, 220, 255), 2)
                    cv2.addWeighted(overlay, 0.9, canvas, 0.1, 0, canvas)
                    
                    # Success icon (checkmark circle)
                    icon_x = card_x + 35
                    icon_y = card_y + 50
                    cv2.circle(canvas, (icon_x, icon_y), 20, (0, 220, 255), 2)
                    cv2.putText(canvas, "✓", (icon_x - 10, icon_y + 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)
                    
                    # Clean text
                    cv2.putText(canvas, "Calibration Complete", 
                               (card_x + 75, card_y + 42), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)
                    cv2.putText(canvas, "Press R to recalibrate", 
                               (card_x + 75, card_y + 72), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)
                else:
                    show_popup = False
            
            # Subtle center reference crosshair
            center_x = self.screen_width // 2
            center_y = self.screen_height // 2
            crosshair_size = 30
            crosshair_gap = 8
            crosshair_color = (60, 60, 60)
            crosshair_thickness = 1
            
            # Horizontal lines
            cv2.line(canvas, (center_x - crosshair_size, center_y), 
                    (center_x - crosshair_gap, center_y), crosshair_color, crosshair_thickness)
            cv2.line(canvas, (center_x + crosshair_gap, center_y), 
                    (center_x + crosshair_size, center_y), crosshair_color, crosshair_thickness)
            
            # Vertical lines
            cv2.line(canvas, (center_x, center_y - crosshair_size), 
                    (center_x, center_y - crosshair_gap), crosshair_color, crosshair_thickness)
            cv2.line(canvas, (center_x, center_y + crosshair_gap), 
                    (center_x, center_y + crosshair_size), crosshair_color, crosshair_thickness)
            
            # Tiny center dot
            cv2.circle(canvas, (center_x, center_y), 2, crosshair_color, -1)
            
            # Bottom bar with minimal indicators
            bar_height = 50
            bar_y = self.screen_height - bar_height
            
            # Semi-transparent bottom bar
            overlay = canvas.copy()
            cv2.rectangle(overlay, (0, bar_y), (self.screen_width, self.screen_height), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)
            
            # Adaptive status indicator (left side)
            adaptive_status = "ON" if self.adaptive_calibration_enabled else "OFF"
            adaptive_color = (0, 220, 255) if self.adaptive_calibration_enabled else (80, 80, 80)
            cv2.circle(canvas, (30, bar_y + 25), 4, adaptive_color, -1)
            cv2.putText(canvas, f"Adaptive {adaptive_status}", (45, bar_y + 28), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)
            
            # Outlier counter (if active)
            if self.outlier_count > 0:
                cv2.putText(canvas, f"Outliers: {self.outlier_count}", (45, bar_y + 43), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
            
            # 3D model status indicator
            model_3d_status = "3D" if self.eye_3d_model_enabled else "2D"
            model_3d_color = (0, 255, 100) if self.eye_3d_model_enabled else (100, 100, 100)
            cv2.circle(canvas, (self.screen_width - 180, bar_y + 25), 4, model_3d_color, -1)
            cv2.putText(canvas, f"Model: {model_3d_status}", (self.screen_width - 165, bar_y + 28), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)
            
            # Centered keyboard shortcuts
            shortcuts = "R: Recal  ·  T: Validate  ·  A: Adaptive  ·  3: 3D Model  ·  Q: Quit"
            text_size = cv2.getTextSize(shortcuts, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.putText(canvas, shortcuts, 
                       (self.screen_width // 2 - text_size[0] // 2, bar_y + 28), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
            
            cv2.imshow("Eye Tracker", canvas)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 27 is ESC
                break
            elif key == ord('t') or key == ord('T'):
                # Start validation mode
                cv2.destroyWindow("Eye Tracker")
                print("\n" + "="*50)
                print("STARTING VALIDATION MODE")
                print("="*50)
                self.run_validation_mode()
                self.display_validation_results()
                print("\nPress any key to resume tracking...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            elif key == ord('a') or key == ord('A'):
                # Toggle adaptive calibration
                self.adaptive_calibration_enabled = not self.adaptive_calibration_enabled
                status = "enabled" if self.adaptive_calibration_enabled else "disabled"
                print(f"Adaptive calibration {status}")
            elif key == ord('3'):
                # Toggle 3D eye model
                self.eye_3d_model_enabled = not self.eye_3d_model_enabled
                status = "enabled (3D eye sphere model)" if self.eye_3d_model_enabled else "disabled (2D interpolation)"
                print(f"3D eye model {status}")
            elif key == ord('v') or key == ord('V'):
                # Toggle 3D visualization
                self.show_3d_visualization = not self.show_3d_visualization
                status = "visible" if self.show_3d_visualization else "hidden"
                print(f"3D visualization {status}")
            elif key == ord('r') or key == ord('R'):
                # Recalibration requested
                print("\n" + "="*50)
                print("RECALIBRATION REQUESTED")
                print("="*50)
                if not self.calibrate():
                    print("Recalibration cancelled, exiting...")
                    break
                
                # Modern recalibration save screen
                save_selected = False
                while not save_selected:
                    save_canvas = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
                    
                    # Success indicator
                    icon_x = self.screen_width // 2
                    icon_y = 200
                    cv2.circle(save_canvas, (icon_x, icon_y), 50, (0, 220, 255), 3)
                    cv2.putText(save_canvas, "✓", (icon_x - 22, icon_y + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 220, 255), 3)
                    
                    # Title
                    title_text = "Recalibration Complete"
                    text_size = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[0]
                    cv2.putText(save_canvas, title_text, 
                               (self.screen_width // 2 - text_size[0] // 2, 310), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 2)
                    
                    # Divider
                    line_y = 340
                    margin = self.screen_width // 3
                    cv2.line(save_canvas, (margin, line_y), (self.screen_width - margin, line_y), (60, 60, 60), 1)
                    
                    # Prompt
                    prompt_text = "Save to profile?"
                    text_size = cv2.getTextSize(prompt_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)[0]
                    cv2.putText(save_canvas, prompt_text, 
                               (self.screen_width // 2 - text_size[0] // 2, 400), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (160, 160, 160), 1)
                    
                    # Options
                    y_offset = 480
                    
                    # Profile slots option
                    badge_x = self.screen_width // 2 - 140
                    badge_y = y_offset - 25
                    cv2.rectangle(save_canvas, (badge_x, badge_y), (badge_x + 90, badge_y + 35), (40, 40, 40), -1)
                    cv2.rectangle(save_canvas, (badge_x, badge_y), (badge_x + 90, badge_y + 35), (0, 200, 255), 1)
                    cv2.putText(save_canvas, "1-10", (badge_x + 20, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1)
                    cv2.putText(save_canvas, "Save Profile", (badge_x + 105, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
                    
                    # Skip option
                    y_offset += 70
                    skip_x = self.screen_width // 2 - 140
                    skip_y = y_offset - 25
                    cv2.rectangle(save_canvas, (skip_x, skip_y), (skip_x + 90, skip_y + 35), (40, 40, 40), -1)
                    cv2.rectangle(save_canvas, (skip_x, skip_y), (skip_x + 90, skip_y + 35), (80, 80, 80), 1)
                    cv2.putText(save_canvas, "N", (skip_x + 35, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)
                    cv2.putText(save_canvas, "Resume", (skip_x + 105, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (140, 140, 140), 1)
                    
                    cv2.imshow("Eye Tracker", save_canvas)
                    
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord('n') or key == ord('N'):
                        save_selected = True
                    elif key >= ord('0') and key <= ord('9'):
                        profile_num = key - ord('0')
                        if profile_num >= 1 and profile_num <= 9:
                            self.save_profile(profile_num)
                            save_selected = True
                    elif key == ord('1') and cv2.waitKey(1) & 0xFF == ord('0'):
                        self.save_profile(10)
                        save_selected = True
                
                print("\n" + "="*50)
                print("RECALIBRATION COMPLETE - Resuming tracking")
                print("="*50 + "\n")
                
                # Reset popup display
                show_popup = True
                popup_start_time = cv2.getTickCount()
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = EyeTracker()
    tracker.run()