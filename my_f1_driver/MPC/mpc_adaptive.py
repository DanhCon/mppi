#!/usr/bin/env python3
"""
Adaptive Planning MPC Node with Spline Rollouts and CVXPY for F1Tenth
===================================================================
Tích hợp thuật toán tránh vật cản bằng Spline Rollouts cục bộ và bộ giải MPC phi tuyến rời rạc.
"""

import rclpy
from rclpy.node import Node
import math
import csv
import numpy as np
from copy import deepcopy

from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from tf2_ros import Buffer, TransformListener, TransformException
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

# Thử import cvxpy
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False


# ────────────────────────────────────────────────────────────────────────────
# TIỆN ÍCH TOÁN HỌC
# ────────────────────────────────────────────────────────────────────────────

def euler_from_quaternion(x, y, z, w) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class State:
    """Trạng thái động học của xe."""
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v


# ────────────────────────────────────────────────────────────────────────────
# NODE MPC ADAPTIVE
# ────────────────────────────────────────────────────────────────────────────

class MPCAdaptiveNode(Node):

    def __init__(self):
        super().__init__("mpc_adaptive_node")

        # ── Tham số xe và thuật toán ──────────────────────────────────────
        self.NX = 4             # x = [x, y, v, yaw]
        self.NU = 2             # u = [accel, steer]
        self.T = 3              # MPC Horizon length
        self.Ts = 0.1           # Chu kỳ trích mẫu MPC (10Hz cho CVXPY chạy ổn định)
        self.DT = self.Ts

        # Kích thước xe và góc bẻ lái tối đa
        self.WB = 0.32          # Chiều dài cơ sở (m)
        self.MAX_STEER = np.deg2rad(20.0)
        self.MAX_DSTEER = np.deg2rad(5.0)  # Tốc độ bẻ lái tối đa (rad/s)
        self.MAX_SPEED = 6.0    # Tốc độ tối đa (m/s)
        self.MIN_SPEED = 0.0
        self.MAX_ACCEL = 2.5    # Gia tốc tối đa (m/s^2)

        # Trọng số MPC
        self.R = np.diag([0.01, 0.01])          # Phạt biên độ đầu vào [accel, steer]
        self.Rd = np.diag([0.01, 1.0])          # Phạt tốc độ thay đổi đầu vào
        self.Q = np.diag([1.0, 1.0, 0.5, 0.5])  # Phạt sai lệch trạng thái [x, y, v, yaw]
        self.Qf = self.Q                        # Trọng số cuối chặng (Terminal cost)

        # ── Tham số LiDAR & Quy hoạch thích ứng ──────────────────────────
        self.angle_min = -2.35
        self.angle_max = 2.35
        self.angle_increment = 0.0043
        self.ranges = []
        self.islidaron = False
        self.safety_dist = 2.5  # Khoảng cách phát hiện chướng ngại vật phía trước (m)

        # Trạng thái hiện tại của xe và biến điều khiển
        self.v_x_current = 0.0
        self.oa = None
        self.odelta = None
        self.start_index = None
        self.waypoints = []

        # ── Khởi tạo các quỹ đạo Spline Rollouts ─────────────────────────
        self.generate_splines()

        # ── Cấu hình ROS 2 ───────────────────────────────────────────────
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.car_frame = "ego_racecar/base_link"
        self.map_frame = "map"

        self.sub_odom = self.create_subscription(
            Odometry, "ego_racecar/odom", self.odom_callback, 10)
        self.sub_scan = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 10)
        self.pub_drive = self.create_publisher(
            AckermannDriveStamped, "/drive", 10)

        # Visualizations
        self.pub_global_path = self.create_publisher(Marker, "/global_path_viz", 10)
        self.pub_local_splines = self.create_publisher(Marker, "/local_splines_viz", 10)
        self.pub_mpc_horizon = self.create_publisher(Marker, "/mpc_horizon_viz", 10)

        # Đường dẫn csv Waypoints mặc định
        csv_path = (
            "/sim_ws/install/waypoint/share/waypoint/f1tenth_waypoint_generator/racelines/f1tenth_waypoint.csv"
        )
        self.load_waypoints(csv_path)

        self.last_mpc_time = self.get_clock().now()
        self.get_logger().info(
            f"Adaptive MPC Node (Spline Rollouts) đã khởi tạo thành công! CVXPY_AVAILABLE={CVXPY_AVAILABLE}"
        )

    # ════════════════════════════════════════════════════════════════════════
    # TỰ SINH SPLINES ROLLOUT TRỰC TUYẾN
    # ════════════════════════════════════════════════════════════════════════

    def generate_splines(self):
        """
        Tạo trước thư viện 30 spline cục bộ bằng mô phỏng động học xe đạp.
        Mỗi spline có 51 điểm (thời gian mô phỏng 0.02s mỗi bước).
        Cấu trúc điểm: [x, y, yaw, yaw, r, theta]
        """
        self.local_path = []
        T_spline = 50
        dt_spline = 0.02

        # Cấu hình Profile vận tốc và góc lái tương ứng
        profiles = [
            (1.0, np.linspace(np.deg2rad(-30.0), np.deg2rad(30.0), 13)),
            (1.5, np.linspace(np.deg2rad(-15.0), np.deg2rad(15.0), 7)),
            (2.0, np.linspace(np.deg2rad(-15.0), np.deg2rad(15.0), 7))
        ]

        for v, steer_angles in profiles:
            # Thêm chạy thẳng (steer = 0)
            steer_angles = np.append(steer_angles, 0.0)
            for delta in steer_angles:
                spline = []
                x_s, y_s, yaw_s = 0.0, 0.0, 0.0
                for t in range(T_spline + 1):
                    r = math.hypot(x_s, y_s)
                    theta = math.atan2(y_s, x_s)
                    spline.append([x_s, y_s, yaw_s, yaw_s, r, theta])

                    # Cập nhật trạng thái
                    x_s += v * math.cos(yaw_s) * dt_spline
                    y_s += v * math.sin(yaw_s) * dt_spline
                    yaw_s += (v / self.WB) * math.tan(delta) * dt_spline
                self.local_path.append(spline)

        self.local_path = np.array(self.local_path)
        self.get_logger().info(f"Đã tự động tạo {self.local_path.shape[0]} spline cục bộ.")

    # ════════════════════════════════════════════════════════════════════════
    # CALLBACK SCAN LIDAR
    # ════════════════════════════════════════════════════════════════════════

    def scan_callback(self, msg: LaserScan):
        self.angle_increment = msg.angle_increment
        self.angle_max = msg.angle_max
        self.angle_min = msg.angle_min
        self.ranges = np.array(msg.ranges, dtype=np.float32)
        self.islidaron = True

        # Thay NaN và Inf bằng max_range
        max_r = msg.range_max if msg.range_max > 0.0 else 20.0
        self.ranges = np.where(np.isfinite(self.ranges) & (self.ranges > msg.range_min), self.ranges, max_r)

        # Thuật toán Disparity Extender đơn giản
        disparity = self.ranges[:-1] - self.ranges[1:]
        disparity_bool = np.abs(disparity) >= 1.0
        disparity_bool_idx = np.where(disparity_bool)[0]

        # Bơm rộng bán kính an toàn (Safety Bubble) quanh điểm thay đổi đột ngột
        bubble = 25
        for idx in disparity_bool_idx:
            min_idx = max(0, idx - bubble)
            max_idx = min(idx + bubble, self.ranges.shape[0])
            self.ranges[min_idx:max_idx] = np.min(self.ranges[min_idx:max_idx])

    # ════════════════════════════════════════════════════════════════════════
    # CALLBACK ODOMETRY - VÒNG LẶP CHÍNH
    # ════════════════════════════════════════════════════════════════════════

    def odom_callback(self, msg: Odometry):
        if not self.waypoints or not self.islidaron:
            return

        now = self.get_clock().now()
        if (now - self.last_mpc_time).nanoseconds / 1e9 < self.Ts:
            return
        self.last_mpc_time = now

        # Lấy tốc độ hiện tại
        self.v_x_current = msg.twist.twist.linear.x

        # ── Lấy vị trí từ TF ─────────────────────────────────────────────
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.car_frame, rclpy.time.Time(seconds=0))
            rx = tf.transform.translation.x
            ry = tf.transform.translation.y
            r_yaw = euler_from_quaternion(
                tf.transform.rotation.x, tf.transform.rotation.y,
                tf.transform.rotation.z, tf.transform.rotation.w)
        except TransformException:
            return

        # Đảm bảo góc lái nằm trong khoảng dương [0, 2pi] để đồng bộ hoá
        if r_yaw < 0:
            r_yaw += 2 * math.pi

        # Tìm điểm gần nhất trên lộ trình cơ sở
        nearest_idx = self._find_nearest_waypoint(rx, ry)

        # ── BỘ LỌC VÀ TRÁNH VẬT CẢN (ADAPTIVE PLANNING) ────────────────
        # Kiểm tra xem có chướng ngại vật ở góc chính diện phía trước hay không
        center_idx = int((0.0 - self.angle_min) / self.angle_increment)
        fov_side = 50  # tương đương khoảng +- 13 độ chính diện
        min_front_dist = np.min(self.ranges[center_idx - fov_side : center_idx + fov_side])

        obstacle_detected = min_front_dist < self.safety_dist
        best_spline_idx = 0
        best_progress_idx = 0
        safe_splines_count = 0

        # Chu bị các điểm trên lộ trình toàn cục để tìm điểm giao lại
        num_search = 35
        global_search_pts = []
        for i in range(num_search):
            pt = self.waypoints[(nearest_idx + i) % len(self.waypoints)]
            global_search_pts.append([pt[0], pt[1]])
        global_search_pts = np.array(global_search_pts)

        # Visualizations
        self.publish_global_path()
        local_markers = self.init_marker_list()

        cos_y = math.cos(r_yaw)
        sin_y = math.sin(r_yaw)

        # Nếu phát hiện vật cản, đánh giá các spline cục bộ
        if obstacle_detected:
            for i in range(self.local_path.shape[0]):
                spline = self.local_path[i]  # shape: (51, 6)
                is_collision = False

                # Kiểm tra va chạm từng điểm trên spline
                for pt in spline:
                    r_dist, theta_r = pt[4], pt[5]
                    # Tìm góc quét tương ứng trong mảng LiDAR
                    scan_idx = int((theta_r - self.angle_min) / self.angle_increment)

                    if 0 <= scan_idx < len(self.ranges):
                        # Lấy khoảng cách nhỏ nhất xung quanh hướng đó để tăng độ an toàn
                        min_laser = np.min(self.ranges[max(0, scan_idx - 3):min(len(self.ranges), scan_idx + 4)])
                        if min_laser < r_dist:
                            is_collision = True
                            break

                # Tính tọa độ thế giới của điểm cuối spline
                last_pt = spline[-1]
                world_last_x = rx + last_pt[0] * cos_y - last_pt[1] * sin_y
                world_last_y = ry + last_pt[0] * sin_y + last_pt[1] * cos_y

                # Lưu marker để vẽ lên RViz
                color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.5) if is_collision else ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.5)
                self.add_spline_to_marker(local_markers, spline, rx, ry, cos_y, sin_y, color)

                if not is_collision:
                    safe_splines_count += 1
                    # Tìm điểm gần nhất trên quỹ đạo toàn cục
                    dists = np.linalg.norm(global_search_pts - np.array([world_last_x, world_last_y]), axis=1)
                    progress_idx = np.argmin(dists)
                    if progress_idx >= best_progress_idx:
                        best_progress_idx = progress_idx
                        best_spline_idx = i

            # Nếu có spline an toàn, chuyển xref sang đi theo spline
            if safe_splines_count > 0:
                selected_spline = self.local_path[best_spline_idx]
                
                # Tô đậm spline được chọn
                gold_color = ColorRGBA(r=1.0, g=0.84, b=0.0, a=1.0)
                self.add_spline_to_marker(local_markers, selected_spline, rx, ry, cos_y, sin_y, gold_color)
                
                # Lấy mẫu các điểm mốc tham chiếu từ Spline
                xref = np.zeros((self.NX, self.T + 1))
                sampled_indices = np.linspace(0, selected_spline.shape[0] - 1, self.T + 1, dtype=int)
                
                for idx_t, step_idx in enumerate(sampled_indices):
                    pt = selected_spline[step_idx]
                    wx = rx + pt[0] * cos_y - pt[1] * sin_y
                    wy = ry + pt[0] * sin_y + pt[1] * cos_y
                    w_yaw = normalize_angle(pt[2] + r_yaw)
                    
                    xref[0, idx_t] = wx
                    xref[1, idx_t] = wy
                    xref[2, idx_t] = pt[2] * 0.0 + max(self.v_x_current, 1.0) # Vận tốc mục tiêu bằng vận tốc hiện tại
                    xref[3, idx_t] = w_yaw
            else:
                # Không tìm được spline an toàn -> Dùng phanh khẩn cấp
                self.get_logger().error("Không tìm thấy quỹ đạo tránh va chạm nào! Dừng xe khẩn cấp.")
                self._publish_drive(0.0, 0.0)
                self.pub_local_splines.publish(local_markers)
                return
        else:
            # Không có vật cản -> Bám theo quỹ đạo tối ưu cơ bản
            xref, _ = self.calc_ref_trajectory(rx, ry, r_yaw, nearest_idx)

        self.pub_local_splines.publish(local_markers)
        self.publish_mpc_horizon(xref)

        # ── GIẢI BÀI TOÁN LTV-MPC ───────────────────────────────────────
        if not CVXPY_AVAILABLE:
            # Fallback đơn giản nếu chưa có CVXPY
            self.get_logger().warn("CVXPY chưa được cài đặt. Lái xe bằng Pure Pursuit tạm thời.")
            steering = float(normalize_angle(xref[3, 0] - r_yaw))
            self._publish_drive(steering, 1.5)
            return

        x0 = [rx, ry, max(self.v_x_current, 0.1), r_yaw]
        dref = np.zeros((1, self.T + 1))

        # Điều khiển lặp tuyến tính
        self.oa, self.odelta, ox, oy, oyaw, ov = self.iterative_linear_mpc_control(xref, x0, dref, self.oa, self.odelta)

        if self.odelta is not None:
            steering_cmd = float(self.odelta[0])
            accel_cmd = float(self.oa[0])
            
            # Tính tốc độ mục tiêu từ mpc
            speed_cmd = float(self.v_x_current + accel_cmd * self.DT)
            speed_cmd = float(np.clip(speed_cmd, self.MIN_SPEED, self.MAX_SPEED))
            
            self._publish_drive(steering_cmd, speed_cmd)
        else:
            # Fallback nếu giải lỗi
            self.get_logger().warn("MPC giải lỗi. Dừng hoặc giảm ga.")
            self._publish_drive(0.0, 0.5)

    # ════════════════════════════════════════════════════════════════════════
    # LTV-MPC SOLVER VỚI CVXPY
    # ════════════════════════════════════════════════════════════════════════

    def iterative_linear_mpc_control(self, xref, x0, dref, oa, od):
        """Giải lặp LPV MPC để hội tụ điểm vận hành tốt nhất."""
        if oa is None or od is None:
            oa = [0.0] * self.T
            od = [0.0] * self.T

        MAX_ITER = 2
        DU_TH = 0.1

        for i in range(MAX_ITER):
            xbar = self.predict_motion(x0, oa, od, xref)
            poa, pod = oa[:], od[:]
            oa, od, ox, oy, oyaw, ov = self.linear_mpc_control(xref, xbar, x0, dref)
            
            if oa is None:
                return poa, pod, None, None, None, None
                
            du = sum(abs(oa - poa)) + sum(abs(od - pod))
            if du <= DU_TH:
                break
        return oa, od, ox, oy, oyaw, ov

    def predict_motion(self, x0, oa, od, xref):
        """Mô phỏng trước chuyển động để tìm điểm vận hành tuyến tính hóa."""
        xbar = xref * 0.0
        for i in range(self.NX):
            xbar[i, 0] = x0[i]
        
        state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
        for ai, di, i in zip(oa, od, range(1, self.T + 1)):
            state = self.update_state(state, ai, di)
            xbar[0, i] = state.x
            xbar[1, i] = state.y
            xbar[2, i] = state.v
            xbar[3, i] = state.yaw
        return xbar

    def update_state(self, state, a, delta):
        delta = np.clip(delta, -self.MAX_STEER, self.MAX_STEER)
        state.x += state.v * math.cos(state.yaw) * self.DT
        state.y += state.v * math.sin(state.yaw) * self.DT
        state.yaw += (state.v / self.WB) * math.tan(delta) * self.DT
        state.v += a * self.DT
        state.v = np.clip(state.v, self.MIN_SPEED, self.MAX_SPEED)
        return state

    def get_linear_model_matrix(self, v, phi, delta):
        """Tuyến tính hóa Jacobian động học xe đạp quanh điểm vận hành."""
        A = np.zeros((self.NX, self.NX))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = self.DT * math.cos(phi)
        A[0, 3] = - self.DT * v * math.sin(phi)
        A[1, 2] = self.DT * math.sin(phi)
        A[1, 3] = self.DT * v * math.cos(phi)
        A[3, 2] = self.DT * math.tan(delta) / self.WB

        B = np.zeros((self.NX, self.NU))
        B[2, 0] = self.DT
        B[3, 1] = self.DT * v / (self.WB * math.cos(delta) ** 2)

        C = np.zeros(self.NX)
        C[0] = self.DT * v * math.sin(phi) * phi
        C[1] = - self.DT * v * math.cos(phi) * phi
        C[3] = - self.DT * v * delta / (self.WB * math.cos(delta) ** 2)
        return A, B, C

    def linear_mpc_control(self, xref, xbar, x0, dref):
        """Giải tối ưu hóa lồi QP bằng CVXPY + ECOS."""
        x = cp.Variable((self.NX, self.T + 1))
        u = cp.Variable((self.NU, self.T))
        cost = 0.0
        constraints = []

        # Ràng buộc trạng thái ban đầu
        constraints.append(x[:, 0] == x0)

        for t in range(self.T):
            cost += cp.quad_form(u[:, t], self.R)
            if t != 0:
                cost += cp.quad_form(xref[:, t] - x[:, t], self.Q)
            
            # Tuyến tính hóa động học
            A, B, C = self.get_linear_model_matrix(xbar[2, t], xbar[3, t], dref[0, t])
            constraints.append(x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C)

            # Ràng buộc tốc độ bẻ lái
            if t < (self.T - 1):
                cost += cp.quad_form(u[:, t + 1] - u[:, t], self.Rd)
                constraints.append(cp.abs(u[1, t + 1] - u[1, t]) <= self.MAX_DSTEER * self.DT)

        cost += cp.quad_form(xref[:, self.T] - x[:, self.T], self.Qf)

        # Ràng buộc chặn cứng trạng thái và hành vi lái
        constraints.append(x[2, :] <= self.MAX_SPEED)
        constraints.append(x[2, :] >= self.MIN_SPEED)
        constraints.append(cp.abs(u[0, :]) <= self.MAX_ACCEL)
        constraints.append(cp.abs(u[1, :]) <= self.MAX_STEER)

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.ECOS, verbose=False)

        if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
            ox = np.array(x.value[0, :]).flatten()
            oy = np.array(x.value[1, :]).flatten()
            ov = np.array(x.value[2, :]).flatten()
            oyaw = np.array(x.value[3, :]).flatten()
            oa = np.array(u.value[0, :]).flatten()
            odelta = np.array(u.value[1, :]).flatten()
            return oa, odelta, ox, oy, oyaw, ov
        else:
            return None, None, None, None, None, None

    # ════════════════════════════════════════════════════════════════════════
    # TÌM ĐƯỜNG THAM CHIẾU TOÀN CỤC
    # ════════════════════════════════════════════════════════════════════════

    def calc_ref_trajectory(self, rx, ry, r_yaw, nearest_idx):
        """Tính toán quỹ đạo tham chiếu toàn cục cơ sở cho MPC."""
        xref = np.zeros((self.NX, self.T + 1))
        step_dist = max(self.v_x_current, 1.0) * self.DT
        curr_idx = nearest_idx
        dist_accum = 0.0

        xref[0, 0] = self.waypoints[nearest_idx][0]
        xref[1, 0] = self.waypoints[nearest_idx][1]
        xref[2, 0] = self.waypoints[nearest_idx][3]  # speed
        xref[3, 0] = self.waypoints[nearest_idx][2]  # yaw

        # Đảm bảo góc không bị quấn lệch lớn
        if abs(r_yaw - xref[3, 0]) > math.pi:
            if r_yaw < xref[3, 0]:
                r_yaw += 2 * math.pi
            else:
                xref[3, 0] += 2 * math.pi

        for i in range(1, self.T + 1):
            target_dist = i * step_dist
            while True:
                nxt = (curr_idx + 1) % len(self.waypoints)
                p1, p2 = self.waypoints[curr_idx], self.waypoints[nxt]
                seg = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                if dist_accum + seg >= target_dist:
                    ratio = (target_dist - dist_accum) / seg if seg > 0 else 0.0
                    fx = p1[0] + ratio * (p2[0] - p1[0])
                    fy = p1[1] + ratio * (p2[1] - p1[1])
                    fyaw = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
                    fspd = p1[3] + ratio * (p2[3] - p1[3])
                    break
                dist_accum += seg
                curr_idx = nxt

            xref[0, i] = fx
            xref[1, i] = fy
            xref[2, i] = float(np.clip(fspd, self.MIN_SPEED, self.MAX_SPEED))
            xref[3, i] = fyaw

            # Xử lý quấn góc góc hướng cho trơn tru
            if xref[3, i] < 1.0 and xref[3, i - 1] > 6.0:
                xref[3, i] += 2 * math.pi
            if (xref[3, i - 1] - xref[3, i]) < -math.pi:
                xref[3, i - 1] += 2 * math.pi

        return xref, nearest_idx

    # ════════════════════════════════════════════════════════════════════════
    # ĐỌC CSV WAYPOINTS
    # ════════════════════════════════════════════════════════════════════════

    def load_waypoints(self, filename: str):
        try:
            with open(filename, 'r') as f:
                for row in csv.reader(f):
                    if not row:
                        continue
                    cols = row[0].split() if len(row) == 1 else row
                    try:
                        # CSV format: x, y, yaw, velocity, ...
                        self.waypoints.append([
                            float(cols[0]),
                            float(cols[1]),
                            float(cols[2]),
                            float(cols[3])
                        ])
                    except:
                        continue
            self.get_logger().info(f"Đã nạp {len(self.waypoints)} điểm từ file CSV.")
        except Exception as e:
            self.get_logger().error(f"Lỗi khi đọc file CSV Waypoints: {e}")

    def _find_nearest_waypoint(self, rx: float, ry: float) -> int:
        if self.start_index is None:
            dists = [math.hypot(rx - p[0], ry - p[1]) for p in self.waypoints]
            self.start_index = int(np.argmin(dists))
            return self.start_index
        idx = self.start_index
        cur_d = math.hypot(rx - self.waypoints[idx][0], ry - self.waypoints[idx][1])
        for _ in range(25):
            nxt = (idx + 1) % len(self.waypoints)
            nxt_d = math.hypot(rx - self.waypoints[nxt][0], ry - self.waypoints[nxt][1])
            if nxt_d < cur_d:
                idx, cur_d = nxt, nxt_d
            else:
                break
        self.start_index = idx
        return idx

    def _publish_drive(self, steering: float, speed: float):
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = float(steering)
        msg.drive.speed = float(speed)
        self.pub_drive.publish(msg)

    # ════════════════════════════════════════════════════════════════════════
    # RVIZ VISUALIZATION MARKERS
    # ════════════════════════════════════════════════════════════════════════

    def publish_global_path(self):
        m = Marker()
        m.header.frame_id = self.map_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "global_path"
        m.id = 1
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.08
        m.color = ColorRGBA(r=0.0, g=0.5, b=1.0, a=0.7)
        for wp in self.waypoints:
            m.points.append(Point(x=wp[0], y=wp[1], z=0.0))
        self.pub_global_path.publish(m)

    def init_marker_list(self) -> Marker:
        m = Marker()
        m.header.frame_id = self.map_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "local_splines"
        m.id = 2
        m.type = Marker.LINE_LIST
        m.action = Marker.ADD
        m.scale.x = 0.03
        return m

    def add_spline_to_marker(self, marker: Marker, spline, rx, ry, cos_y, sin_y, color):
        for i in range(spline.shape[0] - 1):
            pt1 = spline[i]
            pt2 = spline[i + 1]

            w1x = rx + pt1[0] * cos_y - pt1[1] * sin_y
            w1y = ry + pt1[0] * sin_y + pt1[1] * cos_y

            w2x = rx + pt2[0] * cos_y - pt2[1] * sin_y
            w2y = ry + pt2[0] * sin_y + pt2[1] * cos_y

            marker.points.append(Point(x=w1x, y=w1y, z=0.05))
            marker.points.append(Point(x=w2x, y=w2y, z=0.05))
            marker.colors.append(color)
            marker.colors.append(color)

    def publish_mpc_horizon(self, xref):
        m = Marker()
        m.header.frame_id = self.map_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "mpc_horizon"
        m.id = 3
        m.type = Marker.SPHERE_LIST
        m.action = Marker.ADD
        m.scale.x = m.scale.y = m.scale.z = 0.18
        m.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0)
        for t in range(self.T + 1):
            m.points.append(Point(x=xref[0, t], y=xref[1, t], z=0.1))
        self.pub_mpc_horizon.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = MPCAdaptiveNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Gửi lệnh dừng xe trước khi tắt
        stop_msg = AckermannDriveStamped()
        stop_msg.drive.speed = 0.0
        node.pub_drive.publish(stop_msg)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
