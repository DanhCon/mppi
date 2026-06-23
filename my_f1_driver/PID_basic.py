#!/usr/bin/env python3 
import rclpy
from rclpy.node import Node
import math
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class WallFollowingNode(Node):
    def __init__(self):
        super().__init__('wall_following_node')
        
        # 1. Khai báo các Publisher và Subscriber
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        
        # 2. Khởi tạo các tham số bộ điều khiển (Hệ số thực nghiệm từ ĐH Penn)
        self.kp = 1.0
        self.kd = 1.5
        self.prev_error = 0.0

        self.prev_time = None
        
        # 3. Khai báo khoảng cách đích và khoảng cách nhìn trước (Look-ahead)
        self.desired_dist = 1.0  # Mong muốn cách tường phải 1 mét
        self.L = 1.0             # Nhìn trước 1 mét
    def angle_to_index(self, angle_rad, data):
        """
        Góc angle_rad truyền vào theo chuẩn ROS: Thẳng trước mặt là 0, phải là âm, trái là dương
        """
        # Giới hạn góc truyền vào không vượt quá tầm quét của cảm biến
        if angle_rad < data.angle_min or angle_rad > data.angle_max:
            return None
            
        # Công thức ánh xạ tuyến tính từ góc sang chỉ mục mảng
        index = int((angle_rad - data.angle_min) / data.angle_increment)
        return index
    def get_filtered_range(self, index, data, window_size=5):
        """
        Lấy trung bình cộng của một nhóm tia xung quanh chỉ mục chỉ định
        """
        half_window = window_size // 2
        start_idx = max(0, index - half_window)
        end_idx = min(len(data.ranges) - 1, index + half_window + 1)
        
        valid_ranges = []
        for i in range(start_idx, end_idx):
            # Loại bỏ các giá trị lỗi cảm biến
            if not math.isnan(data.ranges[i]) and not math.isinf(data.ranges[i]):
                valid_ranges.append(data.ranges[i])
                
        if len(valid_ranges) == 0:
            return 4.0 # Trả về giá trị an toàn mặc định nếu mất tín hiệu hoàn toàn
            
        return sum(valid_ranges) / len(valid_ranges)
    def lidar_callback(self, data):
        current_time = self.get_clock().now()
        if self.prev_time is None:
            # Chu kỳ đầu tiên, chưa có mốc thời gian cũ để trừ
            self.prev_time = current_time
            return
        delta_t = (current_time - self.prev_time).nanoseconds / 1e9
        self.prev_time = current_time  # Lưu lại mốc thời gian cho chu kỳ tiếp theo
        # Thiết lập góc theta kẹp giữa 2 tia (ví dụ 45 độ)
        theta_deg = 45.0
        theta_rad = math.radians(theta_deg)
        
        # Tính góc của 2 tia theo hệ trục ROS (bám tường PHẢI)
        angle_b = math.radians(-90.0)
        angle_a = math.radians(-90.0 + theta_deg) # = -45 độ
        
        # Tìm chỉ mục mảng
        idx_b = self.angle_to_index(angle_b, data)
        idx_a = self.angle_to_index(angle_a, data)
        
        # Lấy khoảng cách đã qua lọc nhiễu
        b = self.get_filtered_range(idx_b, data)
        a = self.get_filtered_range(idx_a, data)
        
        # --- BẮT ĐẦU TÍNH TOÁN THEO CÔNG THỨC SLIDE ---
        # 1. Tính góc alpha (Heading error)
        alpha = math.atan2(a * math.cos(theta_rad) - b, a * math.sin(theta_rad))
        
        # 2. Tính khoảng cách thực tế tức thời AB
        AB = b * math.cos(alpha)
        
        # 3. Tính khoảng cách dự báo tương lai D_t+1 (Look-ahead)
        D_t_plus_1 = AB + self.L * math.sin(alpha)
        
        # 4. Tính toán sai số hiện tại e(t)
        current_error = self.desired_dist - D_t_plus_1
        # Cấu hình in ra màn hình với tần suất tối đa là 0.5 giây một lần (Thao tác cực kỳ gọn)
        self.get_logger().info(f"Error = {current_error:.4f}", throttle_duration_sec=0.5)

        # 5. Tính thành phần vi phân (Derivative)
        error_derivative = (current_error - self.prev_error)/ delta_t
        self.prev_error = current_error # Lưu lại cho chu kỳ sau
        
        # 6. Tính toán góc lái u(t) qua bộ PD
        steering_angle = (self.kp * current_error) + (self.kd * error_derivative)
        
        # Giới hạn góc lái vật lý của xe (+/- 24 độ)
        steering_angle = max(min(steering_angle, 0.4189), -0.4189)
        
        # --- ĐIỀU CHỈNH TỐC ĐỘ TỰ ĐỘNG THEO GÓC LÁI ---
        # Đường thẳng chạy nhanh, cua gắt tự động giảm tốc để tránh trượt bánh
        steering_deg = math.degrees(abs(steering_angle))
        if steering_deg < 10.0:
            velocity = 7.0   # m/s
        elif steering_deg < 20.0:
            velocity = 4.0   # m/s
        else:
            velocity = 4.0   # m/s
            
        # Gửi lệnh điều khiển xuống xe
        self.publish_drive(steering_angle, velocity)
    def publish_drive(self, steer, speed):
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'base_link'
        
        drive_msg.drive.steering_angle = steer
        drive_msg.drive.speed = speed
        
        self.drive_pub.publish(drive_msg)

# Hàm main để chạy Node
def main(args=None):
    rclpy.init(args=args)
    node = WallFollowingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()