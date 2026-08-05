#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
import geometry_msgs.msg

class BottleCapDetector:
    def __init__(self):
        rospy.init_node('bottle_cap_detector', anonymous=True)
        rospy.loginfo("Bottle Cap Detector started")

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback)
        self.image_pub = rospy.Publisher('/bottle_cap_detector/output_image', Image, queue_size=1)
        self.br = tf2_ros.TransformBroadcaster()
        self.camera_matrix = np.array([[615, 0, 320],
                                       [0, 615, 240],
                                       [0, 0, 1]], dtype=np.float64)
        self.dist_coeffs = np.array([0.1, -0.25, 0, 0, 0], dtype=np.float64)
        self.real_diameter = 0.03  #meters

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"CvBridge error: {e}")
            return

        h, w = cv_image.shape[:2]
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
        )
        undistorted = cv2.undistort(
            cv_image, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix
        )

        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
            param1=120, param2=35, minRadius=15, maxRadius=50
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # Draw circle
                cv2.circle(undistorted, (x, y), r, (0, 255, 0), 2)

                
                focal_length_px = self.camera_matrix[0, 0]  
                pixel_diameter = r * 2
                if pixel_diameter > 0:
                    distance = (self.real_diameter * focal_length_px) / pixel_diameter
                else:
                    distance = 0.0

                
                fx = self.camera_matrix[0, 0]
                fy = self.camera_matrix[1, 1]
                cx = self.camera_matrix[0, 2]
                cy = self.camera_matrix[1, 2]
                x_norm = ((x - cx) / fx)/2
                y_norm = (y - cy) / fy

                
                pos_x = x_norm               
                pos_y = distance      
                pos_z = -y_norm      

                
                cv2.putText(
                    undistorted,
                    f"X={pos_x:.2f}m Y={pos_y:.2f}m",
                    (x - 100, y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )

                rospy.loginfo(f"Bottle cap detected: X={pos_x:.3f} Y={pos_y:.3f} m")

                #TF
                t = geometry_msgs.msg.TransformStamped()
                t.header.stamp = rospy.Time.now()
                t.header.frame_id = "camera_link"
                t.child_frame_id = "bottle_cap"
                t.transform.translation.x = pos_x
                t.transform.translation.y = pos_y
                t.transform.translation.z = pos_z
                t.transform.rotation.x = 0.0
                t.transform.rotation.y = 0.0
                t.transform.rotation.z = 0.0
                t.transform.rotation.w = 1.0
                self.br.sendTransform(t)
                break
        else:
            rospy.loginfo("No bottle cap detected")

        cv2.imshow("Bottle Cap Detector EIEI", undistorted)
        cv2.waitKey(1)
        output_msg = self.bridge.cv2_to_imgmsg(undistorted, encoding="bgr8")
        self.image_pub.publish(output_msg)

if __name__ == '__main__':
    try:
        detector = BottleCapDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
