#!/usr/bin/env python3

import cv2
import numpy as np
import rospy

from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image

from pamaral_object_grasping_pattern_recognition.msg import MediaPipeResults, PointList


class KeypointsPreprocessing:

    def __init__(self):
        self.bridge = CvBridge()
        self.mp_points_image_publisher = rospy.Publisher("mp_points_image", Image, queue_size=1)
        self.preprocessed_points_publisher = rospy.Publisher("preprocessed_points", PointList, queue_size=1)
        self.mediapipe_results_sub = rospy.Subscriber("mediapipe_results", MediaPipeResults, self.mediapipe_results_callback)

    def mediapipe_results_callback(self, msg):
        hands_keypoints = msg.hands_keypoints
        pose_keypoints = msg.pose_keypoints

        points = None
        if len(hands_keypoints) > 0 and len(pose_keypoints) > 0:
            pose_keypoints = [[p.x, p.y, p.z] for p in pose_keypoints]
            hands_keypoints = [[p.x, p.y, p.z] for p in hands_keypoints]

            # obtain right and left hands centroids and valid radius
            left_hand = np.array([pose_keypoints[15], pose_keypoints[17], pose_keypoints[19], pose_keypoints[21]])
            right_hand = np.array([pose_keypoints[16], pose_keypoints[18], pose_keypoints[20], pose_keypoints[22]])

            left_hand_centroid = np.average(left_hand, axis=0)
            right_hand_centroid = np.average(right_hand, axis=0)

            left_hand_radius = 2*np.max(np.linalg.norm(left_hand_centroid - left_hand, axis=1))
            right_hand_radius = 2*np.max(np.linalg.norm(right_hand_centroid - right_hand, axis=1))

            # separate hands keypoints
            hands_keypoints = [hands_keypoints[i*21:(i+1)*21] for i in range(len(hands_keypoints)//21)]

            # check which hand is closer to the centroid of the right hand and if it is within the valid radius
            best_distance = 100000
            for hand in hands_keypoints:
                hand = np.array(hand)
                hand_centroid = np.average(hand, axis=0)

                right_hand_distance = np.linalg.norm(hand_centroid[:2] - right_hand_centroid[:2])
                left_hand_distance = np.linalg.norm(hand_centroid[:2] - left_hand_centroid[:2])

                if right_hand_distance < left_hand_distance:# and right_hand_distance < right_hand_radius:
                    if right_hand_distance < best_distance:
                        points = hand
                        best_distance = right_hand_distance
        
        if points is not None:
            centroid = np.average(points, axis=0)

            points[:,0] -= centroid[0]
            points[:,1] -= centroid[1]
            points[:,2] -= centroid[2]

            max_value = np.max(np.absolute(points))

            points = (points/max_value + 1)/2

            mp_points_image = np.zeros((480, 640, 3), dtype=np.uint8)

            # Draw the points on the image
            point_radius = 8
            point_color = 255  # White

            lines = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
                     [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15],
                     [15, 16], [0, 17], [17, 18], [18, 19], [19, 20], [13, 17]]
            
            # Define the polygon vertices
            polygon_points = np.array([points[0][:2], points[5][:2], points[9][:2], points[13][:2], points[17][:2]])

            polygon_points[:,0] *= 640
            polygon_points[:,1] *= 480

            polygon_points = polygon_points.astype(np.int32)

            # Reshape the points array into shape compatible with fillPoly
            polygon_points = polygon_points.reshape((-1, 1, 2))

            # Specify the color for the polygon (in BGR format)
            polygon_color = (0, 64, 0) # Dark green

            # Fill the polygon with the specified color
            cv2.fillPoly(mp_points_image, [polygon_points], polygon_color)
            
            for p1, p2 in lines:
                cv2.line(mp_points_image, [int(points[p1][0]*640), int(points[p1][1]*480)], [int(points[p2][0]*640), int(points[p2][1]*480)], [0, 255, 0], 2)

            for point in points:
                cv2.circle(mp_points_image, [int(point[0]*640), int(point[1]*480)], point_radius, [0,0,255], -1)
            
            self.mp_points_image_publisher.publish(self.bridge.cv2_to_imgmsg(mp_points_image, "bgr8"))
        
        else:
            points = []

        msg = PointList()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "mp_points"
        msg.points = [Point(p[0], p[1], p[2]) for p in points]
        self.preprocessed_points_publisher.publish(msg)


def main():
    default_node_name = 'keypoints_preprocessing'
    rospy.init_node(default_node_name, anonymous=False)

    KeypointsPreprocessing()

    rospy.spin()


if __name__ == '__main__':
    main()
