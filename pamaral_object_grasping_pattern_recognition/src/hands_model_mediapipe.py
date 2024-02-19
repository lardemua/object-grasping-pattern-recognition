#!/usr/bin/env python3

# set to empty string to force CPU usage
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import actionlib
import cv2
import mediapipe as mp
import numpy as np
import rospy

from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from sensor_msgs.msg import Image
from std_msgs.msg import String

from pamaral_object_grasping_pattern_recognition.msg import MpHand, MpHandsModelAction, MpHandsModelResult


class HandsModelMediapipe:

    def __init__(self):
        # Initialize Hand Landmarker
        base_options = BaseOptions(model_asset_path=f"{os.environ['HOME']}/catkin_ws/src/object_grasping_pattern_recognition/pamaral_object_grasping_pattern_recognition/models/hand_landmarker.task")
        self.options = HandLandmarkerOptions(base_options=base_options, running_mode=RunningMode.VIDEO, num_hands=2)
        self.hand_landmarker = None # HandLandmarker.create_from_options(self.options)

        self.bridge = CvBridge()

        self.mp_drawing_publisher = rospy.Publisher("mp_drawing", Image, queue_size=1)

        self.last_ts = 0

        self.server = actionlib.SimpleActionServer('hands_model', MpHandsModelAction, self.execute, False)
        self.server.start()

        rospy.loginfo("Hand Model Ready")
    
    def draw_hand_points(self, drawing, hand_landmarks):
        points = [[p.x, p.y, p.z] for p in hand_landmarks]

        # Draw the points on the image
        point_radius = 8

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
        cv2.fillPoly(drawing, [polygon_points], polygon_color)
        
        for p1, p2 in lines:
            cv2.line(drawing, [int(points[p1][0]*640), int(points[p1][1]*480)], [int(points[p2][0]*640), int(points[p2][1]*480)], [0, 255, 0], 2)

        for point in points:
            cv2.circle(drawing, [int(point[0]*640), int(point[1]*480)], point_radius, [0,0,255], -1)
        
        return drawing

    def execute(self, msg):
        try:
            # Convert the ROS Image message to OpenCV image
            image = self.bridge.imgmsg_to_cv2(msg.image, "bgr8")

            # Get the timestamp of the image in miliseconds
            timestamp = msg.image.header.stamp
            timestamp = int(timestamp.secs * 1000 + timestamp.nsecs / 1000000)

        except Exception as e:
            rospy.logerr(e)
            return
        
        if timestamp < self.last_ts or timestamp - self.last_ts > 5000:
            self.hand_landmarker = HandLandmarker.create_from_options(self.options)
        
        self.last_ts = timestamp
        
        drawing = image.copy()

        # Convert BGR image to RGB and then to mediapipe image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Process image with MediaPipe hands model in video mode
        results = self.hand_landmarker.detect_for_video(mp_image, timestamp)

        hands = []

        for handedness, hand_landmarks, hand_world_landmarks in zip(results.handedness, results.hand_landmarks, results.hand_world_landmarks):
            handedness = String(handedness[0].category_name.lower())
            hand_landmarks = [Point(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks]
            hand_world_landmarks = [Point(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_world_landmarks]

            # Draw the hand landmarks on the image
            drawing = self.draw_hand_points(drawing, hand_landmarks)

            hands.append(MpHand(handedness=handedness, hand_landmarks=hand_landmarks, hand_world_landmarks=hand_world_landmarks))
        
        # check if preempted
        if self.server.is_preempt_requested():
            self.server.set_preempted()
            return
        
        # return landmarks
        res = MpHandsModelResult(hands=hands)
        self.server.set_succeeded(res)

        # Publish the frame with the hand landmarks
        self.mp_drawing_publisher.publish(self.bridge.cv2_to_imgmsg(drawing, "bgr8"))


def main():
    default_node_name = 'hands_model_mediapipe'
    rospy.init_node(default_node_name, anonymous=False)

    HandsModelMediapipe()

    rospy.spin()


if __name__ == '__main__':
    main()
