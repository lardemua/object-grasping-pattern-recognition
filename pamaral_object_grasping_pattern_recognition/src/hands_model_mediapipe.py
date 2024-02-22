#!/usr/bin/env python3

# set to empty string to force CPU usage
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import actionlib
import cv2
import mediapipe as mp
import rospy

from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from std_msgs.msg import String

from pamaral_object_grasping_pattern_recognition.msg import MpHand, MpHandsModelAction, MpHandsModelResult


class HandsModelMediapipe:

    def __init__(self):
        # Initialize Hand Landmarker
        base_options = BaseOptions(model_asset_path=f"{os.environ['HOME']}/catkin_ws/src/object_grasping_pattern_recognition/pamaral_object_grasping_pattern_recognition/models/hand_landmarker.task")
        self.options = HandLandmarkerOptions(base_options=base_options, running_mode=RunningMode.VIDEO, num_hands=2)
        self.hand_landmarker = None # HandLandmarker.create_from_options(self.options)

        self.bridge = CvBridge()

        self.last_ts = 0

        self.server = actionlib.SimpleActionServer('hands_model', MpHandsModelAction, self.execute, False)
        self.server.start()

        rospy.loginfo("Hand Model Ready")

    def execute(self, msg):
        try:
            # Convert the ROS Image message to OpenCV image
            image = self.bridge.imgmsg_to_cv2(msg.image, "bgr8")

            # Get the timestamp of the image in miliseconds
            timestamp = int(msg.image.header.stamp.to_sec()*1000)

        except Exception as e:
            rospy.logerr(e)
            return
        
        if timestamp < self.last_ts or timestamp - self.last_ts > 2000:
            self.hand_landmarker = HandLandmarker.create_from_options(self.options)
        
        self.last_ts = timestamp

        # Convert BGR image to RGB and then to mediapipe image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Process image with MediaPipe hands model in video mode
        results = self.hand_landmarker.detect_for_video(mp_image, timestamp)

        hands = []

        for handedness, hand_landmarks, hand_world_landmarks in zip(results.handedness, results.hand_landmarks, results.hand_world_landmarks):
            handedness = String(handedness[0].category_name.lower())
            hand_landmarks = [Point(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks]
            hand_world_landmarks = [Point(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_world_landmarks]

            hands.append(MpHand(handedness=handedness, hand_landmarks=hand_landmarks, hand_world_landmarks=hand_world_landmarks))
        
        # check if preempted
        if self.server.is_preempt_requested():
            self.server.set_preempted()
            return
        
        # return landmarks
        res = MpHandsModelResult(hands=hands)
        self.server.set_succeeded(res)


def main():
    default_node_name = 'hands_model_mediapipe'
    rospy.init_node(default_node_name, anonymous=False)

    HandsModelMediapipe()

    rospy.spin()


if __name__ == '__main__':
    main()
