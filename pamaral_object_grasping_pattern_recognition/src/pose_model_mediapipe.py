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
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

from pamaral_object_grasping_pattern_recognition.msg import MpPose, MpPoseModelAction, MpPoseModelResult


class PoseModelMediapipe:

    def __init__(self):
        # Initialize Pose Landmarker
        base_options = BaseOptions(model_asset_path=f"{os.environ['HOME']}/catkin_ws/src/object_grasping_pattern_recognition/pamaral_object_grasping_pattern_recognition/models/pose_landmarker_full.task")
        self.options = PoseLandmarkerOptions(base_options=base_options, running_mode=RunningMode.VIDEO)
        self.pose_landmarker = None # PoseLandmarker.create_from_options(self.options)

        self.bridge = CvBridge()

        self.last_ts = 0

        self.server = actionlib.SimpleActionServer('pose_model', MpPoseModelAction, self.execute, False)
        self.server.start()

        rospy.loginfo("Pose Detection Ready")

    def execute(self, msg):
        try:
            # Convert the ROS Image message to OpenCV image
            image = self.bridge.imgmsg_to_cv2(msg.image, "bgr8")

            # Get the timestamp of the image in miliseconds
            timestamp = int(msg.image.header.stamp.to_sec()*1000)

        except Exception as e:
            rospy.logerr(e)
            return
        
        if timestamp < self.last_ts or timestamp - self.last_ts > 1000:
            self.pose_landmarker = PoseLandmarker.create_from_options(self.options)
        
        self.last_ts = timestamp
        
        # Convert BGR image to RGB and then to mediapipe image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Process image with MediaPipe pose model in video mode
        results = self.pose_landmarker.detect_for_video(mp_image, timestamp)

        pose = None

        # If the pose was detected, extract the coordinates of each landmark
        for pose_landmarks, pose_world_landmarks in zip(results.pose_landmarks, results.pose_world_landmarks):
            pose_landmarks = [Point(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks]
            pose_world_landmarks = [Point(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_world_landmarks]

            pose = MpPose(pose_landmarks=pose_landmarks, pose_world_landmarks=pose_world_landmarks)

        # check if preempted
        if self.server.is_preempt_requested():
            self.server.set_preempted()
            return
        
        # return landmarks
        res = MpPoseModelResult(pose=pose)
        self.server.set_succeeded(res)


def main():
    default_node_name = 'pose_model_mediapipe'
    rospy.init_node(default_node_name, anonymous=False)

    PoseModelMediapipe()

    rospy.spin()


if __name__ == '__main__':
    main()
