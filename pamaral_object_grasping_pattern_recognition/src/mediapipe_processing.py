#!/usr/bin/env python3

import actionlib
import rospy

from sensor_msgs.msg import Image
from std_msgs.msg import UInt32

from pamaral_object_grasping_pattern_recognition.msg import MpResults, MpHandsModelAction, MpPoseModelAction, MpHandsModelGoal, MpPoseModelGoal

class MediaPipeProcessing:

    def __init__(self, input_topic):
        self.input_topic = input_topic

        # Initialize Action Clients for MediaPipe Nodes
        self.hands_model_client = actionlib.SimpleActionClient('hands_model', MpHandsModelAction)
        self.pose_model_client = actionlib.SimpleActionClient('pose_model', MpPoseModelAction)
        self.hands_model_client.wait_for_server()
        self.pose_model_client.wait_for_server()

        # Initialize Publisher for MediaPipe Results
        self.mediapipe_results_publisher = rospy.Publisher("mediapipe_results", MpResults, queue_size=1)
        
        # Initialize Subscriber for Input Images
        self.has_new_msg = False
        self.last_msg = None
        self.image_sub = rospy.Subscriber(input_topic, Image, self.image_callback)

    def image_callback(self, msg):
        self.last_msg, self.has_new_msg = msg, True
    
    def process_image(self):
        while True:
            if self.has_new_msg:
                msg, self.has_new_msg = self.last_msg, False

                # Send image to Mediapipe nodes
                self.hands_model_client.send_goal(MpHandsModelGoal(image=msg))
                self.pose_model_client.send_goal(MpPoseModelGoal(image=msg))

                # Wait for results
                self.hands_model_client.wait_for_result()
                self.pose_model_client.wait_for_result()

                # Get results
                hands = self.hands_model_client.get_result().hands
                pose = self.pose_model_client.get_result().pose

                # Publish results
                self.mediapipe_results_publisher.publish(hands=hands, pose=pose, image_seq = UInt32(msg.header.seq))


def main():
    default_node_name = 'mediapipe_processing'
    rospy.init_node(default_node_name, anonymous=False)

    input_topic = rospy.get_param(rospy.search_param('input_image_topic'))

    mediapipe_processing = MediaPipeProcessing(input_topic=input_topic)

    mediapipe_processing.process_image()

    rospy.spin()


if __name__ == '__main__':
    main()
