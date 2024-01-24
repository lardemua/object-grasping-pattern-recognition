#!/usr/bin/env python3

import actionlib
import rospy

from sensor_msgs.msg import Image

from pamaral_object_grasping_pattern_recognition.msg import MediaPipeResults, HandsModelAction, PoseModelAction, HandsModelGoal, PoseModelGoal

class MediaPipeProcessing:

    def __init__(self, input_topic):
        self.input_topic = input_topic

        # Initialize Action Clients for MediaPipe Nodes
        self.hands_model_client = actionlib.SimpleActionClient('hands_model', HandsModelAction)
        self.pose_model_client = actionlib.SimpleActionClient('pose_model', PoseModelAction)
        self.hands_model_client.wait_for_server()
        self.pose_model_client.wait_for_server()

        # Initialize Publisher for MediaPipe Results
        self.mediapipe_results_publisher = rospy.Publisher("mediapipe_results", MediaPipeResults, queue_size=1)
        
        # Initialize Subscriber for Input Images
        self.image_sub = rospy.Subscriber(input_topic, Image, self.image_callback)

    def image_callback(self, msg):
        # Send image to Mediapipe nodes
        self.hands_model_client.send_goal(HandsModelGoal(image=msg))
        self.pose_model_client.send_goal(PoseModelGoal(image=msg))

        # Wait for results
        self.hands_model_client.wait_for_result()
        self.pose_model_client.wait_for_result()

        # Get results
        handednesses = self.hands_model_client.get_result().handednesses
        hands_keypoints = self.hands_model_client.get_result().points
        pose_keypoints = self.pose_model_client.get_result().points

        # Publish results
        self.mediapipe_results_publisher.publish(handednesses=handednesses, hands_keypoints=hands_keypoints, pose_keypoints=pose_keypoints)


def main():
    default_node_name = 'mediapipe_processing'
    rospy.init_node(default_node_name, anonymous=False)

    input_topic = rospy.get_param(rospy.search_param('input_image_topic'))

    MediaPipeProcessing(input_topic=input_topic)

    rospy.spin()


if __name__ == '__main__':
    main()
