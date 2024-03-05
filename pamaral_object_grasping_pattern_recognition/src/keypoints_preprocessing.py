#!/usr/bin/env python3

import numpy as np
import rospy

from geometry_msgs.msg import Point

from pamaral_object_grasping_pattern_recognition.msg import MpResults, PointList


class KeypointsPreprocessing:

    def __init__(self):
        self.mp_results = None

        self.preprocessed_points_publisher = rospy.Publisher("preprocessed_points", PointList, queue_size=1)
        self.mediapipe_results_sub = rospy.Subscriber("mediapipe_results", MpResults, self.mediapipe_results_callback)

    def mediapipe_results_callback(self, msg):
        self.mp_results = msg
    
    def mediapipe_results_processing(self):
        rate = rospy.Rate(20)

        while not rospy.is_shutdown():
            if self.mp_results is not None:
                hands, pose, self.mp_results = self.mp_results.hands, self.mp_results.pose, None

                points = []

                if pose is not None and len(pose.pose_landmarks) > 0 and len(hands) > 0:
                    pose_keypoints = [[p.x, p.y, p.z] for p in pose.pose_landmarks]
                    hands_keypoints = [[[p.x, p.y, p.z] for p in hand.hand_landmarks] for hand in hands]

                    # obtain right and left hands centroids and valid radius
                    left_hand = np.array([pose_keypoints[15], pose_keypoints[17], pose_keypoints[19], pose_keypoints[21]])
                    right_hand = np.array([pose_keypoints[16], pose_keypoints[18], pose_keypoints[20], pose_keypoints[22]])

                    left_hand_centroid = np.average(left_hand, axis=0)
                    right_hand_centroid = np.average(right_hand, axis=0)

                    # left_hand_radius = 2*np.max(np.linalg.norm(left_hand_centroid - left_hand, axis=1))
                    # right_hand_radius = 2*np.max(np.linalg.norm(right_hand_centroid - right_hand, axis=1))

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
                
                if len(points) > 0:
                    centroid = np.average(points, axis=0)

                    points[:,0] -= centroid[0]
                    points[:,1] -= centroid[1]
                    points[:,2] -= centroid[2]

                    max_value = np.max(np.absolute(points))

                    points = (points/max_value + 1)/2

                msg = PointList()
                msg.header.stamp = rospy.Time.now()
                msg.header.frame_id = "mp_points"
                msg.points = [Point(p[0], p[1], p[2]) for p in points]
                self.preprocessed_points_publisher.publish(msg)
        
            rate.sleep()


def main():
    default_node_name = 'keypoints_preprocessing'
    rospy.init_node(default_node_name, anonymous=False)

    keypoints_preprocessing = KeypointsPreprocessing()

    keypoints_preprocessing.mediapipe_results_processing()

    rospy.spin()


if __name__ == '__main__':
    main()
