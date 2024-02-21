#!/usr/bin/env python3

import cv2
import numpy as np
import rospy

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String

from pamaral_object_grasping_pattern_recognition.msg import MpResults


class DataVisualization:
    """
        Class to publish the image with the hand landmarks and the predicted object label drawn on it
    """

    def __init__(self, input_topic):
        self.object_class = ""
        self.last_image_msgs = []

        self.bridge = CvBridge()

        self.mp_drawing_publisher = rospy.Publisher("mp_drawing", Image, queue_size=1)

        self.object_class_sub = rospy.Subscriber("object_class", String, self.object_class_callback)
        self.image_sub = rospy.Subscriber(input_topic, Image, self.image_callback)
        self.mediapipe_results_sub = rospy.Subscriber("mediapipe_results", MpResults, self.mediapipe_results_callback)
    
    def object_class_callback(self, msg):
        self.object_class = msg.data
    
    def image_callback(self, msg):
        self.last_image_msgs.append(msg)
        self.last_image_msgs = self.last_image_msgs[-20:]
    
    def mediapipe_results_callback(self, msg):
        # Draw results on the image that was processed or else the oldest image
        matching_image_msg = None
        for image_msg in self.last_image_msgs:
            if image_msg.header.seq == msg.image_seq.data:
                matching_image_msg = image_msg
                break

        if matching_image_msg is None:
            matching_image_msg = self.last_image_msgs[0]
        
        try:
            # Convert the ROS Image message to OpenCV image
            image = self.bridge.imgmsg_to_cv2(matching_image_msg, "bgr8")

        except Exception as e:
            rospy.logerr(e)
            return

        for hand in msg.hands:
            hand = [[p.x, p.y, p.z] for p in hand.hand_landmarks]
            # Draw the hand landmarks on the image
            image = self.draw_hand_points(image, hand)
        
        cv2.putText(image, self.object_class, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 128), 5)

        # Publish the frame with the hand landmarks
        self.mp_drawing_publisher.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))

    def draw_hand_points(self, drawing, hand_landmarks):
        # Draw the points on the image
        point_radius = 8

        lines = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
                    [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15],
                    [15, 16], [0, 17], [17, 18], [18, 19], [19, 20], [13, 17]]
        
        # Define the polygon vertices
        polygon_points = np.array([hand_landmarks[0][:2], hand_landmarks[5][:2], hand_landmarks[9][:2], hand_landmarks[13][:2], hand_landmarks[17][:2]])

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
            cv2.line(drawing, [int(hand_landmarks[p1][0]*640), int(hand_landmarks[p1][1]*480)], [int(hand_landmarks[p2][0]*640), int(hand_landmarks[p2][1]*480)], [0, 255, 0], 2)

        for point in hand_landmarks:
            cv2.circle(drawing, [int(point[0]*640), int(point[1]*480)], point_radius, [0,0,255], -1)
        
        return drawing

def main():
    default_node_name = 'data_visualization'
    rospy.init_node(default_node_name, anonymous=False)

    input_topic = rospy.get_param(rospy.search_param('input_image_topic'))

    DataVisualization(input_topic=input_topic)

    rospy.spin()


if __name__ == '__main__':
    main()
