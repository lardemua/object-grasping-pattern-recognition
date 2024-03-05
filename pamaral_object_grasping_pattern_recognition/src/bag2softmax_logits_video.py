#!/usr/bin/env python3

import cv2
import json
import os
import rosbag
import rospy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class DatasetMediaPipePreprocessing:

    def __init__(self, input_folder, output_image_topic, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder

        self.bridge = CvBridge()

        self.out = cv2.VideoWriter(os.path.join(output_folder, 'screwdriver.mp4'),cv2.VideoWriter_fourcc(*'MP4V'), 20, (1280,720))

        # Iterate over all files in the folder
        self.filenames = ["real_time_testing2/screwdriver_pedro_04_03_2024_13_54_03.bag"]
        
        # Create the Image Publisher        
        self.image_publisher = rospy.Publisher(output_image_topic, Image, queue_size=1)
        self.softmax_logits_image_sub = rospy.Subscriber("softmax_and_logits_image", Image, self.softmax_logits_image_callback)

    def softmax_logits_image_callback(self, msg):
        try:
            # Convert the ROS Image message to OpenCV image
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        except Exception as e:
            rospy.logerr(e)
            return

        self.out.write(image)
        
        if len(self.messages) > 0:
            self.image_publisher.publish(self.messages.pop(0))
        
        else:
            # Save the data to a JSON file
            rospy.loginfo("Saving video...")
            self.out.release()
            rospy.loginfo("Video saved")
    
    def read_next_bag_file(self):
        if len(self.filenames) > 0:
            # Reset the data
            self.data = []

            # Read the bag file
            bag = rosbag.Bag(os.path.join(self.input_folder, self.filenames[0]))
            msgs = bag.read_messages()

            # Sort the messages by publication timestamp
            msgs = sorted(msgs, key=lambda x: x[1].header.stamp.secs * 1e9 + x[1].header.stamp.nsecs)
            
            self.messages = [x[1] for x in msgs]

            # Publish 1st message
            self.image_publisher.publish(self.messages.pop(0))
        
        else:
            print("All bag files processed")


def main():
    default_node_name = 'dataset_mediapipe_processing'
    rospy.init_node(default_node_name, anonymous=False)

    input_folder = rospy.get_param(rospy.search_param('input_folder'))
    output_image_topic = rospy.get_param(rospy.search_param('output_image_topic'))
    output_folder = rospy.get_param(rospy.search_param('output_folder'))
    os.makedirs(output_folder, exist_ok=True)

    dataset_mediapipe_processing = DatasetMediaPipePreprocessing(input_folder=input_folder, output_image_topic=output_image_topic, output_folder=output_folder)

    input()

    dataset_mediapipe_processing.read_next_bag_file()

    rospy.spin()


if __name__ == '__main__':
    main()
