#!/usr/bin/env python3

import csv
import numpy as np
import os
import rosbag
import rospy

from sensor_msgs.msg import Image

from pamaral_object_grasping_pattern_recognition.msg import MediaPipeResults


class DatasetMediaPipePreprocessing:

    def __init__(self, input_folder, output_image_topic, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder

        # Iterate over all files in the folder
        self.filenames = []
        for filename in os.listdir(input_folder):
            # Create the absolute path to the file
            bag_path = os.path.join(input_folder, filename)

            # Check if the file path is a file (not a directory)
            if os.path.isfile(bag_path) and bag_path.endswith(".bag"):
                self.filenames.append(filename)
        
        # Create the Image Publisher        
        self.image_publisher = rospy.Publisher(output_image_topic, Image, queue_size=300)
        self.preprocessed_points_sub = rospy.Subscriber("mediapipe_results", MediaPipeResults, self.mediapipe_results_callback)

    def mediapipe_results_callback(self, msg):
        # Extract and write the hands keypoints from the message
        points = [[p.x, p.y, p.z] for p in msg.hands_keypoints]
        points = np.array(points)

        with open(self.hands_csv_path, 'a+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(points)
        
        # Extract and write the pose keypoints from the message
        points = [[p.x, p.y, p.z] for p in msg.pose_keypoints]
        points = np.array(points)

        with open(self.pose_csv_path, 'a+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(points)
        
        # Play next bag file
        self.num_messages -= 1
        if self.num_messages == 0:
            self.filenames = self.filenames[1:]
            self.play_next_bag_file()
    
    def play_next_bag_file(self):
        bag = rosbag.Bag(os.path.join(self.input_folder, self.filenames[0]))

        self.num_messages = bag.get_message_count()
        self.hands_csv_path = os.path.join(f"{self.output_folder}/hands_keypoints", self.filenames[0][:-4]+".csv")
        self.pose_csv_path = os.path.join(f"{self.output_folder}/pose_keypoints", self.filenames[0][:-4]+".csv")

        for topic, msg, t in bag.read_messages():
            self.image_publisher.publish(msg)


def main():
    default_node_name = 'dataset_processing'
    rospy.init_node(default_node_name, anonymous=False)

    input_folder = rospy.get_param(rospy.search_param('input_folder'))
    output_image_topic = rospy.get_param(rospy.search_param('output_image_topic'))
    output_folder = rospy.get_param(rospy.search_param('output_folder'))
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(f"{output_folder}/hands_keypoints", exist_ok=True)
    os.makedirs(f"{output_folder}/pose_keypoints", exist_ok=True)

    dataset_processing = DatasetMediaPipePreprocessing(input_folder=input_folder, output_image_topic=output_image_topic, output_folder=output_folder)

    input()

    dataset_processing.play_next_bag_file()

    rospy.spin()


if __name__ == '__main__':
    main()
