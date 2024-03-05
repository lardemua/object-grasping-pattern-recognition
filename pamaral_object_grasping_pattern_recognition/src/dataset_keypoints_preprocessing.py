#!/usr/bin/env python3

import json
import os
import rospy

from geometry_msgs.msg import Point
from std_msgs.msg import String

from pamaral_object_grasping_pattern_recognition.msg import MpHand, MpPose, MpResults, PointList


class DatasetKeypointsPreprocessing:

    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder

        # Iterate over all files in the folder
        self.filenames = []
        for filename in os.listdir(input_folder):
            # Create the absolute path to the file
            file_path = os.path.join(input_folder, filename)

            # Check if the file path is a file (not a directory)
            if os.path.isfile(file_path) and file_path.endswith(".json"):
                self.filenames.append(filename)
        
        # Create the MediaPipe Results Publisher
        self.mediapipe_results_publisher = rospy.Publisher("mediapipe_results", MpResults, queue_size=1)
        self.preprocessed_points_sub = rospy.Subscriber("preprocessed_points", PointList, self.preprocessed_points_callback)

    def preprocessed_points_callback(self, msg):
        # Extract and write the hands keypoints from the message
        points = [[p.x, p.y, p.z] for p in msg.points]

        self.data.append({'points': points})

        if len(self.mediapipe_results) > 0:
            self.mediapipe_results_publisher.publish(MpResults(**self.mediapipe_results.pop(0)))
        
        else:
            # Save the data to a JSON file
            rospy.loginfo("Saving {} to a JSON file".format(self.filenames[0]))
            with open(os.path.join(self.output_folder, self.filenames.pop(0)[:-5] + ".json"), 'a+') as file:
                json.dump(self.data, file)
            
            self.read_next_json_file()
    
    def read_next_json_file(self):
        if len(self.filenames)>0:
            # Reset the data
            self.data = []

            # Read the JSON file
            with open(os.path.join(self.input_folder, self.filenames[0]), 'r') as file:
                self.mediapipe_results = json.load(file)
            
            for i in range(len(self.mediapipe_results)):
                for j in range(len(self.mediapipe_results[i]['hands'])):
                    self.mediapipe_results[i]['hands'][j]['hand_landmarks'] = [Point(x=point[0], y=point[1], z=point[2]) for point in self.mediapipe_results[i]['hands'][j]['hand_landmarks']]
                    self.mediapipe_results[i]['hands'][j]['hand_world_landmarks'] = [Point(x=point[0], y=point[1], z=point[2]) for point in self.mediapipe_results[i]['hands'][j]['hand_world_landmarks']]
                    self.mediapipe_results[i]['hands'][j]['handedness'] = String(self.mediapipe_results[i]['hands'][j]['handedness'])
                    self.mediapipe_results[i]['hands'][j] = MpHand(**self.mediapipe_results[i]['hands'][j])

                self.mediapipe_results[i]['pose']['pose_landmarks'] = [Point(x=point[0], y=point[1], z=point[2]) for point in self.mediapipe_results[i]['pose']['pose_landmarks']]
                self.mediapipe_results[i]['pose']['pose_world_landmarks'] = [Point(x=point[0], y=point[1], z=point[2]) for point in self.mediapipe_results[i]['pose']['pose_world_landmarks']]
                self.mediapipe_results[i]['pose'] = MpPose(**self.mediapipe_results[i]['pose'])

            # Publish the 1st result
            self.mediapipe_results_publisher.publish(MpResults(**self.mediapipe_results.pop(0)))

        else:
            rospy.loginfo("All files processed")


def main():
    default_node_name = 'dataset_keypoints_preprocessing'
    rospy.init_node(default_node_name, anonymous=False)

    input_folder = rospy.get_param(rospy.search_param('input_folder'))
    output_folder = rospy.get_param(rospy.search_param('output_folder'))
    os.makedirs(output_folder, exist_ok=True)

    dataset_processing = DatasetKeypointsPreprocessing(input_folder=input_folder, output_folder=output_folder)

    input()

    dataset_processing.read_next_json_file()

    rospy.spin()


if __name__ == '__main__':
    main()
