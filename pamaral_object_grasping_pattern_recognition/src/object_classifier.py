#!/usr/bin/env python3

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # set to empty string to force CPU usage

import numpy as np
import rospy
import tensorflow as tf

from std_msgs.msg import String

from pamaral_object_grasping_pattern_recognition.msg import PointList


class ObjectClassifier:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        # self.labels = ["bottle", "cube", "phone", "screwdriver"]
        self.labels = ["ball", "bottle", "woodblock"]

        self.object_class_pub = rospy.Publisher("object_class", String, queue_size=1)
        self.preprocessed_points_sub = rospy.Subscriber("preprocessed_points", PointList, self.preprocessed_points_callback)

    def preprocessed_points_callback(self, msg):
        if len(msg.points)>0:
            points = [[p.x, p.y, p.z] for p in msg.points]
            points = np.array(points)

            # make prediction using loaded model
            prediction = self.model.predict(tf.expand_dims(points, axis=0), verbose=0)
            if np.max(prediction) > 0.99:
                prediction = np.argmax(prediction)
                prediction = self.labels[prediction]

                # publish prediction
                self.object_class_pub.publish(prediction)


def main():
    default_node_name = 'object_classifier'
    rospy.init_node(default_node_name)

    model_path = rospy.get_param(rospy.search_param('model_path'))

    ObjectClassifier(model_path)

    rospy.spin()
    

if __name__ == '__main__':
    main()
