#!/usr/bin/env python3

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # set to empty string to force CPU usage

import numpy as np
import rospy
import tensorflow as tf

from std_msgs.msg import String

from pamaral_object_grasping_pattern_recognition.msg import PointList


class ObjectClassifier:
    def __init__(self, cnn_model_path, transformer_model_path):
        self.cnn_model = tf.keras.models.load_model(cnn_model_path)
        self.transformer_model = tf.keras.models.load_model(transformer_model_path)
        
        # self.labels = ["bottle", "cube", "phone", "screwdriver"]
        self.labels = ["ball", "bottle", "woodblock"]

        self.object_class_pub = rospy.Publisher("object_class", String, queue_size=1)
        self.preprocessed_points_sub = rospy.Subscriber("preprocessed_points", PointList, self.preprocessed_points_callback)

    def preprocessed_points_callback(self, msg):
        if len(msg.points)>0:
            points = [[p.x, p.y, p.z] for p in msg.points]
            points = np.array(points)

            # make prediction using loaded model
            prediction1 = self.cnn_model.predict(tf.expand_dims(points, axis=0), verbose=0)
            prediction2 = self.transformer_model.predict(tf.expand_dims(points, axis=0), verbose=0)
            if np.max(prediction1) > 0.999 and np.max(prediction2) > 0.999:
                prediction1 = np.argmax(prediction1)
                prediction2 = np.argmax(prediction2)

                if prediction1 == prediction2:
                    prediction = self.labels[prediction1]

                    # publish prediction
                    self.object_class_pub.publish(prediction)


def main():
    default_node_name = 'object_classifier'
    rospy.init_node(default_node_name)

    cnn_model_path = rospy.get_param(rospy.search_param('cnn_model_path'))
    transformer_model_path = rospy.get_param(rospy.search_param('transformer_model_path'))

    ObjectClassifier(cnn_model_path, transformer_model_path)

    rospy.spin()
    

if __name__ == '__main__':
    main()
