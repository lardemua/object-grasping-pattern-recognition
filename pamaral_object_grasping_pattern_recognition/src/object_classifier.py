#!/usr/bin/env python3

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # set to empty string to force CPU usage

import numpy as np
import rospy
import tensorflow as tf

from std_msgs.msg import String, Bool, Float32
from tensorflow.math import softmax

from pamaral_object_grasping_pattern_recognition.msg import PointList, ObjectPrediction


class ObjectClassifier:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.model.get_layer("dense_2").activation = None
        
        # self.labels = ["ball", "bottle", "woodblock"]
        # self.labels = ["bottle", "cube", "phone", "screwdriver"]
        self.labels = ["bottle", "cube", "plier", "screwdriver"]

        self.preprocessed_points = None

        self.object_class_pub = rospy.Publisher("object_class", ObjectPrediction, queue_size=1)
        self.preprocessed_points_sub = rospy.Subscriber("preprocessed_points", PointList, self.preprocessed_points_callback)

    def preprocessed_points_callback(self, msg):
        self.preprocessed_points = msg.points

    def preprocessed_points_processing(self):
        rate = rospy.Rate(20)

        while not rospy.is_shutdown():
            if self.preprocessed_points is not None:
                points, self.preprocessed_points = self.preprocessed_points, None
                msg = ObjectPrediction()

                if len(points)>0:
                    msg.is_valid = Bool(True)
                    msg.object_class = String("")

                    points = [[p.x, p.y, p.z] for p in points]
                    points = np.array(points)

                    logits = self.model.predict(tf.expand_dims(points, axis=0), verbose=0)

                    softmaxes = softmax(logits, axis=1).numpy()

                    msg.logits = [Float32(p) for p in logits[0].tolist()]
                    msg.softmaxes = [Float32(p) for p in softmaxes[0].tolist()]

                    if np.max(logits) > 5:
                        prediction = np.argmax(logits)
                        msg.object_class = String(self.labels[prediction])
                
                else:
                    msg.softmaxes = [Float32(0) for _ in range(len(self.labels))]
                    msg.logits = [Float32(0) for _ in range(len(self.labels))]
                    msg.is_valid = Bool(False)
                    
                self.object_class_pub.publish(msg)
                

            rate.sleep()


def main():
    default_node_name = 'object_classifier'
    rospy.init_node(default_node_name)

    model_path = rospy.get_param(rospy.search_param('model_path'))

    object_classifier = ObjectClassifier(model_path)

    object_classifier.preprocessed_points_processing()

    rospy.spin()
    

if __name__ == '__main__':
    main()
