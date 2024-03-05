#!/usr/bin/env python3

import rospy

from pamaral_object_grasping_pattern_recognition.msg import ObjectPrediction

count = 0

def object_class_callback(msg):
    global count
    print(f"{count} - {msg.object_class.data}")
    count+=1

default_node_name = 'test_node'
rospy.init_node(default_node_name)

sub = rospy.Subscriber("object_class", ObjectPrediction, object_class_callback)

rospy.spin()