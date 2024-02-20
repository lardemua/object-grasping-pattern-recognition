#!/usr/bin/env python3

import rospy

from std_msgs.msg import String

count = 0

def object_class_callback(msg):
    global count
    print(f"{count} - {msg.data}")
    count+=1

default_node_name = 'test_node'
rospy.init_node(default_node_name)

sub = rospy.Subscriber("object_class", String, object_class_callback)

rospy.spin()