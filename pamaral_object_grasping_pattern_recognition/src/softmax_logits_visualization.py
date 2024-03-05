#!/usr/bin/env python3

import cv2
import numpy as np
import rospy

from cv_bridge import CvBridge
import matplotlib.pyplot as plt
from sensor_msgs.msg import Image
from PIL import Image as PILImage
import io

from pamaral_object_grasping_pattern_recognition.msg import ObjectPrediction


class DataVisualization:
    """
        Class to publish the image with the hand landmarks and the predicted object label drawn on it
    """

    def __init__(self, input_topic):        
        self.last_image_msgs = []

        self.valid_indexes = []
        self.softmaxes = []
        self.logits = []

        self.bridge = CvBridge()

        self.softmax_and_logits_image_publisher = rospy.Publisher("softmax_and_logits_image", Image, queue_size=1)

        self.object_class_sub = rospy.Subscriber("object_class", ObjectPrediction, self.object_class_callback)
        self.image_sub = rospy.Subscriber(input_topic, Image, self.image_callback)
        
    def image_callback(self, msg):
        self.last_image_msgs.append(msg)
        self.last_image_msgs = self.last_image_msgs[-2:]
    
    def object_class_callback(self, msg):
        self.valid_indexes.append(msg.is_valid.data)
        self.softmaxes.append([p.data for p in msg.softmaxes])
        self.logits.append([p.data for p in msg.logits])

        #self.valid_indexes = self.valid_indexes[-100:]
        #self.softmaxes = self.softmaxes[-100:]
        #self.logits = self.logits[-100:]

        try:
            # Convert the ROS Image message to OpenCV image
            image = self.bridge.imgmsg_to_cv2(self.last_image_msgs[0], "bgr8")

        except Exception as e:
            rospy.logerr(e)
            return
        
        final_image = np.ones((720, 1280, 3), dtype=np.uint8) * 255

        final_image[120:600, 10:650, :] = image
        
        softmaxes_image = plot_softmaxes(np.array(self.softmaxes).T, self.valid_indexes,
                                         ["bottle", "cube", "plier", "screw."], show=False)
        
        logits_image = plot_logits(np.array(self.logits).T, self.valid_indexes,
                                      ["bottle", "cube", "plier", "screw."], show=False)

        final_image[0:360, 670:1270, :] = logits_image
        final_image[360:720, 670:1270, :] = softmaxes_image

        # Publish the frame with the hand landmarks
        self.softmax_and_logits_image_publisher.publish(self.bridge.cv2_to_imgmsg(final_image, "bgr8"))


def plt2img(fig):
    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = PILImage.open(buf).convert('RGB')
    img = np.array(img)
    img = img[:, :, ::-1].copy()
    return img 

def plot_softmaxes(softmaxes, valid_indexes, legend, show=True, save_path=False):
    plt.figure(figsize = (5,3), dpi=120)
    for softmax in softmaxes:
        x = []
        y = []

        for i in range(len(softmax)):
            if valid_indexes[i]:
                x.append(i/20)
                y.append(softmax[i])
            
            else:
                x.append(None)
                y.append(None)
                
        plt.plot(x, y, marker=".")

    plt.xlim([0, 4] if len(softmaxes[0])/20 <= 4 else [len(softmaxes[0])/20-4, len(softmaxes[0])/20])
    #plt.xticks(range(1, epochs+1))
    #plt.title(title)
    plt.legend(legend)
    plt.xlabel("Time (s)")
    plt.ylabel("Softmax Probability")

    plt.tight_layout()
    
    img = plt2img(plt)
    plt.close()

    return img

def plot_logits(softmaxes, valid_indexes, legend, show=True, save_path=False):
    plt.figure(figsize = (5,3), dpi=120)
    for softmax in softmaxes:
        x = []
        y = []

        for i in range(len(softmax)):
            if valid_indexes[i]:
                x.append(i/20)
                y.append(softmax[i])
            
            else:
                x.append(None)
                y.append(None)
                
        plt.plot(x, y, marker=".")

    plt.xlim([0, 4] if len(softmaxes[0])/20 <= 4 else [len(softmaxes[0])/20-4, len(softmaxes[0])/20])
    #plt.xticks(range(1, epochs+1))
    #plt.title(title)
    plt.legend(legend)
    plt.xlabel("Time (s)")
    plt.ylabel("Logits")

    plt.tight_layout()

    img = plt2img(plt)
    plt.close()

    return img

def main():
    default_node_name = 'data_visualization'
    rospy.init_node(default_node_name, anonymous=False)

    input_topic = rospy.get_param(rospy.search_param('input_image_topic'))

    DataVisualization(input_topic=input_topic)

    rospy.spin()


if __name__ == '__main__':
    main()
