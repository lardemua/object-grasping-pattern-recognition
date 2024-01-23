# object-grasping-pattern-recognition

Object Recognition using the Object's Grasping Pattern

## Installation Guide

1. Install [ROS Noetic](https://wiki.ros.org/noetic/Installation/Ubuntu).

2. Clone this repository and its dependencies into your catkin workspace:

    ```
    cd ~/catkin_ws/src
    git clone https://github.com/lardemua/object_grasping_pattern_recognition.git
    ```

3. Compile the catkin workspace:

    ```
    cd ~/catkin_ws && catkin_make
    ```

6. Install python requirements:

    ```
    cd ~/catkin_ws/src/object_grasping_pattern_recognition
    pip install -r requirements.txt
    ```

## Usage Guide

### Record Data as Bag Files

### Extract MediaPipe Data from Bag Files

```
roslaunch pamaral_object_grasping_pattern_recognition dataset_mediapipe_preprocessing.launch
```

### Data Preprocessing

```
roslaunch pamaral_object_grasping_pattern_recognition dataset_keypoints_preprocessing.launch
```

### Train Model
