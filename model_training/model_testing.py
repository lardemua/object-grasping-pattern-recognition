import numpy as np
import tensorflow as tf

from tensorflow.math import softmax
from utils import plt, os, json


def read_dataset(folder_path="./preprocessed_dataset", objects=["bottle", "cube", "phone", "screwdriver"],
                  people=["joel", "manuel", "pedro"], sessions=["1", "2", "3", "4"]):
    x = []
    y = []
    valid_indexes = []

    object_ids = {o: objects.index(o) for o in objects}

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        # Create the absolute path to the file
        file_path = os.path.join(folder_path, filename)

        # Check if the file path is a file (not a directory)
        if os.path.isfile(file_path) and file_path.endswith(".json"):
            filename = filename.split("_")
            object_name = filename[0]
            person = filename[1]

            if object_name in objects and person in people:
                # Open the file
                with open(file_path, "r") as f:
                    data = json.load(f)
                
                for i in range(len(data)):
                    points = data[i]["points"]

                    if len(points) > 0:
                        x.append(data[i]["points"])
                        y.append(object_ids[object_name])
                        valid_indexes.append(True)
                    
                    else:
                        x.append(np.zeros((21, 3)))
                        y.append(object_ids[object_name])
                        valid_indexes.append(False)

    x = np.array(x)
    y = np.array(y)
    valid_indexes = np.array(valid_indexes)

    return x, y, valid_indexes


def plot_softmaxes(softmaxes, valid_indexes, legend, show=True, save_path=False):
    plt.figure(figsize = (10,5))
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

    #plt.xticks(range(1, epochs+1))
    #plt.title(title)
    plt.legend(legend)
    plt.xlabel("Time (s)")
    plt.ylabel("Softmax Probability")

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
    
    if show:
        plt.show()


def plot_logits(softmaxes, valid_indexes, legend, show=True, save_path=False):
    plt.figure(figsize = (10,5))
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

    #plt.xticks(range(1, epochs+1))
    #plt.title(title)
    plt.legend(legend)
    plt.xlabel("Time (s)")
    plt.ylabel("Logits")

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
    
    if show:
        plt.show()


#CLASSES = ["ball", "bottle", "woodblock"]
#CLASSES = ["bottle", "cube", "phone", "screwdriver"]
CLASSES = ["bottle", "cube", "plier", "screwdriver"]

if __name__ == "__main__":
    # model training and evaluation
    model = tf.keras.models.load_model("./results_bottle_cube_plier_screw/cnn_model.h5")
    model.get_layer("dense_2").activation = None

    for object_tested in CLASSES:
        # read data
        x, y, valid_indexes = read_dataset(folder_path="./preprocessed_dataset/real_time_testing2", objects=[object_tested], people=["pedro"])

        # logits
        logits = model.predict(x)

        plot_logits(logits.T, valid_indexes, [o if len(o) <=6 else f"{o[:5]}." for o in CLASSES],
                            show=False, save_path = f"./results/{object_tested}_cnn_logits.svg")
        
        # softmaxes
        predictions = softmax(logits, axis=1).numpy()

        plot_softmaxes(predictions.T, valid_indexes, [o if len(o) <=6 else f"{o[:5]}." for o in CLASSES],
                            show=False, save_path = f"./results/{object_tested}_cnn_softmaxes.svg")
