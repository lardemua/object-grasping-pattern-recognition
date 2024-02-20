import json
import os

# Specify the folder path you want to read files from
folder_path = "./preprocessed_dataset"

counts = dict()
overall = 0

# Check if the folder path exists
if os.path.exists(folder_path) and os.path.isdir(folder_path):
    # Iterate over all files in the folder
    for filename in sorted(os.listdir(folder_path)):
        # Create the absolute path to the file
        file_path = os.path.join(folder_path, filename)

        # Check if the file path is a file (not a directory)
        if os.path.isfile(file_path) and file_path.endswith(".json"):
            with open(file_path, "r") as f:
                data = json.load(f)
            
            count = len(data)
            
            overall += count

            filename = filename.split("_")
            object_name = filename[0]
            participant_name = filename[1]
            session = filename[2][0]

            if object_name not in counts:
                counts[object_name] = dict()
            
            if participant_name not in counts[object_name]:
                counts[object_name][participant_name] = [count]
            else:
                counts[object_name][participant_name].append(count)

else:
    print("The specified folder does not exist or is not a directory.")

for k,v in counts.items():
    print(k)
    for k2,v2 in v.items():
        print(k2, v2)
    print()

print("Overall: ", overall)
