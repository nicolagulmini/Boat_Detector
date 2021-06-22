import os
import json
import sys

JSON_PATH = sys.argv[1]
FILES_PATH = sys.argv[2]

files = os.listdir(FILES_PATH)

with open(JSON_PATH, 'r') as f:
    dictionary = json.load(f)

MAX_IMAGE_INDEX = dictionary['images'][len(dictionary['images'])-1]['id']
positive_files = [dictionary['images'][image_id]['file_name'] for image_id in range(MAX_IMAGE_INDEX)]

negative_files = list(set(files) - set(positive_files))

f = open("bg.txt", "a")
for el in negative_files:
    f.write(FILES_PATH+"/"+el+"\n")
f.close()