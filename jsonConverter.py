import sys
import json

JSON_NAME = sys.argv[1]
IMG_PATH = sys.argv[2]

with open(JSON_NAME, 'r') as f:
    dictionary = json.load(f)

MAX_IMAGE_INDEX = dictionary['images'][len(dictionary['images'])-1]['id']

f = open("info.dat", "a")

for image_id in range(1, MAX_IMAGE_INDEX+1):
    image_entry = dictionary['images'][image_id-1]
    file_name = image_entry['file_name']
    image_annotations = []
    annotations = dictionary['annotations']
    for el in annotations:
        if el['image_id'] == image_id:
            image_annotations.append(el['bbox'])
        elif el['image_id'] > image_id:
            break
    string = IMG_PATH+"/"+file_name + "  " + str(len(image_annotations)) + "  "
    for i in range(len(image_annotations)):
        r = [str(x) for x in image_annotations[i]]
        for _ in range(4):
            string += r[_] + " "
    string = string[:len(string)-2]
    f.write(string)
    f.write('\n')
    
f.close()