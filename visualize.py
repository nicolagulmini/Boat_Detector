import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import sys
import json
import random

JSON_NAME = sys.argv[1]
IMG_PATH = sys.argv[2]

with open(JSON_NAME, 'r') as f:
    dictionary = json.load(f)
MAX_IMAGE_INDEX = dictionary['images'][len(dictionary['images'])-1]['id']

image_id = random.choice([i for i in range(MAX_IMAGE_INDEX)])
image_entry = dictionary['images'][image_id]
file_name = image_entry['file_name']

im = Image.open(IMG_PATH+"/"+file_name)
fig, ax = plt.subplots()
ax.imshow(im)

image_annotations = []
annotations = dictionary['annotations']
for el in annotations:
    if el['image_id'] == image_id+1:
        image_annotations.append(el['bbox'])
    elif el['image_id'] > image_id+1:
        break

for i in range(len(image_annotations)):
    rectangle = image_annotations[i]
    rect = patches.Rectangle((rectangle[0], rectangle[1]), rectangle[2], rectangle[3], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

plt.show()