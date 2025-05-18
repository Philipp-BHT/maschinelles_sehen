import cv2
import glob
from queue import PriorityQueue
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

############################################################
#
#              Simple Image Retrieval
#
############################################################


# implement distance function
def distance(a, b):
    return np.linalg.norm(a - b)



def create_keypoints(w, h):
    keypoints = []
    keypointSize = 11

    # YOUR CODE HERE

    step = 4

    for y in range(0, h, step):
        for x in range(0, w, step):
            keypoints.append(cv2.KeyPoint(x, y, keypointSize))

    return keypoints


# 1. preprocessing and load
images = glob.glob('./images/db/train/*/*.jpg')

# 2. create keypoints on a regular grid (cv2.KeyPoint(r, c, keypointSize), as keypoint size use e.g. 11)
descriptors = []
image_paths = []
keypoints = create_keypoints(256, 256)


# 3. use the keypoints for each image and compute SIFT descriptors
#    for each keypoint. this compute one descriptor for each image.

# YOUR CODE HERE

sift = cv2.SIFT_create()
for img_path in images:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    _, desc = sift.compute(img, keypoints)
    if desc is not None:
        descriptors.append(desc.flatten())
        image_paths.append(img_path)

# 4. use one of the query input image to query the 'image database' that
#    now compress to a single area. Therefore extract the descriptor and
#    compare the descriptor to each image in the database using the L2-norm
#    and save the result into a priority queue (q = PriorityQueue())

# YOUR CODE HERE

test_img = cv2.imread('./images/db/test/car.jpg', cv2.IMREAD_ANYCOLOR)
query_img = cv2.resize(test_img, (256, 256))
_, query_desc = sift.compute(query_img, keypoints)
query_desc = query_desc.flatten()

q = PriorityQueue()

for i, desc in enumerate(descriptors):
    d = distance(query_desc, desc)
    q.put((d, image_paths[i]))

# 5. output (save and/or display) the query results in the order of smallest distance

# YOUR CODE HERE

fig = plt.figure(figsize=(22, 5))
gs = GridSpec(2, 11, figure=fig)


ax_query = fig.add_subplot(gs[:, 0])
query_display_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
ax_query.imshow(query_display_img)
ax_query.set_title("Query")
ax_query.axis('off')

for i in range(len(images)):
    row = i // 10
    col = i % 10 + 1
    ax = fig.add_subplot(gs[row, col])

    _, path = q.get()
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ax.imshow(img)
    ax.axis('off')

plt.tight_layout()
plt.show()
