import numpy as np
import cv2
import glob
from sklearn import svm
from sklearn.preprocessing import LabelEncoder



############################################################
#
#              Support Vector Machine
#              Image Classification
#
############################################################


# 1. Implement a SIFT feature extraction for a set of training images ./images/db/train/** (see 2.3 image retrieval)
# use ~15x15 keypoints on each image with subwindow of 21px (diameter)

def create_keypoints(w, h):
    keypoints = []
    keypointSize = 11

    # YOUR CODE HERE

    step = 4

    for y in range(0, h, step):
        for x in range(0, w, step):
            keypoints.append(cv2.KeyPoint(x, y, keypointSize))

    return keypoints

images = glob.glob('./images/db/train/*/*.jpg')

descriptors = []
image_paths = []
keypoints = create_keypoints(15, 15)
print(len(keypoints))

sift = cv2.SIFT_create()
for img_path in images:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (15, 15))
    _, desc = sift.compute(img, keypoints)
    if desc is not None:
        descriptors.append(desc.flatten())
        image_paths.append(img_path)


# 2. each descriptor (set of features) need to be flattened in one vector
# That means you need a X_train matrix containing a shape of (num_train_images, num_keypoints*num_entry_per_keypoint)
# num_entry_per_keypoint = histogram orientations as talked about in class
# You also need a y_train vector containing the labels encoded as integers

X_train = np.array(descriptors)
labels = [path.split('/')[-2] for path in image_paths]

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(labels)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"Unique labels: {label_encoder.classes_}")

# 3. We use scikit-learn to train a SVM classifier - however you need to test with different kernel options to get
# good results for our dataset.

# Different kernels: 'linear', 'rbf', 'poly', 'sigmoid'
clf = svm.SVC(kernel='rbf', C=1.0, gamma='scale')  # gamma='auto' or 'scale' for rbf/poly

clf.fit(X_train, y_train)

# 4. We test on a variety of test images ./images/db/test/ by extracting an image descriptor
# the same way we did for the training (except for a single image now) and use .predict()
# to classify the image

test_images = glob.glob('./images/db/test/*.jpg')

for path in test_images:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (15, 15))
    _, desc = sift.compute(img, keypoints)

    if desc is not None:
        x_test = desc.flatten().reshape(1, -1)  # reshape for single prediction

        prediction = clf.predict(x_test)
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        # 5. output the class + corresponding name
        print(f"Image: {path}")
        print(f"Predicted class: {predicted_label}")
    else:
        print("Desc is none")


