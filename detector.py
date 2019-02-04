import os
import cv2
import time
import random
import argparse

import tensorflow as tf
import matplotlib.pyplot as plt

import mrcnn.model as modellib
import custom

from tqdm import tqdm
from mrcnn import visualize

from keras.layers import *
from keras.models import Sequential

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="path to the input images")
ap.add_argument("-l", "--limit", required=False, help="optional input image limit")

args, leftovers = ap.parse_known_args()

def one_hot_label(img):
    label = img.split(".")[0]

    if "damage" in str(label):
        ohl = np.array([1,0])

    else:
        ohl = np.array([0,1])

    return ohl

def test_data_with_label(limit):
    test_images = []
    test_images_without_label = []

    if limit == 0:
        limit_new = len(os.listdir(test_data))

    else:
        limit_new = limit

    for i in tqdm(os.listdir(test_data)[:limit_new]):

        if i.startswith('.') == True:
            continue

        path = os.path.join(test_data, i)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256,256))
        test_images.append([np.array(img), one_hot_label(i)])

        img_n = cv2.imread(path)
        test_images_without_label.append((img_n, i))

    return (test_images, test_images_without_label)

test_data = args.path

if args.limit is None:
    print("Es wurde kein Limit für Input Bilder gesetzt.")
    testing_images, test_images_without_label = test_data_with_label(0)

else:
    testing_images, test_images_without_label = test_data_with_label(int(args.limit))

tst_img_data = np.array([i[0] for i in testing_images]).reshape(-1, 256,256,1)
tst_lbl_data = np.array([i[1] for i in testing_images])

model = Sequential()

model.add(InputLayer(input_shape=[256,256,1]))

model.add(Conv2D(filters=32, kernel_size=4, strides=1, padding="same", activation="sigmoid"))
model.add(MaxPool2D(pool_size=4, padding="same"))

model.add(Conv2D(filters=64, kernel_size=7, strides=1, padding="same", activation="sigmoid"))
model.add(MaxPool2D(pool_size=8, padding="same"))

model.add(Conv2D(filters=128, kernel_size=7, strides=1, padding="same", activation="sigmoid"))
model.add(MaxPool2D(pool_size=8, padding="same"))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation="sigmoid"))
model.add(Dropout(rate=0.25))
model.add(Dense(2, activation="softmax"))

custom_WEIGHTS_PATH = "my_modelmit50kbilder.h5"

model.load_weights(custom_WEIGHTS_PATH, by_name=True)

config = custom.CustomConfig()

class InferenceConfig(config.__class__):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

def get_ax(rows=1, cols=1, size=16):

    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

with tf.device(DEVICE):
    model_mask_rcnn = modellib.MaskRCNN(mode="inference", model_dir="logs", config=config)

custom_WEIGHTS_PATHx = "mask_rcnn_damage_00910.h5"

print("Loading weights ", custom_WEIGHTS_PATHx)
model_mask_rcnn.load_weights(custom_WEIGHTS_PATHx, by_name=True)

from importlib import reload
reload(visualize)

for data,data_wl in zip(testing_images,test_images_without_label):

    img = data[0]
    data = img.reshape(1,256,256,1)

    model_out = model.predict([data])

    if np.argmax(model_out) == 1:
        str_label="No Damage"
    else:
        str_label ="Damage"

    print(str_label)

    if str_label == "Damage":

        results = model_mask_rcnn.detect([data_wl[0]], verbose=1)

        xlist = list()

        # load the COCO class labels our Mask R-CNN was trained on
        labelsPath = "/Users/DHarun/Desktop/STD_MASTER/F_Bildverarbeitung/aim2/ABGABE/mask-rcnn2/mask-rcnn-coco/object_detection_classes_coco.txt"
        LABELS = open(labelsPath).read().strip().split("\n")

        # load the set of colors that will be used when visualizing a given
        # instance segmentation
        colorsPath = "/Users/DHarun/Desktop/STD_MASTER/F_Bildverarbeitung/aim2/ABGABE/mask-rcnn2/mask-rcnn-coco/colors.txt"
        COLORS = open(colorsPath).read().strip().split("\n")
        COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
        COLORS = np.array(COLORS, dtype="uint8")

        # derive the paths to the Mask R-CNN weights and model configuration
        weightsPath = "/Users/DHarun/Desktop/STD_MASTER/F_Bildverarbeitung/aim2/ABGABE/mask-rcnn2/mask-rcnn-coco/frozen_inference_graph.pb"
        configPath = "/Users/DHarun/Desktop/STD_MASTER/F_Bildverarbeitung/aim2/ABGABE/mask-rcnn2/mask-rcnn-coco/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"

        # load our Mask R-CNN trained on the COCO dataset (90 classes)
        # from disk
        print("[INFO] loading Mask R-CNN from disk...")
        net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

        # load our input image and grab its spatial dimensions
        image = data_wl[0]

        image_no_alpha = image.copy()
        (H, W) = image.shape[:2]

        # construct a blob from the input image and then perform a forward
        # pass of the Mask R-CNN, giving us (1) the bounding box  coordinates
        # of the objects in the image along with (2) the pixel-wise segmentation
        # for each specific object
        blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        (boxes, masks) = net.forward(["detection_out_final", "detection_masks"])
        end = time.time()

        # show timing information and volume information on Mask R-CNN
        print("[INFO] Mask R-CNN took {:.6f} seconds".format(end - start))
        print("[INFO] boxes shape: {}".format(boxes.shape))
        print("[INFO] masks shape: {}".format(masks.shape))

        # loop over the number of detected objects
        for ix in range(0, boxes.shape[2]):
            # extract the class ID of the detection along with the confidence
            # (i.e., probability) associated with the prediction
            classID = int(boxes[0, 0, ix, 1])
            confidence = boxes[0, 0, ix, 2]

            if confidence > 0.5:

                clone = image.copy()

                box = boxes[0, 0, ix, 3:7] * np.array([W, H, W, H])

                (startX, startY, endX, endY) = box.astype("int")

                boxW = endX - startX
                boxH = endY - startY

                rexs_box = boxW * boxH
                xlist.append((ix, rexs_box, boxW, boxH , endX, startX, endY, startY, confidence, classID))


        biggest_rexs = max(xlist, key=lambda p: p[1])


        for i in [biggest_rexs]:

            mask = masks[i[0], i[9]]

            mask = cv2.resize(mask, (i[2], i[3]), interpolation=cv2.INTER_CUBIC)

            mask = (mask > 0.3)

            roi = image[i[7]:i[6], i[5]:i[4]]

            if 1 > 0:

                visMask = (mask * 255).astype("uint8")
                instance = cv2.bitwise_and(roi, roi, mask=visMask)

            roi = roi[mask]

            color = random.choice(COLORS)

            blended = ((0.9 * color) + (0.0 * roi)).astype("uint8")

            image[i[7]:i[6], i[5]:i[4]][mask] = blended

            ax = get_ax(1)
            r = results[0]

            ### ALPHA 1

            suche_car = [0,229,0]

            result_car = np.count_nonzero((image == suche_car).all(axis = 2))

            save_img = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], "Damage", alpx=1.0)

            suche_damage = [255,0,0]

            result_damage = np.count_nonzero((save_img == suche_damage).all(axis = 2))


            total_pixels = result_car + result_damage

            schaden = result_damage / total_pixels

            schaden = schaden * 100

            ###

            blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")

            image_no_alpha[i[7]:i[6], i[5]:i[4]][mask] = blended

            color = [int(c) for c in color]

            save_imgx = visualize.display_instances(image_no_alpha, r['rois'], r['masks'], r['class_ids'], "Damage", alpx=0.4)

            text = "{0:.2f}% des Fahrzeugs ist beschaedigt".format(schaden)

            cv2.putText(save_imgx, text, (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

            cv2.imwrite("analyzed-{}.png".format(str(data_wl[1])[:-4]), save_imgx)


