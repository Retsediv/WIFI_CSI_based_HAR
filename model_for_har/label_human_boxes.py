import cv2
import numpy as np
import pandas as pd

import os
from os import listdir
from tqdm import tqdm

labels = open("./yolo_data/coco.names").read().strip().split("\n")


def detect_person_box(path_to_the_image):
    #############################################################################
    # Setup model
    #############################################################################
    CONFIDENCE = 0.5
    SCORE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5

    # the neural network configuration
    config_path = "./yolo_data/yolov3.cfg"

    # the YOLO net weights file
    weights_path = "./yolo_data/yolov3.weights"

    # loading all the class labels (objects)
    labels = open("./yolo_data/coco.names").read().strip().split("\n")

    # generating colors for each object for later plotting
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

    # load the YOLO network
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    #############################################################################
    # Load image
    #############################################################################
    image = cv2.imread(path_to_the_image)  # "../dataset/bedroom_lviv/1/images/opencv_frame_630.png"
    h, w = image.shape[:2]

    # create 4D blob
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    #     print("image.shape:", image.shape)
    #     print("blob.shape:", blob.shape) # should be blob.shape: (1, 3, 416, 416)

    #############################################################################
    # Making prediction
    #############################################################################

    # sets the blob as the input of the network
    net.setInput(blob)

    # get all the layer names
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # feed forward (inference) and get the network output
    # measure how much it took in seconds
    layer_outputs = net.forward(ln)

    boxes, confidences, class_ids = [], [], []

    # loop over each of the layer outputs
    for output in layer_outputs:
        # loop over each of the object detections
        for detection in output:
            # extract the class id (label) and confidence (as a probability) of
            # the current object detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # discard out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > CONFIDENCE:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # print("boxes: ", boxes)
    # print("confidences: ", confidences)
    # print("class_ids: ", class_ids)  # 0 -> person

    return boxes, class_ids


def class_id_to_string(id):
    return labels[id]


def main():
    DATALEN = 5228
    human_boxes = [0] * DATALEN
    portion = 10

    data_path = "./dataset/bedroom_lviv/1/"
    image_folder_path = os.path.join(data_path, "images")
    output_csv_path = os.path.join(data_path, "label_boxes.csv")

    files = listdir(image_folder_path)
    files = [os.path.join(image_folder_path, f) for f in files]  # add path to each file
    files.sort(key=lambda x: os.path.getmtime(x))  # sort by date

    start = 0
    for file_path in tqdm(files, total=len(files)):
        person_box = [0, 0, 0, 0]

        boxes, class_ids = detect_person_box(file_path)
        for i, class_id in enumerate(class_ids):
            if class_id == 0:  # this is person
                person_box = boxes[i]
                break

        human_boxes[start*portion:(start+1)*portion] = [person_box] * portion
        start += 1

    coords_df = pd.DataFrame(human_boxes, columns=["x", "y", "width", "height"])
    coords_df.to_csv(output_csv_path)


if __name__ == '__main__':
    main()
