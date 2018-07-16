#######################################################################
############################## Packages ###############################
#######################################################################

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pandas as pd
import cv2
import glob
import timeit

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image as img
from IPython.display import Image, display, clear_output



#######################################################################
################ Initialize Functions/Variables #######################
#######################################################################

# Science Thresholds

person_threshold = 0.50
person_gun_threshold = 0.80


# Intialize Tensorflow session and gpu memory management
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)



# Get Video and dimensions
cap = cv2.VideoCapture('overhead_east1.mp4')
width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)  # float
height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
os.chdir("/root/models/research/object_detection/")

# ## Object detection imports
# Here are the imports from the object detection module.


# Needed if you want to make bounded boxes around person for object detection
from utils import label_map_util
from utils import visualization_utils as vis_util

##################### Model preparation ###############################

# ## Variables
#
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.


# What model to download.
MODEL_NAME = 'faster_rcnn_resnet101_coco_2017_11_08'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box and select person class
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 1

#Load a Object Detection Model(frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Object Recognition model
label_lines = [line.rstrip() for line in tf.gfile.GFile("/tf_files/retrained_labels.txt")]


def initialSetup():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    start_time = timeit.default_timer()

    # This takes 2-5 seconds to run
    # Unpersists graph from file
    with tf.gfile.FastGFile('/tf_files/retrained_graph.pb', 'rb') as h:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(h.read())
        tf.import_graph_def(graph_def, name='')

    print 'Took {} seconds to unpersist the graph'.format(timeit.default_timer() - start_time)


initialSetup()


# ## Helper code for frame processing
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

count = 1
person_count = 1

while(True):
    # Capture frame-by-frame
    count += 1
    ret, image_np = cap.read()
    if count%5 == 0:

        # Loop Variables
        wid = []
        hei = []
        px  = []
        py =  []

        # with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # while True:
            # count += 1
            # ret, image_np = cap.read()
            # for image_path in TEST_IMAGE_PATHS:
            count += 1
            # image = img.open(image_path)
            # image_np = load_image_into_numpy_array(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
#             vis_util.visualize_boxes_and_labels_on_image_array(
#               image_np,
#               np.squeeze(boxes),
#               np.squeeze(classes).astype(np.int32),
#               np.squeeze(scores),
#               category_index,
#               use_normalized_coordinates=True,
#               line_thickness=8)


            aov = 55
            ch = 180
            imageHeight = int(height)
            imageWidth = int(width)
            imageHeightCenter = imageHeight / 2
            imageWidthCenter = imageWidth / 2
            pixelDegree = float(aov) / imageWidth

            # Convert tensorflow data to pandas data frams
            # print boxes
            df = pd.DataFrame(boxes.reshape(300, 4), columns=['y_min', 'x_min', 'y_max', 'x_max'])
            df1 = pd.DataFrame(classes.reshape(300, 1), columns=['classes'])
            df2 = pd.DataFrame(scores.reshape(300, 1), columns=['scores'])
            df5 = pd.concat([df, df1, df2], axis=1)

            # Transform box bound coordinates to pixel coordintate

            df5['y_min_t'] = df5['y_min'].apply(lambda x: x * imageHeight)
            df5['x_min_t'] = df5['x_min'].apply(lambda x: x * imageWidth)
            df5['y_max_t'] = df5['y_max'].apply(lambda x: x * imageHeight)
            df5['x_max_t'] = df5['x_max'].apply(lambda x: x * imageWidth)

            # Create objects pixel location x and y
            # X
            df5['ob_wid_x'] = df5['x_max_t'] - df5["x_min_t"]
            df5['ob_mid_x'] = df5['ob_wid_x'] / 2
            df5['x_loc'] = df5["x_min_t"] + df5['ob_mid_x']
            # Y
            df5['ob_hgt_y'] = df5['y_max_t'] - df5["y_min_t"]
            df5['ob_mid_y'] = df5['ob_hgt_y'] / 2
            df5['y_loc'] = df5["y_min_t"] + df5['ob_mid_y']

            # Find object degree of angle, data is sorted by score, select person with highest score
            df5['object_angle'] = df5['x_loc'].apply(lambda x: -(imageWidthCenter - x) * pixelDegree)

            df6 = df5.loc[(df5['classes'] == 1 ) %  (df5['scores'] > person_threshold)]
            #df6 = df6.loc[df6['scores'] > 0.80]

            if (df6.empty) or (df6.iloc[0]['scores'] < person_threshold):
                continue


            for i in range(0,len(df6.index)):

                df7 = df6.iloc[i]['object_angle']

                w = int(df6.iloc[i]['ob_wid_x'])
                x = int(df6.iloc[i]['x_min_t'])
                h = int(df6.iloc[i]['ob_hgt_y'])
                y = int(df6.iloc[i]["y_min_t"])

                AOB = df7 + ch
                AOB_str = str(round(AOB, 4))
                # print df6.head()
                # print imageHeight, imageWidth

                #labelBuffer = int(df6.iloc[0]['y_min_t']) - int(df6.iloc[0]['ob_hgt_y'] * 0.1)

                # print
                font = cv2.FONT_HERSHEY_SIMPLEX
                #cv2.putText(image_np, AOB_str, (int(df6.iloc[0]['x_min_t']), labelBuffer), font, 0.8, (0, 255, 0), 2)

                halfBody = h / 3
                bigBody1 = w * 3
                bigBody2 = x - (w * 2)
                #roi = image_np[y:y + halfBody, x:x + w]
                #cv2.rectangle(image_np, (x,y), (x+w, y+halfBody), (0, 255, 0), 2)
                #cv2.imwrite('save_image/' + "frame%d.jpg" % count, roi)

                print imageWidth, imageHeight, x, y, w, h, halfBody, df6.iloc[i]['scores']

                wid.append(w)
                hei.append(h)
                px.append(x)
                py.append(y)
                print boxes



        print wid, hei, px, py
        sess.close()


        #         cv2.imshow("Presentation Tracker", cv2.resize(roi, (640, 480)))
        #         plt.figure(figsize=IMAGE_SIZE)
        #         plt.imshow(roi)
        #         if cv2.waitKey(25) & 0xFF == ord('q'):
        #             cv2.destroyAllWindows()
        #             break




        # coding: utf-8

        # In[2]:



        for person in range(0,5):

            with tf.Session() as sess2:
                start_time = timeit.default_timer()

                # Feed the image_data as input to the graph and get first prediction
                softmax_tensor = sess2.graph.get_tensor_by_name('final_result:0')

                print 'Took {} seconds to feed data to graph'.format(timeit.default_timer() - start_time)

                # while True:
                #             frame = grabVideoFeed()

                #             if frame is None:
                #                 raise SystemError('Issue grabbing the frame')
                halfBody = hei[person] / 3
                roi = image_np[py[person]:py[person] + halfBody, px[person]:px[person] + wid[person]]

                frame = cv2.resize(roi, (299, 299), interpolation=cv2.INTER_CUBIC)

                # adhere to TS graph input structure
                numpy_frame = np.asarray(frame)
                numpy_frame = cv2.normalize(numpy_frame.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
                numpy_final = np.expand_dims(numpy_frame, axis=0)

                start_time = timeit.default_timer()

                # This takes 2-5 seconds as well
                predictions = sess2.run(softmax_tensor, {'Mul:0': numpy_final})
                score = predictions.item(1)
                gunScore = str(score)
                # print(predictions.item(1))


                print 'Took {} seconds to perform prediction'.format(timeit.default_timer() - start_time)

                start_time = timeit.default_timer()

                # Sort to show labels of first prediction in order of confidence
                top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

                print 'Took {} seconds to sort the predictions'.format(timeit.default_timer() - start_time)



                print '********* Session Ended *********'
                font = cv2.FONT_HERSHEY_SIMPLEX
                #cv2.rectangle(image_np, (px[person],py[person]), (px[person]+wid[person], py[person]+hei[person]), (0, 0, 255), 2)
                #cv2.rectangle(image_np, (px[person],py[person]), (px[person]+wid[person], py[person]+hei[person]), (0, 255, 0), 2)

                if score > person_gun_threshold:
                    person_count += 1
                    cv2.rectangle(image_np, (px[person],py[person]), (px[person]+wid[person], py[person]+hei[person]), (0, 0, 255), 10)
                    labelBuffer = int(py[person]) - int(hei[person] * 0.1)

                    # print
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(image_np, gunScore, (int(px[person]), labelBuffer), font, 0.8, (0, 255, 0), 2)

                    cv2.imwrite('save_image/' + "frame%d.jpg" % person_count, image_np)
                #cv2.putText(frame, gunScore, (10, 200), font, 0.8, (0, 255, 0), 2)
                sess2.close()

        # Display the resulting frame
        cv2.imshow('frame',cv2.resize(image_np, (1024, 768)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
