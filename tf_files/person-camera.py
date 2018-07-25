
# coding: utf-8

# In[7]:


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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image as img
from IPython.display import Image, display, clear_output
from elasticsearch import Elasticsearch

# Hold warnings
import warnings
warnings.filterwarnings('ignore')



#######################################################################
################ Initialize Functions/Variables #######################
#######################################################################
# Global Variables

# Camera
RECEPTION_EAST="rtsp://admin:1qazxsw2!QAZXSW@@datascience.opswerx.org:20043"
RECEPTION_WEST = "rtsp://admin:1qazxsw2!QAZXSW@@datascience.opswerx.org:20044"
DIRTYWERX_NORTH="rtsp://admin:1qazxsw2!QAZXSW@@datascience.opswerx.org:20045"
DIRTYWERX_SOUTH="rtsp://admin:1qazxsw2!QAZXSW@@datascience.opswerx.org:20046"
THUNDERDRONE_INDOOR_EAST="rtsp://admin:1qazxsw2!QAZXSW@@datascience.opswerx.org:20047"
THUNDERDRONE_INDOOR_WEST="rtsp://admin:1qazxsw2!QAZXSW@@datascience.opswerx.org:20048"
OUTSIDE_WEST="rtsp://admin:1qazxsw2!QAZXSW@@datascience.opswerx.org:20049"
OUTSIDE_NORTH_WEST="rtsp://admin:1qazxsw2!QAZXSW@@datascience.opswerx.org:20050"
OUTSIDE_NORTH="rtsp://admin:1qazxsw2!QAZXSW@@datascience.opswerx.org:20051"
OUTSIDE_NORTH_EAST="rtsp://admin:1qazxsw2!QAZXSW@@datascience.opswerx.org:20052"
DIRTYWERX_RAMP="rtsp://admin:1qazxsw2!QAZXSW@@datascience.opswerx.org:20053"
LOCAL="/tensorflow/models/research/object_detection/vid.mp4"


# Setup ES 
try:
    es = Elasticsearch(
        [
            'https://elastic:diatonouscoggedkittlepins@elasticsearch.orange.opswerx.org:443'
        ],
        verify_certs=True
    )
    print("ES - Connected.")
except Exception as ex:
    print("Error: ", ex)


# GPU Percentage
gpuAmount = int((sys.argv)[2]) * 0.1


# Camera Selection
url = globals()[str((sys.argv)[1])]
print url


# Science Thresholds
person_threshold = 0.50
person_gun_threshold = 0.60


# Intialize Tensorflow session and gpu memory management
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = gpuAmount
session = tf.Session(config=config)

os.chdir("/tensorflow/models/research/object_detection/")

# Get Video and dimensions
#cap = cv2.VideoCapture('draw9.mp4')








#url = "rtsp://admin:1qazxsw2!QAZXSW@@datascience.opswerx.org:20044"
cap = cv2.VideoCapture(url)
print cap

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print width, height

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
os.chdir("/tensorflow/models/research/object_detection/")

# ## Object detection imports
# Here are the imports from the object detection module.


# Needed if you want to make bounded boxes around person for object detection
from utils import label_map_util
from utils import visualization_utils as vis_util

##################### Model Preparation ###############################

# ## Variables
#
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.


# What model to download.
# What model to download.
MODEL_NAME ='faster_rcnn_resnet101_coco_2017_11_08'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
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




#######################################################################
############# Perform object Detection and Recognition ################
#######################################################################



################# Person Object Detection #############################

count = 1
person_count = 1



####################Video Variables##########################


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # float)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print ("fps " + str(fps))



# Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

################################################################33


# Loop Frame by Frame
while(cap.isOpened()):
    # Capture frame-by-frame
    start_time = timeit.default_timer()
    count += 1
    ret, image_np = cap.read()
    if count%30 == 0:
        ret, image_np = cap.read()
        if ret == True:
            start_time = timeit.default_timer()

            # Loop Variables
            wid = []
            hei = []
            head_hei = []
            px  = []
            py =  []
            pxa = []
            pya = []
            pyha = []

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
                
               # print scores, classes
                
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)


                # Convert tensorflow data to pandas data frams
                # print boxes
                df = pd.DataFrame(boxes.reshape(300, 4), columns=['y_min', 'x_min', 'y_max', 'x_max'])
                df1 = pd.DataFrame(classes.reshape(300, 1), columns=['classes'])
                df2 = pd.DataFrame(scores.reshape(300, 1), columns=['scores'])
                df4 = pd.concat([df, df1, df2], axis=1)
                df5 = df4[(df4.classes == 1) & (df4.scores > person_threshold)]
                #df5 = df4.loc[(df4['classes'] == 1 ) &  (df4['scores'] > person_threshold)]
                #print df5

                # Transform box bound coordinates to pixel coordintate

                df5['y_min_t'] = df5['y_min'].apply(lambda x: x * height)
                df5['x_min_t'] = df5['x_min'].apply(lambda x: x * width)
                df5['y_max_t'] = df5['y_max'].apply(lambda x: x * height)
                df5['x_max_t'] = df5['x_max'].apply(lambda x: x * width)

                # Create objects pixel location x and y
                # X
                df5['ob_wid_x'] = df5['x_max_t'] - df5["x_min_t"]
                df5['ob_mid_x'] = df5['ob_wid_x'] / 2
                df5['x_loc'] = df5["x_min_t"] + df5['ob_mid_x']
                # Y
                df5['ob_hgt_y'] = df5['y_max_t'] - df5["y_min_t"]
                df5['ob_mid_y'] = df5['ob_hgt_y'] / 2
                df5['y_loc'] = df5["y_min_t"] + df5['ob_mid_y']

                df6 = df5


                # Scan People in Frame

                if (df6.empty) or (df6.iloc[0]['scores'] < person_threshold):
                    continue


                for i in range(0,len(df6.index)):


                    w = int(df6.iloc[i]['ob_wid_x'])
                    x = int(df6.iloc[i]['x_min_t'])
                    h = int(df6.iloc[i]['ob_hgt_y'])
                    y = int(df6.iloc[i]["y_min_t"])

                    wid.append(w)
                    hei.append(h)
                    px.append(x)
                    py.append(y)


                sess.close()
                print 'Took {} seconds to find people in image'.format(timeit.default_timer() - start_time)




    ##################### Perform Pistol Recognition #######################


            start_time = timeit.default_timer()
            
            
            ipx = []
            ipy = []
            iwid = []
            ihei = []
                   

            for person in range(0,len(df6.index) - 1):

                with tf.Session() as sess2:

                    # Feed the image_data as input to the graph and get first prediction
                    softmax_tensor = sess2.graph.get_tensor_by_name('final_result:0')

                    upperbody = hei[person] / 3
                    roi = image_np[py[person]:py[person] + upperbody, px[person]:px[person] + wid[person]]

                    frame = cv2.resize(roi, (299, 299), interpolation=cv2.INTER_CUBIC)

                    # adhere to TS graph input structure
                    numpy_frame = np.asarray(frame)
                    numpy_frame = cv2.normalize(numpy_frame.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
                    numpy_final = np.expand_dims(numpy_frame, axis=0)


                    # make prediciton
                    predictions = sess2.run(softmax_tensor, {'Mul:0': numpy_final})

                    score = predictions.item(1)
                    gunScore = str(score)
                    
#                     ipx = []
#                     ipy = []
#                     iwid = []
#                     ihei = []

                    # Add Red Box to image under conditions
                    if score > person_gun_threshold:
                        person_count += 1
                        cv2.rectangle(image_np, (px[person],py[person]), (px[person]+wid[person], py[person]+ hei[person]), (0, 0, 255), 10)
                        print(px[person],py[person],wid[person],hei[person])
                        #ipx.append()
                        labelBuffer = int(py[person]) - int(hei[person] * 0.1)

                        # print
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(image_np, gunScore, (int(px[person]), labelBuffer), font, 0.8, (0, 255, 0), 2)
                        
                         # Package bounding box info for ES
                        xmin = px[person] 
                        xmax = (px[person] + wid[person])
                        ymin = py[person] 
                        ymax = (py[person] + hei[person])
                        
                                            
                        tdoc = {
                            #'timestamp': datetime.now(),
                            'content': 'Video information',
                            'text': 'Object detected.',
                            'xmin': xmin,
                            'xmax': xmax,
                            'ymin': ymin,
                            'ymax': ymax,
                         }
                    
                        # Send results to ES
                        if url == RECEPTION_EAST:
                            res = es.index(index="reception-east", doc_type="_doc", body=tdoc)
                            print('ES document sent.')
                            print(tdoc)
                        elif url == RECEPTION_WEST:
                            res = es.index(index="reception-west", doc_type="_doc", body=tdoc)
                            print('ES document sent.')
                            print(tdoc)
                        elif url == OUTSIDE_WEST:
                            res = es.index(index="outside-west", doc_type="_doc", body=tdoc)
                            print('ES document sent.')
                            print(tdoc)
                        elif url == LOCAL:
                            res = es.index(index="test", doc_type="_doc", body=tdoc)
                            print('ES document sent.')
                            print(tdoc)
                            
                        # Save Full Image and Save Object Image
                        #cv2.imwrite('/tf_files/save_image/'+ str((sys.argv)[1]) +"-frame%d.jpg" % person_count, image_np)
                        #cv2.imwrite('/tf_files/save_threat_image/' + str((sys.argv)[1]) + "-frame%d.jpg" % person_count, roi)



                    #cv2.putText(frame, gunScore, (10, 200), font, 0.8, (0, 255, 0), 2)
                    sess2.close()
                    print 'Took {} seconds to perform image recognition on people found'.format(timeit.default_timer() - start_time)

        # Display the resulting frame
            #cv2.rectangle(image_np, (px[person],py[person]), (px[person]+wid[person], py[person]+ hei[person]), (0, 0, 255), 10)
            #cv2.rectangle(image_np, (px[person],py[person]), (px[person]+wid[person] - 100, py[person]+ hei[person] - 100), (0, 0, 255), 10)
#             out.write(image_np)
            cv2.imshow('frame',cv2.resize(image_np, (1024, 768)))
        else:
              cap = cv2.VideoCapture(url)
              print("Error in frame capture")

            

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
out.release()
cv2.destroyAllWindows()        

