'''
Script to test traffic light localization and detection
Note: 1) Use gamma correction to improve the detection
      2) Use image cropped from (unprocessed) original image for classification
'''

import numpy as np
import cv2
import tensorflow as tf
#from keras.models import load_model
from PIL import Image
import os
from matplotlib import pyplot as plt
import time
from glob import glob



cwd = os.path.dirname(os.path.realpath(__file__))


class TLClassifier(object):
    def __init__(self):

        self.signal_classes = ['Red', 'Green', 'Yellow']
       
        self.signal_status = None
        
        self.tl_box = None
        
        self.cls_model_no = 1;
        
        os.chdir(cwd)
        
        #tensorflow localization/classification model
        #************************************************************
        
        
        #detect_model_name = 'traffic_light_inference_graph_site'
        #detect_model_name = 'tl_inference_may_2018'
        #detect_model_name ='tl_infer_11710_may_2018'
        #detect_model_name = 'tl_infer_12729_0518'
        detect_model_name ='tl_infer_12036_0518'
        
            
        #************************************************************    
        
        #detect_model_name = 'ssd_inception_v2_coco_2017_11_17'
        PATH_TO_CKPT = detect_model_name + '/frozen_inference_graph.pb'
        # setup tensorflow graph
        self.detection_graph = tf.Graph()
        
        # configuration for possible GPU use
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # load frozen tensorflow detection model and initialize 
        # the tensorflow graph
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
               serialized_graph = fid.read()
               od_graph_def.ParseFromString(serialized_graph)
               tf.import_graph_def(od_graph_def, name='')
               
            self.sess = tf.Session(graph=self.detection_graph, config=config)
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
              # Each box represents a part of the image where a particular object was detected.
            self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
              # Each score represent how level of confidence for each of the objects.
              # Score is shown on the result image, together with the class label.
            self.scores =self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections =self.detection_graph.get_tensor_by_name('num_detections:0')
    
    # Helper function to convert image into numpy array    
    def load_image_into_numpy_array(self, image):
         (im_width, im_height) = image.size
         return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)       
    
    # Helper function to convert normalized box coordinates to pixels
    def box_normal_to_pixel(self, box, dim):
    
        height, width = dim[0], dim[1]
        box_pixel = [int(box[0]*height), int(box[1]*width), int(box[2]*height), int(box[3]*width)]
        return np.array(box_pixel)       
     
    # Main function for end-to-end bounding box localization and light color
    # classification
    def get_localization_classification(self, image, visual=False):  
        """Determines the locations of the traffic light in the image

        Args:
            image: camera image

        Returns:
            box: list of integer for coordinates [x_left, y_up, x_right, y_down]
            conf: confidence
            cls_idx: 1->Green, 2->Red, 3->Yellow, 4->Unknown (No detection)

        """
    
#        category_index={1: {'id': 1, 'name': 'Green'},
#                        2: {'id': 2, 'name': 'Red'},
#                        3: {'id': 3, 'name': 'Yellow'}}  
        
        with self.detection_graph.as_default():
              image_expanded = np.expand_dims(image, axis=0)
              (boxes, scores, classes, num_detections) = self.sess.run(
                  [self.boxes, self.scores, self.classes, self.num_detections],
                  feed_dict={self.image_tensor: image_expanded})
              
              
              boxes=np.squeeze(boxes)  # bounding boxes
              classes =np.squeeze(classes) # classes
              scores = np.squeeze(scores) # confidence
    
              cls = classes.tolist()
              
              
        
             
              # Find the most confident detection/classification
              idx = 0;
              conf = scores[idx]
              cls_idx = cls[idx]
              
              
              # If there is no detection
              if idx == None:
                  box=[0, 0, 0, 0]
                  print('no detection!')
                  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                  cls_idx = 4.0
                  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
              # If the confidence of detection is too slow, 0.3 for simulator    
             
              #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
              elif scores[idx]<=0.16:  # was 0.15 before
              #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!    
                  box=[0, 0, 0, 0]
                  print('low confidence:', scores[idx])
                  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                  cls_idx = 4.0
                  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
              #If there is a detection and its confidence is high enough    
              else:
                  #*************corner cases***********************************
                  dim = image.shape[0:2]
                  box = self.box_normal_to_pixel(boxes[idx], dim)
                  box_h = box[2] - box[0]
                  box_w = box[3] - box[1]
                  ratio = box_h/(box_w + 0.01)
                  # if the box is too small, 20 pixels for simulator
                  if (box_h <10) or (box_w<10):
                      box =[0, 0, 0, 0]
                      #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                      cls_idx =4.0
                      #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                      print('box too small!', box_h, box_w)
                      
                  # if the h-w ratio is not right, 1.5 for simulator    
                  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                  elif (ratio<1.0):
                  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!    
                      box =[0, 0, 0, 0]
                      #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                      cls_idx = 4.0
                      #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                      print('wrong h-w ratio', ratio)
                  else:    
                       print(box)
                       print('localization confidence: ', scores[idx])
                 #****************end of corner cases***********************      
              self.tl_box = box
             
        return box, conf, cls_idx
        
    

if __name__ == '__main__':
        
        tl_cls =TLClassifier()
        
        os.chdir(cwd)
        #TEST_IMAGE_PATHS= glob(os.path.join('traffic_light_images/', '*.jpg'))
        crop_up, crop_bottom, crop_left, crop_right = 0, 600, 0, 800
        
        no_correct_cls=0;
        det_conf = [];
        low_conf =[];
        wrong_color=[];
        failed_img_names =[];
        #TEST_IMAGE_PATHS= glob(os.path.join('classification/yellow/', '*.*'))
        TEST_IMAGE_PATHS= glob(os.path.join('2018.05.21/Red/', '*.*'))
        #TEST_IMAGE_PATHS= glob(os.path.join('stress_test/Sim', '*.*'))
        TEST_IMAGES = TEST_IMAGE_PATHS
        
        PLOT_IMAGE = False
        for i, image_path in enumerate(TEST_IMAGES):
            print('')
            print('*************************************************')
            
            img = Image.open(image_path)
            img_np = tl_cls.load_image_into_numpy_array(img)
            
            img_local = np.copy(img_np)
            img_local = img_local[crop_up:crop_bottom, 
                                              crop_left:crop_right]
            
            
            start = time.time()
            
            b, conf, cls_idx = tl_cls.get_localization_classification(img_local, visual=False)
            
            end = time.time()
            print('Localization time: ', end-start)
            print(cls_idx)
            
              
            if cls_idx ==1.0:
                  tl_cls.signal_status ="Green"
            elif cls_idx ==2.0:
                  tl_cls.signal_status ="Red"
            elif cls_idx ==3.0:
                  tl_cls.signal_status ="Yellow"
            else:
                  tl_cls.signal_status ="Unknown"      
                  
            
           
            # If there is no detection or low-confidence detection
            if PLOT_IMAGE:
                if np.array_equal(b, np.zeros(4)):
                   plt.figure(figsize=(8,6))
                   plt.imshow(img_np)
                   plt.title(tl_cls.signal_status)
                   plt.show()
                else:    
                   cv2.rectangle(img_np,(b[1],b[0]),(b[3],b[2]),(0,255,0),2)
                   plt.figure(figsize=(8,6))
                   plt.imshow(img_np)
                   plt.title(tl_cls.signal_status)
                   plt.show()
               
      
            det_conf.append(conf)        
            if (tl_cls.signal_status == "Red"): 
                   no_correct_cls +=1
                   
            else:
                   failed_img_names.append(image_path)
                   wrong_color.append(tl_cls.signal_status)
                   low_conf.append(conf)
        acc = no_correct_cls/len(TEST_IMAGES)
        print('Accuracy: ', acc)       
              
        # Plot histogram, change facecolor, title accordingly 
        plt.hist(det_conf, facecolor ='Red')
        plt.title('Red May 2018, Acc='+str(acc)[0:5])
        plt.xlabel('Classification confidence')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()
          