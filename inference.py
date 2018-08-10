from __future__ import print_function
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import argparse
import os
from model import LeNet
from PIL import Image, ImageFilter

parser = argparse.ArgumentParser(description='Hand-written-character recognition')
parser.add_argument('--img_path', type=str, help='Input single image path Ex: test.jpg', required=True)
args = parser.parse_args()

#parameters
input_img_path = args.img_path
resize_img_width = 500
resize_img_height = 300
erode_factor = 6
dilate_factor = 2

#latest model checkpoint
model_ckpt_path = './model_ckpt'
model_name = 'LeNet_20'

#read image
img = cv2.imread(input_img_path)
img = cv2.resize(img, (resize_img_width,resize_img_height)) #100-200 diference
img = cv2.erode(img, np.ones((1, erode_factor)))
img = cv2.dilate(img, np.ones((1, dilate_factor)))

#output char mapping list
def createOutputCharMapping():
    char_mapping = {}
    digits = ['0','1','2','3','4','5','6','7','8','9']
    uppercase_alphabet = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    lowercase_alphabet = ['a','b','d','e','f','g','h','n','q','r','t']
    total_characters = 47
    index = 0
    
    for i in range(len(digits)):
        char_mapping[index] = digits[i]
        index=index + 1
        
    for i in range(len(uppercase_alphabet)):
        char_mapping[index] = uppercase_alphabet[i]
        index=index + 1
        
    for i in range(len(lowercase_alphabet)):
        char_mapping[index] = lowercase_alphabet[i]
        index=index + 1
    
    assert total_characters == index
    
    return char_mapping 
        

def imageprepare(argv):
    """
    This function returns the pixel values.
    The input is a png file location.
    """
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

    tv = list(newImage.getdata())  # get pixel values
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    return tva



def getBoundingBox():
    #convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER_create()
    coordinates, bboxes = mser.detectRegions(gray)
    #format x , y, w, h
    points=[]
    for bbox in bboxes:
        x, y, w, h = bbox
        if [x,y,x+w,y+h] not in points:
            points.append([x,y,x+w,y+h])
    #check inside
    flag=[True]*len(points)
    for p in points:
        x1,y1,x2,y2=p
        for i in range(len(points)):
            if p!=points[i] and flag[i]:
                a1,b1,a2,b2=points[i]
                if a1>=x1 and a2<=x2 and b1>=y1 and b2<=y2:
                    flag[i]=False       
                
    total_bboxes=[]
    for i in range(len(flag)):
        if flag[i]:
            total_bboxes.append(points[i])
            
    total_bboxes.sort()
    print("Total no. of characters to be found = " + str(len(total_bboxes)) )
    
    return total_bboxes
    

def convertToEMNISTformat(total_bboxes):
    """
    This function converts input image
    to EMNIST format image
    """
    batch_size = len(total_bboxes)
    test_input_x = np.zeros((batch_size, 784))#emnist image vector size
    index = 0
    for bbox in total_bboxes:
        x1,y1,x2,y2 = bbox
        cv2.imwrite('./cv2_reference_img.png',img[y1:y2,x1:x2])
        emnist_x = imageprepare('./cv2_reference_img.png')#file path here
        test_input_x[index] = np.array(emnist_x)
        index = index +1
        if os.path.exists('./cv2_reference_img.png'):
            os.remove('./cv2_reference_img.png')
    
    return test_input_x 
        
        
def runInference(test_input_x):
    """Test inference for given image"""
    tf.reset_default_graph()  
    xs = tf.placeholder(tf.float32, [None, 784], name='input')
    dropout = tf.placeholder(tf.float32, name='dropout')
    
    #get the model logits
    model_logits = LeNet(xs , dropout)    
    output = tf.argmax(tf.contrib.layers.softmax(model_logits), 1)
    
    
    if not os.path.exists(model_ckpt_path):
        raise ValueError('[!] model Checkpoint path does not exist...')
         
    try:  
        with tf.Session() as sess:
            print('[*] Reading checkpoint...')
            saver = tf.train.Saver()
            saver.restore(sess, os.path.join(model_ckpt_path, model_name))
            output = sess.run(output, feed_dict={xs: test_input_x, dropout: 0.0})
    except Exception as e:
        print(e)
            
    return output

def getResultOutput(output):
    result=''
    char_mapping = createOutputCharMapping()
    for key in output:
        result = result + char_mapping[key]
    return result   

  
if __name__ == '__main__':
    
    total_bboxes = getBoundingBox()
    test_input_x = convertToEMNISTformat(total_bboxes)
    output = runInference(test_input_x)
    result = getResultOutput(output)
    print('\n')
    print('*'*50)
    print("Model Output is = " + result)
    
    
     
