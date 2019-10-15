import os
import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import ZeroPadding2D,Convolution2D,MaxPooling2D
from tensorflow.keras.layers import Dense,Dropout,Softmax,Flatten,Activation,BatchNormalization
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow.keras.backend as K

#Define VGG_FACE_MODEL architecture
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))

# Load VGG Face model weights
model.load_weights('vgg_face_weights.h5')

# Remove Last Softmax layer and get model upto last flatten layer with outputs 2622 units
vgg_face=Model(inputs=model.layers[0].input,outputs=model.layers[-2].output)

# Load saved model
classifier_model=tf.keras.models.load_model('face_classifier_model.h5')

dnnFaceDetector=dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

# Label names for class numbers
person_rep={0:'Lakshmi Narayana',
 1: 'Vladimir Putin',
 2: 'Angela Merkel',
 3: 'Narendra Modi',
 4: 'Donald Trump',
 5: 'Xi Jinping'}

if __name__ == '__main__':
  file_path=input("Path to image with file size < 100 kb ? ")

  img=cv2.imread(file_path)
  if img is None or img.size is 0 :
    print("Please check image path or some error occured")

  else:
    persons_in_img=[]
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect Faces
    rects=dnnFaceDetector(gray,1)
    left,top,right,bottom=0,0,0,0
    for (i,rect) in enumerate(rects):
      # Extract Each Face
      left=rect.rect.left() #x1
      top=rect.rect.top() #y1
      right=rect.rect.right() #x2
      bottom=rect.rect.bottom() #y2
      width=right-left
      height=bottom-top
      img_crop=img[top:top+height,left:left+width]
      cv2.imwrite(os.getcwd()+'/crop_img.jpg',img_crop)
    
      # Get Embeddings
      crop_img=load_img(os.getcwd()+'/crop_img.jpg',target_size=(224,224))
      crop_img=img_to_array(crop_img)
      crop_img=np.expand_dims(crop_img,axis=0)
      crop_img=preprocess_input(crop_img)
      img_encode=vgg_face(crop_img)

      # Make Predictions
      embed=K.eval(img_encode)
      person=classifier_model.predict(embed)
      name=person_rep[np.argmax(person)]
      os.remove(os.getcwd()+'/crop_img.jpg')
      cv2.rectangle(img,(left,top),(right,bottom),(0,255,0), 2)
      img=cv2.putText(img,name,(left,top-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2,cv2.LINE_AA)
      img=cv2.putText(img,str(np.max(person)),(right,bottom+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)
      persons_in_img.append(name)
    # Save images with bounding box,name and accuracy 
    cv2.imwrite(os.getcwd()+'/recognized_img.jpg',img)
    
    #Person in image
    print('Person(s) in image is/are:')
    print(persons_in_img)

    plt.figure(figsize=(8,4))
    plt.imshow(img[:,:,::-1])
    plt.axis('off')
    plt.show()
