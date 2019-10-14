# Face Recognition with VGG_Face-net transfer learning.
### Step 1: Dataset acquisition
##### Train data
Gathered 50 images of 5 most powerful world leaders Trump,Putin,Jinping,Merkel and Modi of 10 images each.
Also my 10 images.
##### Test data
18 images with 3 images each of above 6 persons including me.

### Step 2: Detect faces.
Using `dlib cnn face detector` find faces in image and crop faces and store them in separate folders sorting by individual person. <br>
Download 'mmod_human_face_detector' to use as 'dlib cnn face detector',br>
```
! wget http://dlib.net/files/mmod_human_face_detector.dat.bz2 
!bzip2 -dk mmod_human_face_detector.dat.bz2
```
After extracting faces and storing them in corresponding folder the directory structure will be :
```
Directory structure :
|Images /
|  |-- (60 images)
|Images_crop /
|  |--angelamerkel
|     |--(10 images)
|  |--jinping / 
|     |--(10 images)
|  |--lakshminarayana / 
|         |--(10 imgaes)
|  |--modi / (10 images)
|  |--putin / (10 images) 
|  |--trump / (10 images)
|Images_test / 
|  |-- .. / (18 images)
|Images_test_crop / 
|  |--angelamerkel / (3 images)
|  |--jinping / (3 images)
|  |--lakshminarayana / (3 imgaes)
|  |--modi / (3 images)
|  |--putin / (3 images) 
|Face_Recognition.ipynb
|mmod_human_face_detector.dat
```
### Step 3: Download VGG_face weights.
As vgg-face weights are not available as `.h5` file to download,from this <a src='https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/'>article</a>
