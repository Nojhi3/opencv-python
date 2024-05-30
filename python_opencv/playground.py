import cv2
import numpy as np
import sys, os

'''imagepath = "./images/img1.jpeg"

img = cv2.imread(imagepath,0)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

bdo = './videos/video.mp4'

# /////////////To train //////////////////////////////////////////////////
'''
haar_file = './xmls/haarcascade_frontalface_default.xml'

# All the faces data will be 
# present this folder 
datasets = 'datasets'
sub_data = 'aks_data'	


path = os.path.join(datasets, sub_data) 
if not os.path.isdir(path): 
    os.mkdir(path)

# defining the size of images 
(width, height) = (130, 100)	 

#'0' is used for my webcam, 
# if you've any other camera 
# attached use '1' like this 
face_cascade = cv2.CascadeClassifier(haar_file) 
webcam = cv2.VideoCapture(0) 

# The program loops until it has 30 images of the face. 
count = 1
while count < 30: 
	(_, im) = webcam.read() 
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
	faces = face_cascade.detectMultiScale(gray, 1.3, 4) 
	for (x, y, w, h) in faces: 
		cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2) 
		face = gray[y:y + h, x:x + w] 
		face_resize = cv2.resize(face, (width, height)) 
		cv2.imwrite('% s/% s.png' % (path, count), face_resize) 
	count += 1
	
	cv2.imshow('OpenCV', im) 
	key = cv2.waitKey(10) 
	if key == 27: 
		break
'''
#//////////////////////////gm//////////////////////////////////////////////

#///////////////////////////after training///////////////////////

'''
size = 4

haar_file = './xmls/haarcascade_frontalface_default.xml'
path = './datasets/aks_data'
datasets = './datasets'

# Part 1: Create fisherRecognizer 
print('Recognizing Face Please Be in sufficient Lights...') 

# Create a list of images and a list of corresponding names 
(images, labels, names, id) = ([], [], {}, 0) 
for (subdirs, dirs, files) in os.walk(datasets): 
	for subdir in dirs: 
		names[id] = subdir 
		subjectpath = os.path.join(datasets, subdir) 
		for filename in os.listdir(subjectpath): 
			path = subjectpath + '/' + filename 
			label = id
			images.append(cv2.imread(path, 0)) 
			labels.append(int(label)) 
		id += 1
(width, height) = (130, 100) 

# Create a Numpy array from the two lists above 
(images, labels) = [np.array(lis) for lis in [images, labels]] 

# OpenCV trains a model from the images 
# NOTE FOR OpenCV2: remove '.face' 
model = cv2.LBPHFaceRecognizer_create() 
model.train(images, labels) 

# Part 2: Use fisherRecognizer on camera stream 
face_cascade = cv2.CascadeClassifier(haar_file) 
webcam = cv2.VideoCapture(0) 
while True: 
    (_, im) = webcam.read() 
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    for (x, y, w, h) in faces: 
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2) 
        face = gray[y:y + h, x:x + w] 
        face_resize = cv2.resize(face, (width, height)) 
		# Try to recognize the face 
        prediction = model.predict(face_resize) 
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3) 
        if prediction[1]<500: 
            cv2.putText(im, '% s - %.0f' %(names[prediction[0]], prediction[1]), (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0)) 
        else:
            cv2.putText(im, 'not recognized', (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0)) 
    cv2.imshow('OpenCV', im) 
    key = cv2.waitKey(10) 
    if key == 27: 
        break


'''
#////////////////////////////////////////////////////////////////////////





'''cap = cv2.VideoCapture(bdo)

while(cap.isOpened()):
    ret, frame = cap.read()
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,'Test karra balla gm',(1500,2000), font, 2,(0,255,255),2,cv2.LINE_4)
    
    cv2.imshow('video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()'''

#////////////////////////////face detetctor////////////////////////////////////
'''
face_cascade = cv2.CascadeClassifier('./xmls/haarcascade_frontalface_default.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()

if not os.path.exists('faces'):
    os.mkdir('faces')
    
cap = cv2.VideoCapture(0)

count = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors = 5, minSize = (30,30))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (100, 100))
        cv2.imwrite('faces/user.' + str(count) + '.jpg', face_resize)
        count += 1
    
    cv2.imshow('face_frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    if count == 300:
        break
    
cap.release()
cv2.destroyAllWindows()

'''
#////////////////////////////////////////////////////////////////

import mediapipe as mp
import time

mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    static_image_mode = False,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
)

mp_drawing = mp.solutions.drawing_utils

capture = cv2.VideoCapture(0)

prevtime = 0
curtime = 0

while capture.isOpened():
    ret, frame = capture.read()
    frame = cv2.resize(frame, (800,600))
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    '''
    #for face detetion
    mp_drawing.draw_landmarks(
      image,
      results.face_landmarks,
      mp_holistic.FACEMESH_CONTOURS,
      mp_drawing.DrawingSpec(
        color=(255,0,255),
        thickness=1,
        circle_radius=1
      ),
      mp_drawing.DrawingSpec(
        color=(0,255,255),
        thickness=1,
        circle_radius=1
      )
    )
    '''
    
    
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS
    )
    
    mp_drawing.draw_landmarks(
      image, 
      results.left_hand_landmarks, 
      mp_holistic.HAND_CONNECTIONS
    )
    
    curtime = time.time()
    fps = 1/(curtime-prevtime)
    prevtime = curtime
    
    cv2.putText(image, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
    
    cv2.imshow('Hands landmark', image)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
    
capture.release()
cv2.destroyAllWindows()
