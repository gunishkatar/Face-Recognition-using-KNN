import numpy as np
import cv2
import os
from urllib2 import urlopen
from KNN import knn 
from sklearn.neighbors import KNeighborsClassifier 

# Initialize camera
url = 'http://192.168.0.102:8080/shot.jpg'

# Load the haar cascade for frontal face
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')

# Address or Location of the Data
dataset_path = 'path of the data'

face_data = []
labels = []
class_id = 0
names = {}

# Dataset prepration
for fx in os.listdir(dataset_path):
	if fx.endswith('.npy'):
		data_item = np.load(dataset_path + fx)
		face_data.append(data_item)
		target = class_id * np.ones((data_item.shape[0],))
		names[class_id] = fx[:len(fx)-4]
		class_id += 1
		labels.append(target)

face_dataset = np.concatenate(face_data, axis=0)

face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))
print face_labels.shape
print face_dataset.shape

trainset = np.concatenate((face_dataset, face_labels), axis=1)
print trainset.shape
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
	imageResp = urlopen(url)
	imageNp = np.array(bytearray(imageResp.read()), dtype=np.uint8)
	frame = cv2.imdecode(imageNp, -1)

	# Convert frame to grayscale
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect multi faces in the image
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	for face in faces:
		x, y, w, h = face

		# Get the face ROI
		offset = 7
		face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
		face_section = cv2.resize(face_section, (100, 100))

		out = knn(trainset, face_section.flatten())

		# Draw rectangle in the original image
		cv2.putText(frame, names[int(out)],(x,y-10), font, 1,(255,0,0),2, lineType = cv2.CV_AA)
		cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

	cv2.imshow("Faces", frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cv2.destroyAllWindows()