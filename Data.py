import numpy as np
import cv2
from urllib2 import urlopen

# Initialize camera
url = 'http://192.168.0.102:8080/shot.jpg'



# Load the haar cascade for frontal face
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')

skip = 0
face_data = []
dataset_path = '/home/gunish/Desktop/Perceptron/Class 4(20 june 2018)/Face_recognition_using_KNN/Data/'

file_name = raw_input("Enter the name of the person: ")

while True:
	imageResp = urlopen(url)
	imageNp = np.array(bytearray(imageResp.read()), dtype=np.uint8)
	frame = cv2.imdecode(imageNp, -1)

	# Convert frame to grayscale
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect multi faces in the image
	faces = face_cascade.detectMultiScale(gray, 1.3, 5) #Detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles.   # Returns rows each containing 4 columns viz x, y, w, h
	k = 1

	faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)

	# update the frame number
	skip += 1

	for face in faces[:1]:  # Gives 1st row from Faces' List
		x, y, w, h = face

		# Get the face ROI
		offset = 7
		face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
		face_section = cv2.resize(face_section, (100, 100))
		
		if skip % 10 == 0:
			face_data.append(face_section)
			print len(face_data)

		# Display the face ROI
		cv2.imshow(str(k), face_section)  # Open Small Window Showing the frames captured
		k += 1

		# Draw rectangle in the original image
		cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

	cv2.imshow("Faces", frame)  # Opens Larger Windows Which shows the output of the Camera

	if cv2.waitKey(1) & 0xFF == ord('q'):   # tells cv to close window when user presses the  key 'q'
		break

# Convert face list to numpy array
face_data = np.asarray(face_data) # Convert Data of Faces from list to numpy Array
face_data = face_data.reshape((face_data.shape[0], -1))
print face_data.shape

# Save the dataset in filesystem
np.save(dataset_path + file_name, face_data)
print "Dataset saved at: {}".format(dataset_path + file_name + '.npy')

cv2.destroyAllWindows()	