# USAGE
# python yolo.py --image images/baggage_claim.jpg --yolo yolo-coco

# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os

class yoloCNN():
	
	# definir parametros iniciais como:
	# yolo_path, image_path, confidence, threshould

	# COLORS, LABELS

	# image, (H,W) - of image

	# net, layersOutput, idx (NMS)

	# boxes, classesID, confidences

	def __init__(self,yoloPath = "yolo-coco", argConfidence = 0.5, threshold = 0.3):

		self.argConfidence = argConfidence
		self.threshold = threshold

		labelsPath = os.path.sep.join([yoloPath, "coco.names"])
		self.LABELS = open(labelsPath).read().strip().split("\n")

		# initialize a list of colors to represent each possible class label
		np.random.seed(42)
		self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3),	dtype="uint8")

		# derive the paths to the YOLO weights and model configuration
		weightsPath = os.path.sep.join([yoloPath, "yolov3.weights"])
		configPath = os.path.sep.join([yoloPath, "yolov3.cfg"])

		# load our YOLO object detector trained on COCO dataset (80 classes)
		print("[INFO] loading YOLO from disk...")
		self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


##################
	def get_layersOutputs_image(self,image_path = "videos/maePic.png"):

		# load our input image and grab its spatial dimensions
		self.image = cv2.imread(image_path)
		(self.H, self.W) = self.image.shape[:2]

		# determine only the *output* layer names that we need from YOLO
		ln = self.net.getLayerNames()
		ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

		# construct a blob from the input image and then perform a forward
		# pass of the YOLO object detector, giving us our bounding boxes and
		# associated probabilities
		blob = cv2.dnn.blobFromImage(self.image, 1 / 255.0, (416, 416),	swapRB=True, crop=False)
		self.net.setInput(blob)

		start = time.time()
		layerOutputs = self.net.forward(ln)
		end = time.time()

		# show timing information on YOLO
		print("[INFO] YOLO took {:.6f} seconds".format(end - start))

		self.layerOutputs = layerOutputs


###############################

	def get_objects_from_image(self):
		

		# initialize our lists of detected bounding boxes, confidences, and
		# class IDs, respectively
		self.boxes = []
		self.confidences = []
		self.classIDs = []

		# loop over each of the layer outputs
		for output in self.layerOutputs:
			# loop over each of the detections
			for detection in output:
				# extract the class ID and confidence (i.e., probability) of
				# the current object detection
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]

				# filter out weak predictions by ensuring the detected
				# probability is greater than the minimum probability
				if confidence > self.argConfidence:
					# scale the bounding box coordinates back relative to the
					# size of the image, keeping in mind that YOLO actually
					# returns the center (x, y)-coordinates of the bounding
					# box followed by the boxes' width and height
					box = detection[0:4] * np.array([self.W, self.H, self.W, self.H])
					(centerX, centerY, width, height) = box.astype("int")

					# use the center (x, y)-coordinates to derive the top and
					# and left corner of the bounding box
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))

					# update our list of bounding box coordinates, confidences,
					# and class IDs
					self.boxes.append([x, y, int(width), int(height)])
					self.confidences.append(float(confidence))
					self.classIDs.append(classID)

		# apply non-maxima suppression to suppress weak, overlapping bounding
		# boxes
		self.idxs = cv2.dnn.NMSBoxes(self.boxes, self.confidences, self.argConfidence,
			self.threshold)

###############################

	def draw_detected_objects(self):
		# ensure at least one detection exists
		if len(self.idxs) > 0:
			# loop over the indexes we are keeping
			for i in self.idxs.flatten():
				# extract the bounding box coordinates
				(x, y) = (self.boxes[i][0], self.boxes[i][1])
				(w, h) = (self.boxes[i][2], self.boxes[i][3])

				# draw a bounding box rectangle and label on the image
				color = [int(c) for c in self.COLORS[self.classIDs[i]]]
				cv2.rectangle(self.image, (x, y), (x + w, y + h), color, 2)
				text = "{}: {:.4f}".format(self.LABELS[self.classIDs[i]], self.confidences[i])
				cv2.putText(self.image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
					0.5, color, 2)

		# show the output image
		cv2.imshow("Image", self.image)
		cv2.waitKey(0)



yoloCNN = yoloCNN()
yoloCNN.get_layersOutputs_image()
yoloCNN.get_objects_from_image()
yoloCNN.draw_detected_objects()