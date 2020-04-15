# USAGE
# python yolo.py --image images/baggage_claim.jpg --yolo yolo-coco

# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os

class yoloCNN():
	
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

		# determine only the *output* layer names that we need from YOLO
		self.ln = self.net.getLayerNames()
		self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

		(self.W, self.H) = (None, None)


	def get_objects(self,image):

		# if the frame dimensions are empty, grab them
		if self.W is None or self.H is None:
			(self.H, self.W) = image.shape[:2]


		blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),	swapRB=True, crop=False)
		self.net.setInput(blob)

		# start = time.time()
		layerOutputs = self.net.forward(self.ln)
		# end = time.time()

		# show timing information on YOLO
		#print("[INFO] YOLO took {:.6f} seconds".format(end - start))

		# initialize our lists of detected bounding boxes, confidences, and class IDs, respectively
		boxes = []
		confidences = []
		classIDs = []

		# loop over each of the layer outputs
		for output in layerOutputs:
			# loop over each of the detections
			for detection in output:
				# extract the class ID and confidence (i.e., probability) of the current object detection
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]

				# filter out weak predictions by ensuring the detected
				# probability is greater than the minimum probability
				if confidence > self.argConfidence:
					# scale the bounding box coordinates back relative to the size of the image, keeping in mind that YOLO actually
					# returns the center (x, y)-coordinates of the bounding box followed by the boxes' width and height
					box = detection[0:4] * np.array([self.W, self.H, self.W, self.H])
					(centerX, centerY, width, height) = box.astype("int")

					# use the center (x, y)-coordinates to derive the top and left corner of the bounding box
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))

					# update our list of bounding box coordinates, confidences, and class IDs
					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)

		# apply non-maxima suppression to suppress weak, overlapping bounding boxes
		idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.argConfidence, self.threshold)


		# TODO: definir uma estrutura de dados para os objetos identificados, dados: Seu ID, posição, classe
		# retornar array desta estrutura de dados.
		# separar a função Draw do metodo de get_objects
		
		if len(idxs) > 0:
			for i in idxs.flatten():

				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])

				color = [int(c) for c in self.COLORS[classIDs[i]]]
				cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
				text = "{}: {:.4f}".format(self.LABELS[classIDs[i]],	confidences[i])
				cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


			# show the output image
			cv2.imshow("Image", image)
			cv2.waitKey(0)



yoloCNN = yoloCNN(yoloPath = "yolo-coco", argConfidence = 0.5, threshold = 0.3)
image = cv2.imread("videos/maePic.png")
yoloCNN.get_objects(image)