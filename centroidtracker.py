# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
from yolo.detection import detection


class CentroidTracker():
	def __init__(self, maxDisappeared=50):
		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()

		self.maxDisappeared = maxDisappeared

	def register(self, detection):
		detection.id = self.nextObjectID
		self.objects[self.nextObjectID] = detection
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1

	def deregister(self, objectID):
		del self.objects[objectID]
		del self.disappeared[objectID]
	
	def getList(self):
		# for detection in self.objects.values():
		# 	print("x:{} y:{} w:{} h:{}".format(detection.x,detection.y,detection.x+detection.w,detection.y+detection.h))
		return self.objects.copy()

	def update(self, objects_array):

		if len(objects_array) == 0:
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1

				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)

			return self.objects

		# criando dic de obj identificados
		inputCentroids = [None]*len(objects_array)
		inputObjects = [None]*len(objects_array)
		for (i, obj) in enumerate(objects_array):
			inputCentroids[i] = obj.get_centroid()
			inputObjects[i] = obj

		# nunca rastreou, adiciona os identificados
		if len(self.objects) == 0:
			for obj in objects_array:
				self.register(obj)

		# otherwise, are are currently tracking objects so we need to
		# try to match the input centroids to existing object
		# centroids
		else:
			# grab the set of object IDs and corresponding centroids
			objectIDs = list(self.objects.keys())
			objectCentroids = [None]*len(self.objects)
			
			for (i,obj) in enumerate(self.objects.values()):
				objectCentroids[i] = obj.get_centroid()

			# compute the distance between each pair of object
			# centroids and input centroids, respectively -- our
			# goal will be to match an input centroid to an existing
			# object centroid
			D = dist.cdist(objectCentroids, inputCentroids)

			# in order to perform this matching we must (1) find the
			# smallest value in each row and then (2) sort the row
			# indexes based on their minimum values so that the row
			# with the smallest value as at the *front* of the index
			# list
			rows = D.min(axis=1).argsort()

			# next, we perform a similar process on the columns by
			# finding the smallest value in each column and then
			# sorting using the previously computed row index list
			cols = D.argmin(axis=1)[rows]

			# in order to determine if we need to update, register,
			# or deregister an object we need to keep track of which
			# of the rows and column indexes we have already examined
			usedRows = set()
			usedCols = set()

			# loop over the combination of the (row, column) index
			# tuples
			for (row, col) in zip(rows, cols):
				# if we have already examined either the row or
				# column value before, ignore it
				# val
				if row in usedRows or col in usedCols:
					continue

				# otherwise, grab the object ID for the current row,
				# set its new centroid, and reset the disappeared
				# counter
				objectID = objectIDs[row]
				inputObjects[col].id = objectID
				self.objects[objectID] = inputObjects[col]
				self.disappeared[objectID] = 0

				# indicate that we have examined each of the row and
				# column indexes, respectively
				usedRows.add(row)
				usedCols.add(col)

			# compute both the row and column index we have NOT yet
			# examined
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			# in the event that the number of object centroids is
			# equal or greater than the number of input centroids
			# we need to check and see if some of these objects have
			# potentially disappeared
			if D.shape[0] >= D.shape[1]:
				# loop over the unused row indexes
				for row in unusedRows:
					# grab the object ID for the corresponding row
					# index and increment the disappeared counter
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1

					# check to see if the number of consecutive
					# frames the object has been marked "disappeared"
					# for warrants deregistering the object
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)

			# otherwise, if the number of input centroids is greater
			# than the number of existing object centroids we need to
			# register each new input centroid as a trackable object
			else:
				for col in unusedCols:
					self.register(inputObjects[col])

		# return the set of trackable objects
		return self.objects
	