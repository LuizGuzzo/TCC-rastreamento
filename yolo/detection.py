import cv2

# Estrutura de Dado
class detection():

	def __init__(self,x,y,w,h,id,color,category,confidence):
		self.color = color
		self.x = x
		self.y = y
		self.w = w
		self.h = h
		self.id = id #not in use
		self.category = category
		self.confidence = confidence

	def draw(self,image):

		cv2.rectangle(image, (self.x, self.y), (self.x + self.w, self.y + self.h), self.color, 2)
		text = "{}: {:.4f}".format(self.category, self.confidence)
		cv2.putText(image, text, (self.x, self.y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 2)

		return image
