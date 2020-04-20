import cv2

# Estrutura de Dado
class detection(): # obj_detected

	def __init__(self,x,y,w,h,id,color,category,confidence):
		self.color = color
		self.x = x
		self.y = y
		self.w = w
		self.h = h
		self.avgX = int(x+w/2)
		self.avgY = int(y+h/2)
		self.id = id #not in use
		self.category = category
		self.confidence = confidence

	def draw(self,image):

		cv2.rectangle(image, (self.x, self.y), (self.x + self.w, self.y + self.h), self.color, 2)
		text = "{}: {:.4f}".format(self.category, self.confidence)
		cv2.putText(image, text, (self.x, self.y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 2)

		return image

	def check_centroid(self,centroid):
		if (self.x <= centroid[0] <= self.x +self.w) and (self.y <= centroid[0] <= self.y +self.h):
			return True
		return False

	def check_category(self,alvo):
		if self.category == alvo.category:
			return True
		return False

	# def get_centroid(self):
	# 	return (self.avgX , self.avgY)

	def set_color(self,color):
		self.color = color

	def set_prediction(self,centroid):
		self.avgX = centroid[0]
		self.avgY = centroid[1]