import cv2

# Estrutura de Dado
class detection(): # obj_detected

	def __init__(self,x,y,w,h,idx,color,category,confidence):
		self.color = color
		self.x = x
		self.y = y
		self.w = w
		self.h = h
		self.area = self.w * self.h
		self.centerX = int(x+w/2)
		self.centerY = int(y+h/2)
		self.id = idx
		self.category = category
		self.confidence = confidence

	def draw(self,image):

		# draw a bounding box rectangle and label on the image
		cv2.rectangle(image, (self.x, self.y), (self.x + self.w, self.y + self.h), self.color, 2)
		text = "i:{} | {}: {:.4f}".format(self.id,self.category, self.confidence)
		cv2.putText(image, text, (self.x, self.y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 2)
		
		# draw in the center of the box and label the coordinates below the box
		cv2.circle(image,(self.centerX,self.centerY),2,self.color)
		centertext1 = "x: {}|y: {}".format(self.x,self.y)
		centertext2 = "w: {}|h: {}".format(self.w,self.h)
		centertext3 = "center: {}|centerY: {}".format(self.centerX,self.centerY)

		cv2.putText(image,centertext1,(self.x, self.y + self.h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 2)
		cv2.putText(image,centertext2,(self.x, self.y + self.h+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 2)
		cv2.putText(image,centertext3,(self.x, self.y + self.h+60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 2)

		return image

	def check_centroid(self,centroid):
		# print("centroid X:{} Y:{}".format(centroid[0],centroid[1]))
		# print("x: {} + w: {} = {}".format(self.x,self.w,self.x +self.w))
		# print("y: {} + h: {} = {}".format(self.y,self.h,self.y +self.h))

		if (self.x <= centroid[0] <= self.x +self.w) and (self.y <= centroid[1] <= self.y +self.h):
			return True
		return False

	def check_category(self,category):
		if self.category == category:
			return True
		return False

	def get_centroid(self):
		return (self.centerX , self.centerY)

	def set_color(self,color):
		self.color = color

	def set_prediction(self,centroid):
		self.centerX = centroid[0]
		self.centerY = centroid[1]