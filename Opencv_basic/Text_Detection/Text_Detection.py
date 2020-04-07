import cv2
import numpy as np

image = cv2.imread(r"AARsummary_65_1.png")

image_copy = image.copy()
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
image_copy_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
_,thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
hk = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,hk, iterations=2)
conts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
conts = conts[0] if len(conts) == 2 else conts[1]
for c in conts:
    cv2.drawContours(image_copy, [c], -1, (255,255,255), 5)


result1 = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

mask1=cv2.threshold(result1, 130,255,cv2.THRESH_BINARY_INV)[1]

mask1= cv2.dilate(mask1, np.ones((3,3), np.uint8), iterations=4)
mask1= cv2.morphologyEx(mask1, cv2.MORPH_ERODE, np.ones((3,3), np.uint8), iterations=1)
mask1 = cv2.erode(mask1, np.ones((3,3), np.uint8), iterations=2)

cont= cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

for c in cont:
    (x,y,w,h) = cv2.boundingRect(c)
    cv2.rectangle(image, (x,y),(x+w,y+h), (255,0,0),2)
image= cv2.resize(image,(800,800))

cv2.imshow('result', image)
cv2.imwrite("output.png",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
