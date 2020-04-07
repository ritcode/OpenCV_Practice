import cv2
import numpy as np

image = cv2.imread('Capture.PNG')
result = image.copy()
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Remove horizontal lines
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    cv2.drawContours(result, [c], -1, (255,255,255), 5)

# Remove vertical lines
#vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
#remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
#cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#for c in cnts:
 #   cv2.drawContours(result, [c], -1, (255,255,255), 5)

#gray2 = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
#sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
#sharpen = cv2.filter2D(gray2, -1, sharpen_kernel)
#abcd = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#close = cv2.morphologyEx(abcd, cv2.MORPH_CLOSE, kernel, iterations=1)
#result = 255 - close

result1 = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)

mask1=cv2.threshold(result1, 130,255,cv2.THRESH_BINARY_INV)[1]

mask1= cv2.dilate(mask1, np.ones((3,3), np.uint8), iterations=4)
mask1= cv2.morphologyEx(mask1, cv2.MORPH_ERODE, np.ones((3,3), np.uint8), iterations=1)
mask1 = cv2.erode(mask1, np.ones((3,3), np.uint8), iterations=2)

cont= cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

for c in cont:
    (x,y,w,h) = cv2.boundingRect(c)
   # cv2.rectangle(ni,(x+w,y+h), (255,255,255), 2)
    cv2.rectangle(result1, (x,y),(x+w,y+h), (0,255,255), 2)







thresh = cv2.resize(thresh, (800,800))
cv2.imshow('thresh', thresh)
result1= cv2.resize(result1,(800,800))
cv2.imshow('result', result1)
cv2.imwrite('result.png', result1)
cv2.waitKey()
