
import cv2
import numpy as np
capture_video = cv2.VideoCapture(0)
fourcc=cv2.VideoWriter_fourcc(*"XVID")
out=cv2.VideoWriter('out3.avi',fourcc,20.0,(640,480))

time.sleep(1)
count = 0
background = 0
for i in range(60):
    return_val, background = capture_video.read()
    if return_val == False:
        continue
while (capture_video.isOpened()):
    return_val, img = capture_video.read()
    if not return_val:
        break
    count = count + 1
    img = np.flip(img, axis=1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 120, 70])
    upper = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower, upper)
    mask2=cv2.inRange(hsv,np.array([170,120,70]),np.array([180,255,255])
    mask1 = mask1 + mask2

    # Refining the mask corresponding to the detected red color
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3),
                                                            np.uint8), iterations=2)
    mask1 = cv2.dilate(mask1, np.ones((3, 3), np.uint8), iterations=1)
    mask2 = cv2.bitwise_not(mask1)

    # Generating the final output
    res1 = cv2.bitwise_and(background, background, mask=mask1)
    res2 = cv2.bitwise_and(img, img, mask=mask2)
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)
    out.write(final_output)
   
    cv2.imshow("frame", final_output)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

out.release()
capture_video.release()
cv2.destroyAllWindows()