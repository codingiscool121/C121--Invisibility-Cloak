import cv2 as cv
import numpy as np
import time

videocode = cv.VideoWriter_fourcc(*'XVID')
outputfile = cv.VideoWriter("video.avi",videocode,20.0,(640,480))
capture= cv.VideoCapture(0)

time.sleep(2)
bg = 0
#Capturing background image before video to represent the background. 60 frames per second.
for i in range(0,60):
    ret,bg = capture.read()

#Flipping the image so it's not a mirror image
bg= np.flip(bg, axis= 1)

#While camera is open, it keeps capturing images and writing to the file.
while(capture.isOpened()):
    ret, image = capture.read()
    if not ret:
        break
    #Every time an image is captrued, it is flipped using np.flip().
    image = np.flip(image, axis=1)
   #Converting it into a hsv colored image, so it can be filtered easier. Before, it was in BGR.
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    #Filter for finding the color red in the image. The color that we filter depends on the number we choose here. Each color has it's unique range of HSV values.
    lower_red = np.array([30,120,50])
    upper_red = np.array([50,255,255])

    #Filtering the image to find the red color.
    mask1 = cv.inRange(hsv, lower_red, upper_red)
    lower_red = np.array([150, 120, 170])
    upper_red = np.array([180,255,255])
    #Filtering the image to find the red color. There is a second mask, as there are different ranges of red. To be more accurate, we have two masks. Mask 2 is not necessary, but is added anyway.
    mask2 = cv.inRange(hsv, lower_red, upper_red)
    #Merging the two masks, so we can find the red color.
    mask1=mask1+mask2
    #Dilating the image to make the red color larger.
    mask1 = cv.morphologyEx(mask1, cv.MORPH_OPEN, np.ones((3,3),np.uint8))
    mask1 = cv.morphologyEx(mask1, cv.MORPH_DILATE, np.ones((3,3),np.uint8))
    #Finding where we don't have red, we will give the actual image.
    mask2 = cv.bitwise_not(mask1)
    #We want to find the objects that are not red, so we will give the actual image.
    image1 = cv.bitwise_and(image,image,mask=mask2)
    #We want to find the objects that are red, so we will give the background image. 
    image2 = cv.bitwise_and(bg, bg, mask=mask1)
    #Merging two images together.
    finaloutput = cv.addWeighted(image1,1,image2,1,0)
    #Writing the final image to the file.
    outputfile.write(finaloutput)
    #Displaying the final image.
    cv.imshow('frame',finaloutput)
    #Waiting for the user to press the 1 key to stop the program.
    cv.waitKey(1)

#Capture.release= turns off camera
#cv.destroyAllWindows= closes the camera window
capture.release()
cv.destroyAllWindows()        