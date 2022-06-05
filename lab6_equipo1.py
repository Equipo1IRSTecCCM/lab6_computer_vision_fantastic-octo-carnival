'''
TE3002B Implementación de robótica inteligente
Equipo 1
    Diego Reyna Reyes A01657387
    Samantha Barrón Martínez A01652135
    Jorge Antonio Hoyo García A01658142
Laboratory 06
Ciudad de México, 05/06/2022
'''
import numpy as np
import cv2
#Load Image 'objects01.jpg OR 'objects02.jpg (either one of them works)
img = cv2.imread("objects02.jpg")
#Convert the image to Gray Scale
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Crop the image in 9 objects (they are ordered in a 3x3 form in the original image) and add them to a list using two nested for loops
w = img_gray.shape[0]
h = img_gray.shape[1]
size = [int(w/3),int(h/3)]
cropped_img = []
for i in range(3):
    for j in range(3):
        cropped_img.append(img_gray[i*size[0]:(i+1)*size[0],j*size[1]:(j+1)*size[1]])
#Create a list with the different names of each image
names = ['liney','1897','ribbony','stargazey','tappy','sealy','roundy','wheaty','niney']
#Create an Orb detector with 1000 key-points and a scaling pyramid factor of 1.2
orb = cv2.ORB_create(1000, 1.2)
#Create a matcher with Hamming norm and crossCheck equal true
bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
#For all images detect all the key points and descriptors using an orb and append the descriptors to a list
descriptors = []
keypoints = []
for im in cropped_img:
    kp, des = orb.detectAndCompute(im, None)
    
    descriptors.append(des)
    keypoints.append(kp)
#Open an infinite video stream
cap= cv2.VideoCapture(0)
#Use an infinite loop that would stop with an enter key
while True:
    #Capture frame and convert to Gray Scale
    ret, frame= cap.read() 
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if cv2.waitKey(1) & 0xFF == 27 or not ret: #End conditions
        break
    kp, des = orb.detectAndCompute(frame_gray, None)
    #Compare the detectors from the captured frame with the detectors from the image
    matches = []
    num_mat = []
    for desn in descriptors:
        mat = bf.match(des, desn)
        matches.append(sorted(mat, key = lambda x:x.distance))
        num_mat.append(len(matches[-1]))
    #Select the detector with the largest number of matches
    most_mat = num_mat.index(max(num_mat))
    #Print the matches of the image with the largest number of matches
    #Print the number the matches at the bottom of the image
    cv2.putText(frame, "Matches: "+str(num_mat[most_mat]), (20,frame.shape[0]-20), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
    #Only if the number of matches is above 200 print the label corresponding to the image with the largest number of matches at the top of the video frame. Otherwise, if the number of matches is below 200 print nothing
    if num_mat[most_mat] >= 200:
        cv2.putText(frame, names[most_mat], (20,20), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
    cv2.imshow('frame',frame)
cv2.destroyAllWindows()