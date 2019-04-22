import numpy as np #for arrays and objetos complex objects management
import cv2 as cv #lib for detection e 


if __name__ == '__main__':
    face_classifier = cv.CascadeClassifier ('haarcascade_frontalface_default.xml') #Select classifier
    image = cv.imread ('image.jpg')
    image_gray = cv.cvtColor (image, cv.COLOR_BGR2GRAY) #Convert to grey scale -> Many OpenCV methods require that the image be in this color space.
    #Use the classifier in the chosen image so that the face is detected and draw a rectangle on it for marking.

    #x, y, w and h are the coordinates in the Cartesian plane (x, y).
    #The drawing method of the rectangle asks for these 4 variables, where we will draw (image), the color and thickness of the traces.
    faces = face_classifier.detectMultiScale (image_gray, 1.3, 5)
    #for (x, y, w, h) in faces:
    #    cv.rectangle (image, (x,y), (x+w,y+h), (255, 0, 0), 2)

    #cv.imshow ('imagem', image)
    #cv.waitKey (0)
    #cv.destroyAllWindows ()
    #print ("Test\n")