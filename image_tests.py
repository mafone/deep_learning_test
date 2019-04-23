from PIL import Image #pip install image
import cv2 as cv
import numpy

#im = Image.open("foto.jpg")
im = cv.imread ('foto.jpg')
np_im = numpy.array(im)
print (np_im.shape)


np_im = np_im - 100
new_im = Image.fromarray (np_im)
new_im.save("numpy_altered_sample.png")