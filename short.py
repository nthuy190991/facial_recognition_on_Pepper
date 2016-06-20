from binascii import b2a_hex
import numpy as np
import cv2
import sys
from PIL import Image

#try:
#    v = sys.argv[1] # Verbose mode
#    v = int(v)
#except IndexError:
#    print "python short.py <verbose_flag>"
#    v = 0

v=1

fr = open("raw_data_write_from_wb.txt", "rb")
data_read = fr.read()
fr.close()

if (v==1): print len(data_read)

data = np.zeros((len(data_read),1))

for k in range(0,len(data_read)):
    data[k] = int(b2a_hex(data_read[k]), 16)

if (v==1): print int(data[0:1])
if (v==1): print '\n------------------\n'
#
data_reshape = np.reshape(data, (3,160,120), order='F')
imgR = data_reshape[0]
imgG = data_reshape[1]
imgB = data_reshape[2]

imgRGB = np.dstack((imgB,imgG,imgR))
if (v==1): print imgRGB

cv2.imwrite('output_py.png', imgRGB)
#
# if (v==1): print data_reshape
# if (v==1): print imgRGB
# # img = np.zeros((160,120,3))
# # img = data_reshape[0]
# # if (v==1): print len(img[0])
# # if (v==1): print img
# #cv2.imwrite('output_py.jpg', img)
if (v==1): print '\n------------------\n'
#
# #
matr = cv2.imread('output_py.png')
if (v==1): print matr

cv2.imshow('output py',imgRGB)
cv2.imshow('output png',matr)

cv2.waitKey(5000)
cv2.destroyAllWindows()
