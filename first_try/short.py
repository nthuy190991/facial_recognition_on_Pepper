from binascii import b2a_hex, a2b_hex
import numpy as np
import cv2
import sys

def read_data():
    fr = open("raw_data_write_from_wb.txt", "rb")
    data_read = fr.read()
    fr.close()
    return data_read

try:
    v = sys.argv[1] # Verbose mode
    v = int(v)
except IndexError:
    print "python short.py <verbose_flag>"
    v = 0

data_read = read_data()

#if (v==1): print len(data_read)
#
#data = np.empty((len(data_read)))
#dataint = np.zeros((len(data_read),1))
#data_bin = b2a_hex(str(data_read))
#binary_data= a2b_hex(data_bin)

data8uint = np.fromstring(data_read, np.uint8)
#frameFromHTML = cv2.imdecode(data8uint, cv2.IMREAD_COLOR)
#for k in range(0,len(data_read)):
#    data[k] = int(data_bin[2*k:2*k+2], 16)

#print data
#if (v==1): print dataint[0:10]
#if (v==1): print int(data[0:1])
#if (v==1): print '\n------------------\n'
##
data_reshape = np.reshape(data8uint, (3,160,120), order='F')
imgR = data_reshape[0]
imgG = data_reshape[1]
imgB = data_reshape[2]

imgRGB = np.dstack((data_reshape[2].T,data_reshape[1].T,data_reshape[0].T))

img  = imgRGB.astype(int)
#img2 = cv2.imdecode(data, cv2.IMREAD_COLOR)

#if (v==1): print 'img=',img


cv2.imwrite('output_py.jpg', img)
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
matr = cv2.imread('output_py.jpg')
if (v==1): print matr

if (v==1): print 'img=', img

#cv2.imshow('output py', img2)
cv2.imshow('output matlab',matr)

cv2.waitKey(0)
cv2.destroyAllWindows()
