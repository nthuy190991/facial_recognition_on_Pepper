import time
import sys
import qi
import numpy as np
import cv2
from threading import Thread
from binascii import b2a_hex


def rotateImage(img):
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2,rows/2), -90, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def display_video():
    time.sleep(0.1)
    while True:
        if (data!=None):
            t0 = time.time()
            _data = data
            width, height, nbLayers = _data[0:3] # from Documentation
            image_data = np.zeros((len(_data[6]),1)) # data[6]: array of height*width*nbLayes containing image data
            data_bin = b2a_hex(str(_data[6]))
            for k in range(0,len(_data[6])):
                #image_data[k] = int(b2a_hex(str(_data[6])[k]), 16)
                image_data[k] = int(data_bin[2*k:2*k+2], 16)
            image_reshape = np.reshape(image_data, (nbLayers, width, height), order='F')
            imgRGB = np.dstack((image_reshape[2],image_reshape[1],image_reshape[0]))
            cv2.imwrite('output.png', imgRGB)
            frame = cv2.imread('output.png')
            frame_dst = rotateImage(frame)
            key = cv2.waitKey(1)
            if (key==27):
                break
            cv2.imshow('Output', frame_dst)
    cv2.destroyAllWindows()

class App(object):
    def __init__(self):

        self.ip = "10.69.128.84"
        self.port = 9559

        self.session = qi.Session()

        try:
            self.session.connect("tcp://" + self.ip + ":" + str(self.port))
        except RuntimeError:
            print ( "Can't connect to Naoqi at ip \"" + self.ip + "\" on port " + str(self.port) +".\n"
                    "Please check your script arguments. Run with -h option for help.")
            sys.exit(1)

        # self.ALVideoDevice = self.session.service('ALVideoDevice')

        # self.ALVideoDevice.unsubscribe("CameraTop_0")
        # self.ALVideoDevice.setParameter(0, 14, 1)
        # self.handle = self.ALVideoDevice.subscribeCamera("CameraTop", 0, 1, 11, 5)

        self.ALTabletService = self.session.service("ALTabletService")


    def run(self):
        # global data

        # thread_video = Thread(target=display_video)
        # thread_video.start()
        # while True:
        #     self.ALVideoDevice.releaseImage(self.handle)
        #     #print(self.ALVideoDevice.getImageRemote(self.handle))
        #     data = self.ALVideoDevice.getImageRemote(self.handle)
            
        self.show_images()

    def show_images(self):
        self.ALTabletService.showImage('http://192.18.0.1/img/milky-way-galaxy-wallpaper-hd.jpg')
        time.sleep(30)
        # Hide the web view
        self.ALTabletService.hideImage()

data = None

app = App()
app.run()
