from threading import Thread
from time import sleep
import cv2 as cv
from module.FaceRecognition.Detect import runScanner
import sys
from module.ObjectDetection.Detect import CrowdCounting
from module.GarbageDetection.Detect import GarbageDetector

class TMS:
    def __init__(self) -> None:
        self.frame = None
        self.running = True
        self.cc = CrowdCounting()
        pass
    
    def runCamera(self):
        
        cam = cv.VideoCapture(0)
        # cam = cv.VideoCapture("D:\\Development\\Projects\\M.L\\TMS\\sample\\college2.mp4")
        while cam.isOpened():
            success,frame = cam.read()
            if success:
                self.frame = frame
            if self.running == False:
                exit()
        cam.release()
        cv.destroyAllWindows()
        self.frame = None
        pass

    
    def detectObject(self):
        self.frame = self.cc.countPeople(self.frame)
        pass
    
    def detectGarbage(self):
        hasGarbage, self.framme = GarbageDetector(self.frame)
        if hasGarbage:
            print("Garbage detected")
        pass
    
    def run(self):
        while self.frame is not None:
            # self.detectObject()
            # self.scanFace()
            self.detectGarbage()
            cv.imshow('Scanner',self.frame)
            
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            
            
        self.running = False            
    
    def scanFace(self):
            self.frame = runScanner(self.frame)
            # frame = cv.resize(frame,(520,520))
            
    
import tensorflow as tf

if __name__ == '__main__':
    print(tf.__version__)

    # print('1: ', tf.config.list_physical_devices('GPU'))
    # print('2: ', tf.test.is_built_with_cuda)
    # print('3: ', tf.test.gpu_device_name())
    # print('4: ', tf.config.get_visible_devices())

    tms = TMS()
    thread1 = Thread(target = tms.runCamera )
    thread1.start()
    sleep(5)
    tms.run()
    # thread2 = Thread(target = tms.detectObject )
    # thread2.start()
