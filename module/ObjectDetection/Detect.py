import numpy as np
import cv2 as cv

CLASS_FILE = "module\\ObjectDetection\\models\\coco.names"
WEIGHT_FILE = "module\\ObjectDetection\\models\yolov3.weights"
CONFIG_FILE = "module\\ObjectDetection\\models\yolov3.cfg"
CONFIDENCE_THRESOLD = 0.5
IMAGE_FILE = "D:\Wallpaper\object-detection.jpg"
LABELS = open(CLASS_FILE).read().strip().splitlines()
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")


class CrowdCounting:

    def __init__(self) -> None:
        self.yolo = cv.dnn.readNet(WEIGHT_FILE, CONFIG_FILE)
        self.yolo.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        self.yolo.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

    def get_output_layers(self):
        layer_names = self.yolo.getLayerNames()
        output_layers = [layer_names[i-1] for i in self.yolo.getUnconnectedOutLayers()]
        return output_layers

    def loadFrameIntoModel(self):
        blob = cv.dnn.blobFromImage(self.frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        self.output = self.yolo.forward(self.get_output_layers())

    def computeScore(self):
        boxes = []
        confidences = []
        classIDs = []
        for out in self.output:
            for detection in out:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > CONFIDENCE_THRESOLD:
                    box = detection[0:4] * \
                    np.array([self.W, self.H, self.W, self.H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        return boxes,confidences,classIDs
    
    def drawBox(self,boxes,confidences,classIDs):
        idxs = cv.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESOLD,CONFIDENCE_THRESOLD)
        
        self.people = 0
        self.car = 0
        self.bike = 0
        self.cow = 0
        
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                if "person" in LABELS[classIDs[i]]:
            # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    self.people += 1
                    color = [int(c) for c in COLORS[classIDs[i]]]

                    # cv.rectangle(self.frame, (x, y), (x + w, y + h), color, 2)
                    
                if "car" in LABELS[classIDs[i]]:
            # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    self.car += 1
                    color = [int(c) for c in COLORS[classIDs[i]]]

                    # cv.rectangle(self.frame, (x, y), (x + w, y + h), color, 2)
                    
                if "motorbike" in LABELS[classIDs[i]]:
            # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    self.bike += 1
                    color = [int(c) for c in COLORS[classIDs[i]]]

                    # cv.rectangle(self.frame, (x, y), (x + w, y + h), color, 2)
                    
                if "cow" in LABELS[classIDs[i]]:
            # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    self.cow += 1
                    color = [int(c) for c in COLORS[classIDs[i]]]

                    # cv.rectangle(self.frame, (x, y), (x + w, y + h), color, 2)
                    
                    # text = "{}: {}".format(LABELS[classIDs[i]], people)
                
                    # cv.putText(self.frame, text, (50, 50 ), cv.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
                # print(self.people)
           
            
        
                    
    def countPeople(self,frame):
        self.frame = frame
        # self.frame = cv.resize(frame, (512, 512),3)
        (self.H,self.W) = self.frame.shape[:2]
        self.loadFrameIntoModel()
        box,confidence,classid = self.computeScore()
        self.drawBox(box,confidence,classid)
        font = cv.FONT_HERSHEY_SIMPLEX
        context = f"People : {self.people} | Car : {self.car} | Bike : {self.bike} | Cow : {self.cow} "
        print(context)
        cv.putText(self.frame, context , (10,450), font, 1, (0, 255, 0), 1, cv.LINE_AA)
        return self.frame
    
    def previewFrame(self):
        cv.imshow("Frame",self.frame)
        # cv.waitKey(10000) 
        
if __name__ == '__main__' :
    
    cc = CrowdCounting()
    # img = cv.imread(IMAGE_FILE)
    # cc.countPeople(img)
    # cc.previewFrame()
    # vid = cv.VideoCapture("D:\\Production\\Dataset\\Dataset_Problem_4\\Crowd\\Sample_file_vehicle5.mp4")
    # vid = cv.VideoCapture("D:\\Wallpaper\\videoplayback.mp4")
    vid = cv.VideoCapture(0)
    # vid = cv.VideoCapture("D:\\Images\\videoplayback.mp4")
    if vid is None:
        print("Omk")
        exit()
    while(vid.isOpened()):
      
        ret, frame = vid.read()
        frame = cv.resize(frame, (1080, 720),3)
        cc.countPeople(frame)
        cc.previewFrame()
        # cv.imshow('frame', frame)
    
        if cv.waitKey(1) & 0xFF == ord('q'):
            break   
        
    vid.release()
    cv.destroyAllWindows()