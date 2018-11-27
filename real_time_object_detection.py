# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel --picamera 1 --time 1 --detections 0

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# Collect frames from a video stream
def collectFrames(VideoStream,numberOfFrames):

        count = 0
        frameArray = []
        while count <= 2*numberOfFrames:

                # grab the frame from the threaded video stream and resize it
                # to have a maximum width of 400 pixels, rotate to correct camera angle
                frame = VideoStream.read()
                frame = imutils.resize(frame, width=400)
                frame = imutils.rotate_bound(frame, 270)

                # store frame
                frameArray.append(frame)
                count = count + 1


                # collect frame every 0.1 seconds
                time.sleep(.05)

                # logger
                print('Time: ' + str(time.time()) + ', Frames Collected: ' + str(count))

        return frameArray

def classifyFrame(net, confidenceThreshold, classificationType, frame, CLASSES, COLORS):

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()
        label = ''
        # loop over the detections
        for i in np.arange(0, detections.shape[2]):

                # extract the confidence (i.e., probability) associated with
                # the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if confidence > confidenceThreshold:

                        # extract the index of the class label from the
                        # `detections`, then compute the (x, y)-coordinates of
                        # the bounding box for the object
                        idx = int(detections[0, 0, i, 1])
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        # draw the prediction on the frame
                        label = "{}: {:.2f}%".format(CLASSES[idx],
                                confidence * 100)
                        print('Label: ' + str(label))

                        # If only detecting people
                        if classificationType == 0:
                                if 'person' in label:
                                        cv2.rectangle(frame, (startX, startY), (endX, endY),
                                                (0,255,0), 2)
                                        y = startY - 15 if startY - 15 > 15 else startY + 15
                                        cv2.putText(frame, label, (startX, y),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                        # If detecting everything
                        elif classificationType == 1:
                                        cv2.rectangle(frame, (startX, startY), (endX, endY),
                                                COLORS[idx], 2)
                                        y = startY - 15 if startY - 15 > 15 else startY + 15
                                        cv2.putText(frame, label, (startX, y),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        # let the processor cool to prevent crashes
        time.sleep(.2)

        return [frame,label]

# classified objects on frames
def classifyFrames(net,confidenceThreshold,frameArray,classificationType,CLASSES, COLORS):

        # loop through frames and find detections
        count = 0
        classifiedFrameArray = []

        for frame in frameArray:

                [frame,label] = classifyFrame(net, confidenceThreshold, classificationType, frame, CLASSES, COLORS)

                # store updated frame
                classifiedFrameArray.append(frame)
                count = count + 1
                # logger
                print('Time: ' + str(time.time()) + ', Frames Processed: ' + str(count))

        return classifiedFrameArray


# write video from frames
def writeVideo(frameArray,outputPath):
        # loop through frames and write to video
        frameCount = 0
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        writer = None
        for frame in frameArray:
                # initialize writer
                if writer is None:
                        (h, w) = frame.shape[:2]
                        writer = cv2.VideoWriter(outputPath,fourcc,20,(w, h),True)
                # write frame to video
                writer.write(frame);  
                frameCount = frameCount + 1
                # logger
                print('Time: ' + str(time.time()) + ', Frames Written: ' + str(frameCount))

def initVideoStream(camera):
        # initialize the video stream, allow the cammera sensor to warmup,
        print("[INFO] starting video stream...")
        vs = VideoStream(usePiCamera=camera > 0).start()
        time.sleep(2.0)
        return vs

def closeVideoStream(VideoStream):
        print("[INFO] closing video stream...")
        cv2.destroyAllWindows()
        VideoStream.stop()


def run(net, confidenceThreshold, camera, numberOfFrames, outputPath, classificationType, CLASSES, COLORS, VideoStream):
        frameArray = collectFrames(VideoStream,numberOfFrames)
        classifiedFrameArray = classifyFrames(net, confidenceThreshold, frameArray, classificationType, CLASSES, COLORS)
        writeVideo(classifiedFrameArray,outputPath)   

def idle(net, confidenceThreshold, camera, numberOfFrames, outputPath, classificationType, CLASSES, COLORS):
        VideoStream = initVideoStream(camera)
        while True:

                # collect frame every 0.1 seconds
                time.sleep(.1)

                # grab the frame from the threaded video stream and resize it
                # to have a maximum width of 400 pixels, rotate to correct camera angle
                frame = VideoStream.read()
                frame = imutils.resize(frame, width=400)
                frame = imutils.rotate_bound(frame, 270)

                [frame, label] = classifyFrame(net, confidenceThreshold, classificationType, frame, CLASSES, COLORS)
                print(label)
                if 'person' in label:
                        print('PERSION DETECTED..start')
                        run(net, confidenceThreshold, camera, numberOfFrames, outputPath, classificationType, CLASSES, COLORS, VideoStream)
                        print('PERSION DETECTED..end')

def main():

        # Start Time
        startTime = time.time()

        # construct the argument parse and parse the arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("-p", "--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt file")
        ap.add_argument("-m", "--model", required=True,
                help="path to Caffe pre-trained model")
        ap.add_argument("-c", "--confidence", type=float, default=0.2,
                help="minimum probability to filter weak detections")
        ap.add_argument("-r", "--picamera", type=int, default=1,
                help="whether or not the Raspberry Pi camera should be used")
        ap.add_argument("-t", "--time",type=int, default=3,
                help="time in seconds")
        ap.add_argument("-o", "--outputPath", default='test.avi',
                help="output video file path")
        ap.add_argument("-f", "--detections", type=int, default=0,
                help="0 will only detect people, 1 will detect all the net is capable of")
        args = vars(ap.parse_args())

        prototxt = args["prototxt"]
        model = args["model"]
        confidenceThreshold = args["confidence"]
        camera = args["picamera"]
        timeOfClips = args['time']
        outputPath = args['outputPath']
        classificationType = args["detections"]


        # load our serialized model from disk
        print("[INFO] loading model...")
        net = cv2.dnn.readNetFromCaffe(prototxt, model)

        # initialize the list of class labels MobileNet SSD was trained to
        # detect, then generate a set of bounding box colors for each class
        CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                "sofa", "train", "tvmonitor"]
        COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

        numberOfFrames = timeOfClips/0.1
        
  
        idle(net, confidenceThreshold, camera, numberOfFrames, outputPath, classificationType, CLASSES, COLORS)

        # end time
        endTime = time.time()
        elapsedTime = endTime - startTime

        # logger
        print('Number Of Frames: ' + str(numberOfFrames) + ' frames')
        print('Length Of Video : ' + str(args['time']) + ' seconds')
        print('Path To Video : ' + str(outputPath))
        print('Elapsed Time: ' + str(elapsedTime))

main()

    




