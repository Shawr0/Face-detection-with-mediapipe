

import cv2 as cv
import mediapipe as mp
import numpy as np
import pygame as pg

#from pydub import AudioSegment
#import winsound as ws

#pygame se pehle above 2 use kiye the, but they were very confusing aur usme independent audio process nhi chal rha tha, so wasnt getting desired result, issliye used pygame, dk about its efficiency but it works as intended.

pg.mixer.init()

## shortforms cuz yes
mpfacedet = mp.tasks.vision.FaceDetector
mpfacedetops = mp.tasks.vision.FaceDetectorOptions
mpfacedetres = mp.tasks.vision.FaceDetectorResult
mpops = mp.tasks.BaseOptions
mpdraw = mp.tasks.vision.drawing_utils


# vid capture ke liye
vid = cv.VideoCapture(0)

#sound ke liye
seg = pg.mixer.Sound("C:\\Users\\Shawr\\Documents\\progs\\opencv\\beep2.mp3")


# used to initialize the 'det' object, isme model path + min confidence (currently 50%) defined hai.
options = mpfacedetops(
    base_options=mpops(
        model_asset_path="C:\\Users\\Shawr\\Documents\\progs\\opencv\\blaze_face_short_range.tflite"
    ),
    min_detection_confidence=0.5,
)

# actual detector jise hum baadme call karenge.
det = mpfacedet.create_from_options(options)

# standard video capturing from here
# --------------------------------------------
while True:
    ret, framed = vid.read()
    if not ret:
        print("bruh")
        break
    frame = cv.flip(framed, 1)
    cv.imshow("pre-hehe", frame)
    # --------------------------------------------- to here

    # In same while loop, converting BGR to RGB cuz mp doesnt like bgr (or ig the model is trained on rgb formats)
    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # converting rgb image to mediapipe image object, which is then passed to the detector.
    mpimg = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    results = det.detect(mpimg)

    # Still in while loop, here we are first checking if a face exists, after which we are running a loop in the list results.detections, which is a list of all the faces present currently in the frame, then for all those images, we define a bounding_box to extract its x & y coords, height & width, after which we draw a rectangle on the current face in the current frame using that data

    #TIL: Since its a if(true) format below, implies results.detections is boolean, but no, .detections attribute is a list of all faces in the results obj, and a non empty list returns a true value aur empty list returns false value if used as boolean value.
    if results.detections:
        for i in results.detections:
            box = i.bounding_box
            x = box.origin_x
            y = box.origin_y
            w = box.width
            h = box.height

            # cv.rectangle(kispe draw karna, kaha pe draw karna, size of dibba, colour of outline, thickness i think)
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            #cv.line(similar to above removing size parameter), added later for the line-ducking thingy.
            cv.line(frame, (0, 200), (int(vid.get(cv.CAP_PROP_FRAME_WIDTH)), 200), (255, 0, 0), 2)

            #neeche waali line mei y>200 likhna pada kyukyi ---> TIL: opencv (aur widely comp graphics as a whole mei) mei origin is located on top left corner not like in cartesian where origin is in bottom left. 
            if(y>200 and not pg.mixer.get_busy()):
                seg.play(loops= 3)
        
    else:
        seg.stop()

        

    # displaying the video output, after we have drawn what we wanted on it
    cv.imshow("hehe", frame)

    # some way to quit the program
    if cv.waitKey(1) == ord("q"):
        break

# jus to be safe, nahi karne se vscode mei toh koi error nahi aa rha tha
vid.release()
cv.destroyAllWindows()
