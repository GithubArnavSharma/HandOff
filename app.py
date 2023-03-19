from flask import Flask, render_template, jsonify, Response
from pygame import mixer
import cv2
from cvzone.HandTrackingModule import HandDetector
import datetime
import time

app = Flask(__name__)

trained_data = cv2.CascadeClassifier('frontal-data.xml')

detector = HandDetector(maxHands=2, detectionCon=0.725)


camera = cv2.VideoCapture(0)

mixer.init()
mixer.music.load('ding.mp3')

def point_in_face(pointx, pointy, rectx, recty, rectw, recth):
    if (pointx > rectx and pointx < rectx+rectw and pointy > recty and pointy < recty+recth): return True
   

timesTouched = 0
isInFace = None

def gen_frames():  # generate frame by frame from camera
    global timesTouched  


    while True:
        ret, img = camera.read()

        if not ret or img is None: break
        else:
            


            greyscale_frame = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgRGB = cv2.blur(imgRGB, (2,2))

            face_cordinates = trained_data.detectMultiScale(greyscale_frame)
            rectx, recty, rectw, recth = -1, -1, -1, -1

            for (x,y,w,h) in face_cordinates:
                if (w > 100 and h > 100):
                    rectx, recty, rectw, recth = x, y, w, h
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)        

            hands = detector.findHands(img, draw=False)

            if len(hands) > 0:
                hand = hands[0]
                if len(hand['lmList']) == 0:
                    hand = hands[1]

                fingerList = hand["lmList"]
                
                isInFace = False
        
                for x, y, _ in fingerList:
                    cv2.circle(img, (x, y), 15, (139, 0, 0), cv2.FILLED)
                    if rectx != -1 and point_in_face(x, y, rectx, recty, rectw, recth):
                        isInFace = True

                if isInFace:
                  
                    mixer.music.play()
                    time.sleep(0.25)
                    mixer.music.stop()
              
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
