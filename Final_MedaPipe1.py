import cv2
import mediapipe as mp
import time
import numpy as np
from livenessmodel import get_liveness_model

cap = cv2.VideoCapture(0)
pTime = 0

model = get_liveness_model()
font = cv2.FONT_HERSHEY_DUPLEX

# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")
input_vid = []

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)
real_face = 0

while True:
    sucess, img = cap.read()

    # To convert input BGR image to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    # print(results)

    # To visualize the key point we are using in built
    if results.detections:
        for id, detection in enumerate(results.detections):
            # blurAmount = cv2.Laplacian(img, cv2.CV_64F).var()
            liveimg = cv2.resize(img, (100, 100))
            liveimg = cv2.cvtColor(liveimg, cv2.COLOR_BGR2GRAY)
            blurAmount = cv2.Laplacian(liveimg, -1).var()
            input_vid.append(liveimg)
            inp = np.array([input_vid[-24:]])
            inp = inp / 255
            if inp.size == 240000:
                inp = inp.reshape(1, 24, 100, 100, 1)
                pred = model.predict(inp)
                livenessVal = pred[0][0]
            else:
                livenessVal = 0.96
            bboxc = detection.location_data.relative_bounding_box

            h, w, c = img.shape
            bbox = int(bboxc.xmin * w), int(bboxc.ymin * h), \
                   int(bboxc.width * w), int(bboxc.height * h)

            cv2.putText(img, str(livenessVal), (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 0), 1)

            if livenessVal > 0.95:
                if blurAmount > 50:
                    real_face += 1
                    if real_face >= 10:
                        cv2.rectangle(img, bbox, (255, 0, 255), 2)
                        # print("Real Face")
                        cv2.putText(img, "Real Face", (100, 75), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                                    (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 1,
                                    (255, 0, 255), 2)
                        real_face = 0
            else:
                cv2.putText(img, "WARNING: SPOOF DETECTED", (100, 75), font, 1.0, (0, 0, 255), 2)

            # To get the detection score
            cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                        (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 1,
                        (255, 0, 255), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img,f'FPS: {int(fps)}',(28,70),cv2.FONT_HERSHEY_PLAIN,2,(0,255,255),2)
    cv2.imshow("image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.VideoCapture(0).release()
cv2.destroyAllWindows()