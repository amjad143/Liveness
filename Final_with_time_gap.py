import cv2
from config import *
import numpy as np
import imutils
import random
import mediapipe as mp
import math

cv2.namedWindow('liveness_detection')
cam = cv2.VideoCapture(0)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh_eye = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Left eyes indices
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# right eyes indices
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

RIGHT_IRIS = [474, 475, 476, 477]
LEFT_IRIS = [469, 470, 471, 472]

L_H_LEFT = [33]
L_H_RIGHT = [133]
R_H_LEFT = [362]
R_H_RIGHT = [263]

def show_image(cam,text,color = (0,0,255)):
    ret, im = cam.read()
    im = imutils.resize(im, width=720)
    cv2.putText(im,text,(10,50),cv2.FONT_HERSHEY_COMPLEX,1,color,2)
    return im

def question_bank(index):
    questions = [
                "turn face up",
                "turn face down",
                "turn face right",
                "turn face left",
                "blink",
                "turn eye left",
                "turn eye right"]
    return questions[index]

def face_orientation():
    result = 0
    success, image = cam.read()

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False

    # Get the result
    results = face_mesh.process(image)

    # To improve performance
    image.flags.writeable = True

    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])

                    # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # See where the user's head tilting
            if y < -10:
                result = "LEFT"
            elif y > 10:
                result = "RIGHT"
            elif x < -10:
                result = "DOWN"
            elif x > 10:
                result = "UP"
    # print(result)
    return result

# landmark detection function
def landmarksDetection(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in
                  results.multi_face_landmarks[0].landmark]
    if draw:
        [cv2.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks
    return mesh_coord

# Euclaidean distance
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
    return distance

def euclidean_distance(point1, point2):
    x1, y1 = point1.ravel()
    x2, y2 = point2.ravel()
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance

# Blinking Ratio
def blinkRatio(img, landmarks, right_indices, left_indices):
    # Right eyes
    # horizontal line
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]

    # LEFT_EYE
    # horizontal line
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # vertical line
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)

    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    reRatio = rhDistance / rvDistance
    leRatio = lhDistance / lvDistance

    ratio = (reRatio + leRatio) / 2
    return ratio

def blink_detection():
    result = 0

    ret, frame = cam.read()
    frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        mesh_coords = landmarksDetection(frame, results, False)
        ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)

        if ratio > 3.6:
            # print("BLINK")
            result = "BLINK"

    return result

def iris_position(iris_center, right_point, left_point):
    center_to_right_dist = euclidean_distance(iris_center, right_point)
    total_distance = euclidean_distance(right_point, left_point)
    ratio = center_to_right_dist / total_distance
    iris_position = ""
    if ratio <= 0.42:
        iris_position = "RIGHT"
    elif ratio > 0.42 and ratio <= 0.57:
        iris_position = "CENTER"
    else:
        iris_position = "LEFT"
    return iris_position, ratio

def eyes_position():
    result = 0
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_h, img_w = frame.shape[:2]
    results = face_mesh_eye.process(rgb_frame)
    if results.multi_face_landmarks:
        mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

        (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
        (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])

        center_left = np.array([l_cx, l_cy], dtype=np.int32)
        center_right = np.array([r_cx, r_cy], dtype=np.int32)

        iris_pos, ratio = iris_position(center_right, mesh_points[R_H_RIGHT], mesh_points[R_H_LEFT][0])

        # print(iris_pos)
        if iris_pos == "CENTER":
            result = 0
        else:
            result = iris_pos

    return result

def challenge_result(question, out_model):
    if question == "turn face up":
        if out_model == 0:
            challenge = "fail"
        elif out_model == "UP":
            challenge = "pass"
        else:
            challenge = "fail"

    elif question == "turn face down":
        if out_model == 0:
            challenge = "fail"
        elif out_model == "DOWN":
            challenge = "pass"
        else:
            challenge = "fail"

    elif question == "turn face right":
        if out_model == 0:
            challenge = "fail"
        elif out_model == "RIGHT":
            challenge = "pass"
        else:
            challenge = "fail"

    elif question == "turn face left":
        if out_model == 0:
            challenge = "fail"
        elif out_model == "LEFT":
            challenge = "pass"
        else:
            challenge = "fail"

    elif question == "blink":
        if out_model == 0:
            challenge = "fail"
        elif out_model == "BLINK":
            challenge = "pass"
        else:
            challenge = "fail"

    elif question == "turn eye left":
        if out_model == 0:
            challenge = "fail"
        elif out_model == "LEFT":
            challenge = "pass"
        else:
            challenge = "fail"

    elif question == "turn eye right":
        if out_model == 0:
            challenge = "fail"
        elif out_model == "RIGHT":
            challenge = "pass"
        else:
            challenge = "fail"

    return challenge

for i_questions in range(0, limit_questions):
    index_question = random.randint(0, 6)
    question = question_bank(index_question)

    im = show_image(cam, question)
    cv2.imshow('liveness_detection', im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    for i_try in range(limit_try):
        ret, im = cam.read()
        im = imutils.resize(im, width=720)
        im = cv2.flip(im, 1)

        if question == "blink":
            out_model = blink_detection()
        elif question == "turn eye left" or question == "turn eye right":
            out_model = eyes_position()
        else:
            out_model = face_orientation()
        challenge_res = challenge_result(question, out_model)

        im = show_image(cam, question)
        cv2.imshow('liveness_detection', im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if challenge_res == "pass":
            counter_timer = 0
            while counter_timer <= counter_time_gap:
                im = show_image(cam, question + " : ok")
                cv2.imshow('liveness_detection', im)
                counter_timer += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                   break

            counter_ok_consecutives += 1
            if counter_ok_consecutives == limit_consecutives:
                counter_ok_questions += 1
                counter_try = 0
                counter_ok_consecutives = 0
                break
            else:
                continue

        elif challenge_res == "fail":
            counter_try += 1
            show_image(cam, question + " : fail")
        elif i_try == limit_try - 1:
            break

    if counter_ok_questions == limit_questions:
        while True:
            im = show_image(cam, "LIVENESS SUCCESSFUL", color=(0, 255, 0))
            cv2.imshow('liveness_detection', im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    elif i_try == limit_try - 1:
        while True:
            im = show_image(cam, "LIVENESS FAILED")
            cv2.imshow('liveness_detection', im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        break
