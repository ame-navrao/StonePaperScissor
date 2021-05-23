import cv2 as cv
import mediapipe as mp


class HandTrackingModule():
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5, max_hands=2):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.max_hands = max_hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(False, 2, min_detection_confidence, min_tracking_confidence)

    def find_hands(self, img, draw=True):
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        imgRGB = cv.cvtColor(cv.flip(img, 1), cv.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        imgRGB.flags.writeable = False
        self.results = self.hands.process(imgRGB)
        # Draw the hand annotations on the image.
        imgRGB.flags.writeable = True
        imgBGR = cv.cvtColor(imgRGB, cv.COLOR_RGB2BGR)
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(imgBGR, hand_landmarks,
                                                   self.mp_hands.HAND_CONNECTIONS)
        return imgBGR

    def find_position(self, img, hand_no=0, draw=True):
        lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(my_hand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lm_list.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 15, (255, 0, 255), cv.FILLED)

        return lm_list



# cap = cv.VideoCapture(0)
# detector = HandTrackingModule()
#
#
# while True:
#     success, img = cap.read()
#     img = detector.find_hands(img)
#     lmList = detector.find_position(img)
#     if len(lmList) != 0:
#         pass
#         # print(lmList[4])
#
#     cv.imshow("Image", img)
#     if cv.waitKey(1) & 0xFF == ord("q"):
#         break
# cap.release()
# cv.destroyAllWindows()

