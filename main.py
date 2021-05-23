# Importing the libraries
import cv2 as cv
import HandTrackingModule as HTM
import time
import pandas as pd
import matplotlib.pyplot as plt

# Global Variable
TEXT_POS = (30, 30)
FONT = cv.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
COLOR = (0, 255, 255)
THICKNESS = 2
LINE_TYPE = cv.LINE_AA
MOVES = ["Stone", "Paper", "Scissor"]
TIMER = 5
WIN_SCORE = 5
USER1_SCORE = 0
USER1_CHOICE = ""
USER2_SCORE = 0
USER2_CHOICE = ""

res = ""
final_text = ""
final_result = []
tip = [8, 12, 16, 20]

# Access the Laptop Camera
cap1 = cv.VideoCapture(0)
# Access the External Camera attached with your system
cap2 = cv.VideoCapture(1)


detector1 = HTM.HandTrackingModule()
detector2 = HTM.HandTrackingModule()


# Calculate Result
def win(u1_choice, u2_choice):
    if len(u1_choice) > 0 and len(u2_choice) > 0 and u1_choice != "Invalid Symbol" and u2_choice != "Invalid Symbol":
        if u1_choice == u2_choice:
            return "Draw"
        if (u1_choice == MOVES[0] and u2_choice == MOVES[1]) \
                or (u1_choice == MOVES[1] and u2_choice == MOVES[2]) \
                or (u1_choice == MOVES[2] and u2_choice == MOVES[0]):
            return "Win"
        return "Lose"
    return "User Not Ready"


# Show Hand Gesture Text
def render_text(image, text):
    cv.putText(image, text, TEXT_POS, FONT, FONT_SCALE, COLOR, THICKNESS, LINE_TYPE)


# Show Counter
def show_counter(image, text, ct):
    cv.rectangle(image, (390, 10), (620, 100), (0, 0, 0), -1, cv.LINE_AA)
    cv.putText(image, text,
               (400, 40), FONT, FONT_SCALE / 1.5, COLOR, 1, LINE_TYPE)
    cv.putText(image, f"{TIMER - ct} seconds",
               (400, 80), FONT, FONT_SCALE * 1.25, COLOR, 1, LINE_TYPE)


# Show User Score
def show_score(image, text):
    cv.putText(image, text, (200, 30), FONT, FONT_SCALE / 2, COLOR, 1, LINE_TYPE)


# Logic to detect Stone, Paper & Scissor
def landmark_detection(landmarks_list, img):
    temp = ""
    if landmarks_list != 0:
        finger_arr = []
        flag = False
        for i in tip:
            try:
                if landmarks_list[i][2] < landmarks_list[i - 2][2]:
                    finger_arr.append(1)
                else:
                    finger_arr.append(0)
            except IndexError:
                flag = True
                break
        if not flag:
            ones = finger_arr.count(1)
            zeros = finger_arr.count(0)
            if ones == 4:
                render_text(img, MOVES[1])
                temp = MOVES[1]
            elif zeros == 4:
                render_text(img, MOVES[0])
                temp = MOVES[0]
            elif ones == 3 or ones == 1:
                render_text(img, "Invalid Symbol")
                temp = "Invalid Symbol"
            elif ones == 2:
                if finger_arr[0] == 1 and finger_arr[1] == 1:
                    render_text(img, "Scissor")
                    temp = MOVES[2]
                else:
                    render_text(img, "Invalid Symbol")
                    temp = "Invalid Symbol"
    return temp


start_time = time.time()

while True:
    ret1, img1 = cap1.read()
    ret2, img2 = cap2.read()
    # Detect Hand
    img1 = detector1.find_hands(img1)
    img2 = detector2.find_hands(img2)
    # Detect Landmark on Hand
    landmarks_list1 = detector1.find_position(img1, draw=False)
    landmarks_list2 = detector2.find_position(img2, draw=False)

    USER1_CHOICE = landmark_detection(landmarks_list1, img1)
    USER2_CHOICE = landmark_detection(landmarks_list2, img2)

    if int(time.time() - start_time) == TIMER:
        if USER1_CHOICE in MOVES:
            res = win(USER2_CHOICE, USER1_CHOICE)
        if res == "Win":
            USER1_SCORE += 1
        elif res == "Lose":
            USER2_SCORE += 1
        u1_win_status = ""
        u2_win_status = ""
        if res == "Draw":
            u1_win_status = "Draw"
            u2_win_status = "Draw"
        elif res == "Win":
            u1_win_status = "Win"
            u2_win_status = "Lose"
        else:
            u1_win_status = "Lose"
            u2_win_status = "Win"

        if res != "User Not Ready" and len(USER1_CHOICE) > 0 and len(
                USER2_CHOICE) > 0 and USER1_CHOICE != "Invalid Symbol" and USER2_CHOICE != "Invalid Symbol":
            final_result.append([USER1_CHOICE, USER2_CHOICE, u1_win_status, u2_win_status, USER1_SCORE, USER2_SCORE])
        start_time = time.time()
    else:
        ct1 = int(time.time() - start_time)
        show_counter(img1, "Select Choice in", ct1)
        show_counter(img2, "Select Choice in", ct1)

    show_score(img1, f"User1:{USER1_SCORE} User2:{USER2_SCORE}")
    show_score(img2, f"User1:{USER1_SCORE} User2:{USER2_SCORE}")

    cv.imshow("User1 Screen", img1)
    cv.imshow("User2 Screen", img2)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break
    if USER2_SCORE == WIN_SCORE or USER1_SCORE == WIN_SCORE:
        break

cap1.release()
cap2.release()
cv.destroyAllWindows()

if USER1_SCORE == WIN_SCORE:
    final_text = "USER1 Won :)"
else:
    final_text = "USER2 Won :)"

# Final Result
fig, ax = plt.subplots(1, 1)

column_labels = ["USER1 Choice", "USER2 Choice", "USER1 Result", "USER2 Result", "USER1 Score", "USER2 Score"]
df = pd.DataFrame(final_result, columns=column_labels)
ax.axis('tight')
ax.axis('off')
ax.table(cellText=df.values, colLabels=df.columns, loc="center")
plt.title(final_text)
plt.text(-0.03, 0.04, f"USER1 Score:{USER1_SCORE} USER2 Score:{USER2_SCORE}", fontsize=12)
plt.show()
