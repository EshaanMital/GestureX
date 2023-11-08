import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import tkinter as tk
from tkinter import Text, Label, Frame, PhotoImage
from tkinter import ttk
from PIL import Image, ImageTk

# Create a Tkinter window
window = tk.Tk()
window.title("Hand Gesture Recognition")

# Create a frame for the video feed
video_frame = Frame(window, bd=2, relief=tk.SUNKEN)
video_frame.grid(row=0, column=0, padx=10, pady=10, rowspan=3)

# Open the camera
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
          "W", "X", "Y", "Z"]

detected_letters = []  # List to store detected letters
start_time = None
delay = 8  # 8 seconds delay

# Create a label for displaying the video feed within the video frame
video_label = Label(video_frame)
video_label.pack()

# Function to update the detected letters display in the GUI
def update_detected_letters():
    detected_string = ''.join(detected_letters)
    detected_letters_text.config(state=tk.NORMAL)
    detected_letters_text.delete("1.0", "end")
    detected_letters_text.insert("1.0", "Detected Letters:\n" + detected_string)
    detected_letters_text.config(state=tk.DISABLED)

# Function to update the timer display in the GUI
def update_timer():
    if start_time is not None:
        current_time = time.time()
        elapsed_time = current_time - start_time
        remaining_time = max(0, delay - elapsed_time)
        timer_label.config(text=f"Time Left: {remaining_time:.1f} seconds")
        window.after(100, update_timer)

# Create a Text widget to display the detected letters
detected_letters_text = Text(window, height=10, width=30)
detected_letters_text.grid(row=0, column=1, padx=10, pady=10)
update_detected_letters()  # Initialize the detected letters display
detected_letters_text.config(state=tk.DISABLED)

# Create a label for displaying the timer
timer_label = Label(window, text="Time Left: 8.0 seconds")
timer_label.grid(row=1, column=1, padx=10, pady=10)

# Main loop
def main_loop():
    global start_time
    while True:
        success, img = cap.read()
        img = cv2.resize(img, (640, 480))

        hands, img = detector.findHands(img)

        if start_time is not None:
            current_time = time.time()
            elapsed_time = current_time - start_time
            remaining_time = max(0, delay - elapsed_time)
            timer_label.config(text=f"Time Left: {remaining_time:.1f} seconds")

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:  # Check if imgCrop is valid
                imgCropShape = imgCrop.shape

                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)

                current_time = time.time()

                if start_time is None:
                    start_time = current_time

                if current_time - start_time >= delay:
                    detected_letter = labels[index]
                    detected_letters.append(detected_letter)
                    start_time = None

                update_detected_letters()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 480))
        photo = ImageTk.PhotoImage(image=Image.fromarray(img))

        video_label.configure(image=photo)
        video_label.image = photo
        window.update()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Detected Letters:", detected_letters)
    cv2.destroyAllWindows()
    cap.release()

# Create a Start button to begin gesture recognition
start_button = ttk.Button(window, text="Start Recognition", command=main_loop)
start_button.grid(row=2, column=1, padx=10, pady=10)

# Create an Exit button to close the application
exit_button = ttk.Button(window, text="Exit", command=window.destroy)
exit_button.grid(row=3, column=1, padx=10, pady=10)

# Update the timer display
update_timer()

window.mainloop()
