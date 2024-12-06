import cv2
import pickle
import numpy as np
import os

if not os.path.exists('data/'):
    os.makedirs('data/')

# File to store used names
used_names_file = 'data/used_names.pkl'

# Initialize used names file if it doesn't exist
if not os.path.exists(used_names_file):
    with open(used_names_file, 'wb') as f:
        pickle.dump(set(), f)

# Validate Aadhar number input
while True:
    name = input("Enter your 12-digit Aadhar number: ")
    if len(name) == 12 and name.isdigit():
        with open(used_names_file, 'rb') as f:
            used_names = pickle.load(f)
        if name in used_names:
            print("This Aadhar number has already been used. Please use a different one.")
        else:
            break
    else:
        print("Invalid input. Please enter a valid 12-digit numeric Aadhar number.")

age = int(input("Enter your age: "))

# Check if age is greater than 18
if age < 18:
    print("You must be older than 18 to Vote.")
else:
    # Add the entered name to the set of used names
    with open(used_names_file, 'rb') as f:
        used_names = pickle.load(f)
    used_names.add(name)
    with open(used_names_file, 'wb') as f:
        pickle.dump(used_names, f)

    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces_data = []

    i = 0
    framesTotal = 51
    captureAfterFrame = 2

    while True:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            crop_img = frame[y:y + h, x:x + w]
            resized_img = cv2.resize(crop_img, (50, 50))
            if len(faces_data) <= framesTotal and i % captureAfterFrame == 0:
                faces_data.append(resized_img)
            i = i + 1
            # Increase the font scale and thickness for the text
            cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (50, 50, 255), 3)

# Increase the rectangle thickness and make the rectangle larger
            padding = 10  # Adding some padding to make the rectangle bigger
            cv2.rectangle(frame, (x - padding, y - padding), (x + w + padding, y + h + padding), (50, 50, 255), 3)


        cv2.imshow('frame', frame)
        k = cv2.waitKey(1)
        if k == ord('q') or len(faces_data) >= framesTotal:
            break

    video.release()
    cv2.destroyAllWindows()

    faces_data = np.asarray(faces_data)
    faces_data = faces_data.reshape((framesTotal, -1))
    print(faces_data)

    if 'names.pkl' not in os.listdir('data/'):
        names = [name] * framesTotal
        with open('data/names.pkl', 'wb') as f:
            pickle.dump(names, f)
    else:
        with open('data/names.pkl', 'rb') as f:
            names = pickle.load(f)
        names = names + [name] * framesTotal
        with open('data/names.pkl', 'wb') as f:
            pickle.dump(names, f)

    if 'faces_data.pkl' not in os.listdir('data/'):
        with open('data/faces_data.pkl', 'wb') as f:
            pickle.dump(faces_data, f)
    else:
        with open('data/faces_data.pkl', 'rb') as f:
            faces = pickle.load(f)
        faces = np.append(faces, faces_data, axis=0)
        with open('data/faces_data.pkl', 'wb') as f:
            pickle.dump(faces, f)