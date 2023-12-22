from io import BytesIO
import cv2
import uvicorn
from PIL import Image
from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
import mediapipe as mp
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import time
app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = YOLO("path-of-where-your-model-is-located")

class_names = ['Acne', 'Dark Circle', 'Pigmentation', 'Spots', 'Wrinkle']

def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    while True:
        ret, frame = cap.read()  # Read a frame from the video feed
        if not ret:
            break
        cv2.imshow('Frame', frame)
            # Check for the 'q' key to quit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
        results = face_mesh.process(image_rgb)


        image = Image.fromarray(frame)
        image_path = "frame_image.jpg"
        image.save(image_path)
        MODEL("frame_image.jpg", save=False, conf=0.1)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        cv2.imshow("Face Landmarks", frame)
    cap.release()
    cv2.destroyAllWindows()

def save_image(image):
    image.save("save_image.jpg")
    image_cv = cv2.imread("save_image.jpg")
    image_cv = cv2.resize(image_cv, (640, 640))
    output_image_path = 'resized_image.jpg'
    cv2.imwrite(output_image_path, image_cv)
    return output_image_path


def extract_and_resize_face(input_image_path):
    # Initialize Mediapipe FaceMesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    # Read the input image
    image = cv2.imread(input_image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image with Mediapipe FaceMesh
    results = face_mesh.process(image_rgb)

    # If face(s) detected, extract and resize the face
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get x, y coordinates of the landmarks
            landmarks_x = [point.x for point in face_landmarks.landmark]
            landmarks_y = [point.y for point in face_landmarks.landmark]

            # Get the bounding box around the face
            x_min = min(landmarks_x)
            x_max = max(landmarks_x)
            y_min = min(landmarks_y)
            y_max = max(landmarks_y)

            # Calculate width and height of the bounding box
            width = int((x_max - x_min) * image.shape[1])
            height = int((y_max - y_min) * image.shape[0])

            # Calculate coordinates to crop the face
            x = int(x_min * image.shape[1])
            y = int(y_min * image.shape[0])

            # Crop the face using the calculated coordinates
            cropped_face = image[max(0, y):y+height, max(0, x):x+width]

            # Resize the cropped face to 640x640
            resized_face = cv2.resize(cropped_face, (640, 640))

            # Define the path for the extracted face image
            extracted_face_path = 'extracted_face.jpg'

            # Save the resized face image
            cv2.imwrite(extracted_face_path, resized_face)

            # Release resources
            face_mesh.close()

            # Return the path of the extracted face image
            return extracted_face_path

    # If no face detected or extraction fails, return None
    return None


def read_file_as_image(data) -> Image:
    image = Image.open(BytesIO(data))
    return image


def detect_classes(results):
    hash_map = {}
    list_of_concerns = []
    for i in results[0].boxes.cls.tolist():
        i = int(i)
        if i not in hash_map:
            hash_map[i] = i
            list_of_concerns.append(class_names[i])
    return list_of_concerns


def format_detected_classes(detected_classes):
    classes = ""
    for i in range(len(detected_classes)):
        if i == len(detected_classes) - 1:
            classes += detected_classes[i]
        else:
            classes += detected_classes[i] + ", "
    return classes


@app.websocket("/video_feed")
async def video_feed(websocket: WebSocket):
    await websocket.accept()
    for frame in generate_frames():
        await websocket.send_bytes(frame)


@app.get("/server-status")
async def ping():
    return "The server is up "


@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):

    image = read_file_as_image(await file.read())
    path = save_image(image)
    extracted_face_path = extract_and_resize_face(path)
    if extracted_face_path:
        results = MODEL.predict(extracted_face_path, save=True, conf=0.1)
        detected_classes = detect_classes(results)
        return {
            'class': format_detected_classes(detected_classes)
        }

    else:
        results = MODEL.predict(path, save=True, conf=0.1)
        detected_classes = detect_classes(results)
        return {
            'class':  format_detected_classes(detected_classes)
        }


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
