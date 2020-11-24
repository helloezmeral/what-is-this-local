# wget https://github.com/helloezmeral/what-is-this-local/blob/main/model_unquant.tflite?raw=true -O model_unquant.tflite
# wget https://github.com/helloezmeral/what-is-this-local/blob/main/labels.txt?raw=true -O labels.txt

import numpy as np
import tensorflow as tf

import cv2
from PIL import Image, ImageOps

import time

np.set_printoptions(suppress=True)

def jpgResize(image, w, h):
    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (w, h)
    return ImageOps.fit(image, size, Image.ANTIALIAS)

def run_prediction(in_image_array):

    # Normalize the image
    normalized_image_array = (in_image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    input_data[0] = normalized_image_array

    # run the inference
    input_shape = input_details[0]['shape']
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data, labels[np.argmax(output_data)]

label_file = "./labels.txt"
labels = []
with open(label_file, "r") as f:
        labels = [line.strip().split(' ',1) for line in f]

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

cap = cv2.VideoCapture('http://172.16.11.10:8080/stream.mjpeg')
thisTime = time.time()
cvText = ""
while True:
    ret, frame = cap.read()
    cap.set(cv2.CAP_PROP_FPS, 30)
    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)  
    cv2.resizeWindow("Video", 720, 720)  
    cv2.putText(frame, cvText, (0, 100), cv2.FONT_HERSHEY_PLAIN, 5.5, (0,255,0), 5) # must be on top of imshow
    cv2.imshow('Video', frame)


    image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    image = jpgResize(image, 224, 224)
    # Drop some of the framerate
    if time.time() - thisTime > 0.5:
        thisTime = time.time()
        prediction , label = run_prediction(np.asarray(image))
        print("Prediction: ", label[1])
        cvText = label[1]

    if cv2.waitKey(1) == 27:
        exit(0)
