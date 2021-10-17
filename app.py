
from tensorflow import keras
from keras.models import load_model
from tensorflow.keras.applications.vgg16 import  decode_predictions
import time
import cv2
import numpy as np

model=keras.model.load_model('model.h5')
#model = load_model('model.h5')


def prediction(frame):
    img=cv2.resize(frame,(224,224))
    img=img.reshape(1,224,224,3)

    y_pred = model.predict(img)
    print(decode_predictions(y_pred,top=1))


cap=cv2.VideoCapture(0)

while True:

    ret,frame=cap.read()
    prediction(frame)
    time.sleep(0.01)

    cv2.imshow("hello",frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break


cv2.destroyAllWindows()

