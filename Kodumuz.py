from keras.models import load_model
import cv2
import numpy as np
import time  # Eklediğimiz zaman modülü

np.set_printoptions(suppress=True)

model = load_model("<Keras modelin dosya yolu>", compile=False)
class_names = open("<labels.txt'in dosya yolu>", "r").readlines()

camera = cv2.VideoCapture(0)

while True:
    ret, image = camera.read()
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    cv2.imshow("Webcam Image", image)

    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1

    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

        

    keyboard_input = cv2.waitKey(1)

    if keyboard_input == ord("q"):
        break


    # 5 saniyede bir döngüyü beklet
    time.sleep(0.3)


camera.release()
cv2.destroyAllWindows()

# Civan Çelebi
