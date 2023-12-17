import cv2
import numpy as np
import time

np.set_printoptions(suppress=True)

# USB kamera objesini başlatın
camera = cv2.VideoCapture(0)  # 0, 1, 2, ... şeklinde değiştirerek farklı kameraları deneyebilirsiniz

# Model ve etiket dosyalarını Raspberry Pi'nin dosya sistemine uygun şekilde belirtin
model_path = "/path/to/your/keras_model.h5"
labels_path = "/path/to/your/labels.txt"

model = load_model(model_path, compile=False)
class_names = open(labels_path, "r").readlines()

while True:
    ret, image = camera.read()
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    cv2.imshow("USB Camera Image", image)

    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1

    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    keyboard_input = cv2.waitKey(1)

    # q tuşuna basıldığında döngüden çık
    if keyboard_input == ord("q"):
        break

    # 0.3 saniyede bir döngüyü beklet
    time.sleep(0.3)

camera.release()
cv2.destroyAllWindows()
