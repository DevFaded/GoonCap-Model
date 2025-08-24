import cv2, numpy as np, string
from keras.models import load_model

def solvecap(image_path, model_path="capmodel.h5", min_conf=0.5):
    charsize = 30
    caplen = 5
    charlist = list(string.ascii_lowercase + string.digits)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((2,2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    w = thresh.shape[1] // caplen
    chars = [cv2.resize(thresh[:, i*w:(i+1)*w], (charsize, charsize)) for i in range(caplen)]
    X = np.array(chars).reshape(-1, charsize, charsize, 1) / 255.0
    model = load_model(model_path)
    pred = model.predict(X)
    result = ""
    for p in pred:
        idx = np.argmax(p)
        conf = p[idx]
        if conf < min_conf:
            result += '?'
        else:
            result += charlist[idx]
    return result

result = solvecap("image.png")
print("Predicted CAPTCHA:", result)
