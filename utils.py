import cv2
import base64
import numpy as np


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def imageToBase64(image):
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.imencode('.jpg', image)[1]
    base64Code = str(base64.b64encode(image))[2:-1]
    return base64Code


def base64ToImage(base64Code):
    img_data = base64.b64decode(base64Code)
    img_array = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img
