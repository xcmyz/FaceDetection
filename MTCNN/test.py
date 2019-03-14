from model import detect_faces, show_bboxes, PNet, RNet, ONet
from PIL import Image
import numpy as np


def test(filename, save_name, model):
    image = Image.open(filename)
    bounding_boxes, landmarks = detect_faces(image, model)
    show_bboxes(image, bounding_boxes,
                facial_landmarks=landmarks).save(save_name)


if __name__ == "__main__":

    pnet = PNet()
    rnet = RNet()
    onet = ONet()

    model = (pnet, rnet, onet)

    # test("test_1.jpg", "1.jpg", model)
    # test("test_2.jpg", "2.jpg", model)
    # test("test_3.jpg", "3.jpg", model)

    test("test.jpg", "1.jpg", model)
