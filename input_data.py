import json
import os
import random

import numpy
from PIL import Image

import cv2


def sparse_label(raw_label):
    sparse_indices = numpy.zeros(shape=(50))
    for digit_index in range(0, 5):
        raw_digit = list(raw_label)[digit_index]
        sparse_index = (int(raw_digit)) + (digit_index * 10)
        sparse_indices[sparse_index] = 1
    return sparse_indices


def extract_image(filename):
    def guassian_blur(img, a, b):
        blur = cv2.GaussianBlur(img, (a, b), 0)
        ret, th = otsu_s(blur)
        return ret, th

    def bilatrial_blur(img):
        blur = cv2.bilateralFilter(img, 9, 75, 75)
        ret, th = otsu_s(blur)
        return ret, th

    def otsu_s(img):
        ret, th = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return ret, th

    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret1, th1 = guassian_blur(image, 5, 5)
    ret2, th2 = bilatrial_blur(image)

    image = numpy.array(image).reshape([100, 50, 1])
    """
    depoint = 0
    for x in range(100):
        for y in range(50):
            count = 0
            if image[x][y][0] == 255:
                continue
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx = x + dx
                    ny = y + dy
                    if nx == x and ny == y:
                        continue
                    if nx < 0 or ny < 0 or nx >= 100 or ny >= 50:
                        continue
                    if image[nx][ny][0] != 255:
                        count += 1
            if count == 0:
                image[x][y][0] = 255
                depoint += 1
    """
    return image


class DataSet():

    def __init__(self, path):
        self.path = path
        self.pool = []
        self.load_data()

    def load_data(self):
        data = json.loads(open(os.path.join(self.path, 'data.json')).read())
        for id, x in enumerate(data):
            self.pool.append((
                sparse_label(x[0]),
                extract_image(os.path.join(self.path, x[1]))
            ))

    def next_batch(self, size):
        random.shuffle(self.pool)
        labels = [self.pool[id][0] for id in range(size)]
        images = [self.pool[id][1] for id in range(size)]
        return images, labels


train = DataSet('./train')
test = DataSet('./test')

if __name__ == '__main__':
    print(len(train.pool))
    print(len(test.pool))
