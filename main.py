import io

from fastapi import FastAPI, File
import re

import cv2
import numpy as np
import pyocr
from PIL import Image

app = FastAPI()

# OCRエンジンを取得
engines = pyocr.get_available_tools()
engine = engines[0]

# 対応言語取得
langs = engine.get_available_languages()
print("対応言語:", langs)

pattern = re.compile(r"[0-9]{6}")

def detect_number(img, card_luminance_percentage):
    a = luminance_threshold(img, card_luminance_percentage)
    _, img = cv2.threshold(img, a, 255, cv2.THRESH_BINARY)
    img = img[900:1000, 650:img.shape[1] - 50]

    txt = engine.image_to_string(Image.fromarray(img), lang="eng")
    return txt


# TODO: パフォーマンス度外視
# 0.2~0.3くらい
def luminance_threshold(gray_img, card_luminance_percentage):
    """
    グレースケールでの値(輝度と呼ぶ)が `x` 以上のポイントの数が20%を超えるような最大のxを計算する
    ただし、 `100 <= x <= 200` とする
    """
    number_threshold = gray_img.size * card_luminance_percentage
    flat = gray_img.flatten()
    # 200 -> 100
    for diff_luminance in range(100):
        if np.count_nonzero(flat > 200 - diff_luminance) >= number_threshold:
            return 200 - diff_luminance
    return 100


class CardDetector:
    def __init__(self, img):
        self.threshold = 100
        self.gray_img = img

    def detect(self):
        _, binarized = cv2.threshold(self.gray_img, self.threshold, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binarized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 面積最大のものを選択
        card_cnt = max(contours, key=cv2.contourArea)

        # 輪郭を凸形で近似
        # 輪郭の全長に固定値で 0.1 の係数をかけるので十分
        # ある程度まともにカードを写す前提では係数のチューニングはほぼ不要と思われる(OCRの調整では必要かも)
        epsilon = 0.05 * cv2.arcLength(card_cnt, True)
        approx = cv2.approxPolyDP(card_cnt, epsilon, True)

        # カードの横幅(画像がカードが縦になっているので、射影変換の際にはwidthとheightが逆になっている)
        card_img_width = 1000  # 適当な値
        card_img_height = card_img_width  # 免許証のration(=nanacoのratio)で割って産出

        src = np.float32(list(map(lambda x: x[0], approx)))
        if src.shape[0] != 4:
            return None

        dst = np.float32([[0, 0], [0, card_img_width], [card_img_height, card_img_width], [card_img_height, 0]])

        project_matrix = cv2.getPerspectiveTransform(src, dst)

        transformed = cv2.warpPerspective(self.gray_img, project_matrix, (card_img_height, card_img_width))
        return transformed


@app.post("/")
async def root(file: bytes = File(...)):
    img = np.array(Image.open(io.BytesIO(file)),np.uint8)
    detector = CardDetector(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    for i in range(4):
        detector.threshold = luminance_threshold(detector.gray_img, 0.1 + i * 0.1)
        transformed = detector.detect()
        if transformed is not None:
            break
    else:
        return {"error": 0}

    number = ""
    # 向き判定
    for _ in range(4):
        for i in range(3):
            number = detect_number(transformed, 0.95 + i * 0.01).replace("O", "0").replace(" ", "").replace("'", "")
            if pattern.match(number):
                break
        if pattern.match(number):
            break
        transformed = cv2.rotate(transformed, cv2.ROTATE_90_CLOCKWISE)
    else:
        return {"error": 1}

    return {"number": number}
