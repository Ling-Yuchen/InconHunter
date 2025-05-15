import os
import re
from typing import List
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

from utils import Text, timeit


def text_sentences_recognition(texts: List[Text]) -> List[Text]:
    """Merge separate words into a sentence"""
    changed = True
    while changed:
        changed = False
        temp_set = []
        for text_a in texts:
            merged = False
            for text_b in temp_set:
                if text_a.is_on_same_line(
                        text_b, "h",
                        bias_justify=0.2 * min(text_a.height, text_b.height),
                        bias_gap=2 * max(text_a.word_width, text_b.word_width)
                ):
                    text_b.merge_text(text_a)
                    merged = True
                    changed = True
                    break
            if not merged:
                temp_set.append(text_a)
        texts = temp_set.copy()

    for i, text in enumerate(texts):
        text.id = i
    return texts


def merge_intersected_texts(texts: List[Text]) -> List[Text]:
    """Merge intersected texts (sentences or words)"""
    changed = True
    while changed:
        changed = False
        temp_set = []
        for text_a in texts:
            merged = False
            for text_b in temp_set:
                if text_a.is_intersected(text_b, bias=2):
                    text_b.merge_text(text_a)
                    merged = True
                    changed = True
                    break
            if not merged:
                temp_set.append(text_a)
        texts = temp_set.copy()
    return texts


def text_cvt_orc_format(ocr_result):
    texts = []
    if ocr_result is not None:
        for i, result in enumerate(ocr_result):
            error = False
            x_coordinates = []
            y_coordinates = []
            text_location = result["boundingPoly"]["vertices"]
            content = result["description"]
            for loc in text_location:
                if "x" not in loc or "y" not in loc:
                    error = True
                    break
                x_coordinates.append(loc["x"])
                y_coordinates.append(loc["y"])
            if error: continue
            location = {
                "left": min(x_coordinates),
                "top": min(y_coordinates),
                "right": max(x_coordinates),
                "bottom": max(y_coordinates),
            }
            texts.append(Text(i, content, location))
    return texts


def text_cvt_orc_format_paddle(paddle_result):
    texts = []
    if paddle_result is not None:
        for i, line in enumerate(paddle_result):
            points = np.array(line[0])
            location = {
                "left": int(min(points[:, 0])),
                "top": int(min(points[:, 1])),
                "right": int(max(points[:, 0])),
                "bottom": int(max(points[:, 1])),
            }
            content = line[1][0]
            texts.append(Text(i, content, location))
    return texts


class OCRDetector:
    def __init__(self):
        self.threshold = 0.8
        self.model_folder = Path(__file__).parent.resolve() / "ocr_models"
        self.models = {
            "chinese_cht_mobile_v2.0": PaddleOCR(
                lang="chinese_cht",
                det_model_dir=str(self.model_folder / "ch_ppocr_mobile_v2.0_det_infer"),
                cls_model_dir=str(self.model_folder / "ch_ppocr_mobile_v2.0_cls_infer"),
                rec_model_dir=str(self.model_folder / "chinese_cht_mobile_v2.0_rec_infer"),
                use_gpu=False, total_process_num=os.cpu_count(), use_mp=True, show_log=False),
            "ch_ppocr_mobile_v2.0_xx": PaddleOCR(
                lang="ch",
                det_model_dir=str(self.model_folder / "ch_ppocr_mobile_v2.0_det_infer"),
                cls_model_dir=str(self.model_folder / "ch_ppocr_mobile_v2.0_cls_infer"),
                rec_model_dir=str(self.model_folder / "ch_ppocr_mobile_v2.0_rec_infer"),
                use_gpu=False, total_process_num=os.cpu_count(), use_mp=True, show_log=False),
            "ch_PP-OCRv2_xx": PaddleOCR(
                lang="ch",
                det_model_dir=str(self.model_folder / "ch_PP-OCRv2_det_infer"),
                cls_model_dir=str(self.model_folder / "ch_ppocr_mobile_v2.0_cls_infer"),
                rec_model_dir=str(self.model_folder / "ch_PP-OCRv2_rec_infer"),
                use_gpu=False, total_process_num=os.cpu_count(), use_mp=True, show_log=False),
            "ch_ppocr_server_v2.0_xx": PaddleOCR(
                lang="ch",
                det_model_dir=str(self.model_folder / "ch_ppocr_server_v2.0_det_infer"),
                cls_model_dir=str(self.model_folder / "ch_ppocr_mobile_v2.0_cls_infer"),
                rec_model_dir=str(self.model_folder / "ch_ppocr_server_v2.0_rec_infer"),
                use_gpu=False, total_process_num=os.cpu_count(), use_mp=True, show_log=False)
        }

    def get_model(self, ocr_model: str):
        return self.models.get(ocr_model, self.models["ch_ppocr_server_v2.0_xx"])

    def apply_threshold(self, result):
        return [item for item in result if item[1][1] >= self.threshold]
    
    def detect(self, img_path) -> List[str]:
        img = cv2.imread(img_path)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        model = self.get_model("ch_ppocr_mobile_v2.0_xx")
        result = model.ocr(np.array(img), cls=False)[0]
        result = self.apply_threshold(result)
        texts = text_cvt_orc_format_paddle(result)
        texts = merge_intersected_texts(texts)
        texts = text_sentences_recognition(texts)
        return [t.content for t in texts]

ocr_detector = OCRDetector()


@timeit
def ocr_detect(img_path: str):
    result = []
    inter_result = ocr_detector.detect(img_path)
    for text in inter_result:
       result.extend(re.split(r"\s+", text))
    print(result)
    return result