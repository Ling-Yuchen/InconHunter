from typing import List

from llm import query
from logger import logger
from ocr_detect import ocr_detect
from utils import timeit, load_reports, download_img_from_url, dataset_base


class RuleEngine:

    def __init__(self):

        self.prompt1 = """You are a professional assistant reviewing crowdsourced test reports.
You will be given a description of a test issue.
Check whether the issue provides enough contextual information and then determine whether the issue's external manifestation (e.g., visual errors, misalignment, missing elements, empty results, or elements in wrong order) can be observed in the screenshot.
Note that:
  1. Issue with enough contextual information should contain (1) clear statement of a bug/typo, and (2) the triggering operation(s).
  2. If the issue does not provide enough contextual information, you should return false anyway.
  3. Even if the root cause of the issue (e.g., logic failure) cannot be directly confirmed, if its visual symptoms can be visible in the screenshot, return true.
  4. For issues related to "user interaction failures" (e.g., clicking on buttons not triggering actions), if the failure cannot be concluded from no visual clue, return false.
  5. Issues related to performance or over long waiting time can be reflected by blank screen or dimming screen, return true.
  6. If the issue mentions specific text or phrase appearing in the screenshot, return true.
  7. For issues related to "disorganized/disordered content or layout", return true.
  8. If the issue mentions that certain messages appear, return true.
  9. If the issue mentions that certain expected element does not appear, return true.
Return a brief JSON response: {result: true/false, reason: <one-short-sentence explanation>}"""

        self.prompt2 = """You are a professional assistant reviewing crowdsourced test reports.
You will be given a description of a test issue.
Check if the described issue is **directly** and **completely** reflected in text-based UI components (e.g., error messages, labels, or other text elements) in the app's GUI. The text should **explicitly state or describe** the issue without requiring inference or interpretation.
Note that:
  1. If the issue involves text or character overlap, always return `false`, as this type of issue cannot be described by valid text snippets.
  2. If the target UI component only contains one single character or sign ("!", "?", "$", ...), you should consider it as a special case and return `false`.
  3. If the issue describes a typo problem (a specific text different from expected), return `true`.
Return a brief JSON response: {result: true/false, reason: <one-short-sentence explanation>}"""

        self.prompt3 = """You are a professional assistant reviewing crowdsourced test reports.
You will be given a description of a test issue and a set of OCR detection result of the screenshot.
Check if the OCR detection result contains information that aligns the issue description.
Note that:
  1. For issues involving "comparison between different state of the app", static OCR result is not enough to describe and you should return false.
  2. For issues involving "expected content v.s. actual content", you should ignore the first rule in note.
  3. For issues involving "missing element" and the OCR result does not contain the expected element, you should return true.
Return a brief JSON response: {result: true/false, reason: <one-short-sentence explanation>}"""

        self.prompt4 = """You are a professional assistant reviewing crowdsourced test reports.
You will be given a description of a test issue and a corresponding screenshot.
Check if the app GUI in screenshot is consistent with the described issue (i.e., the screenshot **visually and precisely** reflects (part of) the issue).
Note that:
  1. For performance issues, if blank/black screen or screen dimming or unrendered content is provided, you should return true.
  2. Do not focus on the triggering operations or appearance frequency in the issue, focus on the bug behavior itself.
  3. For issues involving "missing content in a certain widget", you should first locate the corresponding widget and then check if there are suspicious blanks.
  4. For issues involving "page disorder or confusion", you should check if there is content masking, content overlapping, image rendering, layout disorganization, etc.
  5. For issues involving "comparison between different state of the app", one screenshot is not enough to describe and you should return false.
  6. For issues involving "continuous loading", you should carefully check if there is a loading icon or progress bar (maybe not obvious).
Return a brief JSON response: {result: true/false, reason: <one-short-sentence explanation>}"""

        self.prompt5 = """You are a professional assistant reviewing crowdsourced test reports.
You will be given a description of a test issue and text snippets extracted from the corresponding screenshot.
Check if the extracted text snippets contain information that **directly** and **completely** confirm the described issue.
Note that:
  1. For crash issues, if the text snippets indicate that the app has already stopped or crashed, you should return true.
Return a brief JSON response: {result: true/false, reason: <one-short-sentence explanation>}"""

        self.prompt6 = """You are a professional assistant reviewing crowdsourced test reports.
You will be given a set of OCR detection result of the screenshot, which represents the state of a mobile app.
Analyze the provided text and generate a concise, one-sentence description of the app's state or activity.
The description should be clear and directly reflect the context of the app based on the extracted text.
Return a brief JSON response: {description: <description>}"""

        self.prompt7 = """You are a professional assistant reviewing crowdsourced test reports.
You will be given a description of a test issue in an app and a screenshot representing the app's state.
Check whether the issue provides enough contextual information and then determine whether the described issue contextually aligns the app state.
Note that:
  1. Issue with enough contextual information should contain (1) clear statement of a bug/typo, and (2) the triggering operation(s).
  2. Contextually alignment means that (1) the described triggering operation(s) can lead to the provided app state, and (2) the app state can directly lead to the reported issue **in exactly one step** based on the context provided, i.e., the issue should occur immediately after performing only one action in the described app state, not as a result of multiple steps or sequences of actions.
  3. If the issue does not provide enough contextual information, you should return false anyway.
  4. Do not focus on whether the bug is visually observable in the GUI, but rather on whether the app's state aligns with the context that could trigger the issue **in only one action**.
  5. There's no need for evidence or indication that the issue will definitely be triggered, just consider the possibility. For example, if the state shows available sharing options, you should anticipate that sharing failure may happen.
  6. The issue description may mention multiple steps for reproduction, but some of them may have been performed to reach the provided state.
Return a brief JSON response: {result: true/false, reason: <one-short-sentence explanation>}"""

        self.prompt8 = """You are a professional assistant reviewing crowdsourced test reports.
You will be given a description of a test issue and a description of an app state.
Check whether the issue provides enough contextual information and then determine whether the described issue contextually aligns the app state.
Note that:
  1. Issue with enough contextual information should contain (1) clear statement of a bug/typo, and (2) the triggering operation(s).
  2. Contextually alignment means that (1) the described triggering operation(s) can lead to the provided app state, and (2) the app state can directly lead to the reported issue **in exactly one step** based on the context provided, i.e., the issue should occur immediately after performing only one action in the described app state, not as a result of multiple steps or sequences of actions.
  3. If the issue does not provide enough contextual information, you should return false anyway.
  4. Do not focus on whether the bug is visually observable in the GUI, but rather on whether the app's state aligns with the context that could trigger the issue **in only one action**.
  5. There's no need for evidence or indication that the issue will definitely be triggered, just consider the possibility. For example, if the state shows available sharing options, you should anticipate that sharing failure may happen.
  6. The issue description may mention multiple steps for reproduction, but some of them may have been performed to reach the provided state.
Return a brief JSON response: {result: true/false, reason: <one-short-sentence explanation>}"""

        self.prompt9 = """You are a professional assistant reviewing crowdsourced test reports.
You will be given a description of a test issue and a corresponding screenshot.
Check if the app GUI in screenshot is consistent with the described issue.
Return a brief JSON response: {result: true/false, reason: <reason for your judgment>}"""

    @timeit
    def visible_in_screenshot(self, report_txt: str) -> bool:
        """区分报告的bug是否存在显式的界面表现"""
        result, in_use, out_use, in_cost, out_cost = \
            query(user_msg_txt=report_txt, system_msg=self.prompt1, model="gpt-4o-mini")
        logger.info(result)
        logger.info(f"Input token: {in_use} (${in_cost:6f}); Output token: {out_use} (${out_cost:6f})")
        return result["result"]

    @timeit
    def verify_invisible_in_screenshot(self, report_txt: str, screen_txt: List[str]) -> bool:
        """根据OCR识别结果验证报告的bug是否存在显式的界面表现"""
        prompt = f"Description: {report_txt}\nText Snippets: [{','.join(t for t in screen_txt)}]"
        result, in_use, out_use, in_cost, out_cost = \
            query(user_msg_txt=prompt, system_msg=self.prompt5, model="gpt-4o-mini")
        logger.info(result)
        logger.info(f"Input token: {in_use} (${in_cost:6f}); Output token: {out_use} (${out_cost:6f})")
        return not result["result"]

    @timeit
    def direct_reflect_from_ui_text(self, report_txt: str) -> bool:
        """判断是否能仅通过文本语义匹配来确认一致性"""
        result, in_use, out_use, in_cost, out_cost = \
            query(user_msg_txt=report_txt, system_msg=self.prompt2, model="gpt-4o-mini")
        logger.info(result)
        logger.info(f"Input token: {in_use} (${in_cost:6f}); Output token: {out_use} (${out_cost:6f})")
        return result["result"]

    @timeit
    def detect_consistency_by_ui_text(self, report_txt: str, candidates: List[str]) -> bool:
        """使用OCR识别结果进行文本语义匹配来检测一致性"""
        prompt = f"Description: {report_txt}\nOCR Result: [{','.join(t for t in candidates)}]"
        result, in_use, out_use, in_cost, out_cost = \
            query(user_msg_txt=prompt, system_msg=self.prompt3, model="gpt-4o-mini")
        logger.info(result)
        logger.info(f"Input token: {in_use} (${in_cost:6f}); Output token: {out_use} (${out_cost:6f})")
        return result["result"]

    @timeit
    def detect_consistency_by_vision(self, report_txt: str, report_img: str) -> bool:
        """使用MLLM直接进行bug的显式界面特征匹配来检测一致性"""
        result, in_use, out_use, in_cost, out_cost = \
            query(user_msg_txt=report_txt, user_msg_img=report_img, system_msg=self.prompt4, model="gpt-4o")
        logger.info(result)
        logger.info(f"Input token: {in_use} (${in_cost:6f}); Output token: {out_use} (${out_cost:6f})")
        return result["result"]

    @timeit
    def describe_app_state_by_ocr_result(self, screen_txt: List[str]) -> str:
        """根据OCR识别结果来描述截图所展示的页面状态"""
        prompt = f"[{','.join(t for t in screen_txt)}]"
        result, in_use, out_use, in_cost, out_cost = \
            query(user_msg_txt=prompt, system_msg=self.prompt6, model="gpt-4o-mini")
        logger.info(result)
        logger.info(f"Input token: {in_use} (${in_cost:6f}); Output token: {out_use} (${out_cost:6f})")
        return result["description"]

    @timeit
    def detect_consistency_by_visual_state(self, report_txt: str, report_img: str) -> str:
        """根据截图所展示的页面状态来检测一致性"""
        result, in_use, out_use, in_cost, out_cost = \
            query(user_msg_txt=report_txt, user_msg_img=report_img, system_msg=self.prompt7, model="gpt-4o")
        logger.info(result)
        logger.info(f"Input token: {in_use} (${in_cost:6f}); Output token: {out_use} (${out_cost:6f})")
        return result["result"]

    @timeit
    def detect_consistency_by_textual_state(self, report_txt: str, description: str) -> bool:
        """根据页面状态的文本描述来检测一致性"""
        prompt = f"Test Issue: {report_txt}\nApp State: {description}"
        result, in_use, out_use, in_cost, out_cost = \
            query(user_msg_txt=prompt, system_msg=self.prompt8, model="gpt-4o-mini")
        logger.info(result)
        logger.info(f"Input token: {in_use} (${in_cost:6f}); Output token: {out_use} (${out_cost:6f})")
        return result["result"]

    @timeit
    def preprocess(self, report_txt: str) -> bool:
        result, in_use, out_use, in_cost, out_cost = \
            query(user_msg_txt=report_txt, system_msg=self.prompt0, model="gpt-4o-mini")
        logger.info(result)
        logger.info(f"Input token: {in_use} (${in_cost:6f}); Output token: {out_use} (${out_cost:6f})")
        return result["result"]

    @timeit
    def run(self, report_id: str, report_txt: str, report_img: str):
        visible = self.visible_in_screenshot(report_txt)
        if visible:
            text_exist = self.direct_reflect_from_ui_text(report_txt)
            if text_exist:
                candidates = ocr_detect(report_img)
                consistent = self.detect_consistency_by_ui_text(report_txt, candidates)
            else:
                consistent = self.detect_consistency_by_vision(report_txt, report_img)
        else:
            screen_txt = ocr_detect(report_img)
            invisible = self.verify_invisible_in_screenshot(report_txt, screen_txt)
            if not invisible:
                consistent = True
            else:
                description = self.describe_app_state_by_ocr_result(screen_txt)
                consistent = self.detect_consistency_by_textual_state(report_txt, description)
                if not consistent:
                    consistent = self.detect_consistency_by_visual_state(report_txt, report_img)
        logger.info(f"Report #{report_id} Consistent? {consistent}")
        return consistent

    @timeit
    def run_without_check_visibility(self, report_id: str, report_txt: str, report_img: str):
        text_exist = self.direct_reflect_from_ui_text(report_txt)
        if text_exist:
            candidates = ocr_detect(report_img)
            consistent = self.detect_consistency_by_ui_text(report_txt, candidates)
        else:
            consistent = self.detect_consistency_by_vision(report_txt, report_img)
        logger.info(f"Report #{report_id} Consistent? {consistent}")
        return consistent

    @timeit
    def run_without_using_ocr(self, report_id: str, report_txt: str, report_img: str):
        visible = self.visible_in_screenshot(report_txt)
        if visible:
            consistent = self.detect_consistency_by_vision(report_txt, report_img)
        else:
            consistent = self.detect_consistency_by_visual_state(report_txt, report_img)
        logger.info(f"Report #{report_id} Consistent? {consistent}")
        return consistent

    @timeit
    def run_without_verify_visibility(self, report_id: str, report_txt: str, report_img: str):
        visible = self.visible_in_screenshot(report_txt)
        if visible:
            text_exist = self.direct_reflect_from_ui_text(report_txt)
            if text_exist:
                candidates = ocr_detect(report_img)
                consistent = self.detect_consistency_by_ui_text(report_txt, candidates)
            else:
                consistent = self.detect_consistency_by_vision(report_txt, report_img)
        else:
            screen_txt = ocr_detect(report_img)
            description = self.describe_app_state_by_ocr_result(screen_txt)
            consistent = self.detect_consistency_by_textual_state(report_txt, description)
            if not consistent:
                consistent = self.detect_consistency_by_visual_state(report_txt, report_img)
        logger.info(f"Report #{report_id} Consistent? {consistent}")
        return consistent

    @timeit
    def run_with_bare_llm(self, report_id: str, report_txt: str, report_img: str):
        result, in_use, out_use, in_cost, out_cost = \
            query(user_msg_txt=report_txt, user_msg_img=report_img, system_msg=self.prompt9, model="gpt-4o-mini")
        logger.info(result)
        logger.info(f"Input token: {in_use} (${in_cost:6f}); Output token: {out_use} (${out_cost:6f})")
        logger.info(f"Report #{report_id} Consistent? {result['result']}")
        return result["result"]

    @staticmethod
    def download_dataset():
        dataset = load_reports()
        for item in dataset:
            download_img_from_url(item["image_url"], item["index"])


def main():
    re = RuleEngine()
    # re.download_dataset()
    reports = load_reports()
    for report in reports:
        idx = report["index"]
        text = report["description"]
        img = str(dataset_base / "images" / f"{idx}.jpg")
        try:
            re.run(idx, text, img)
            # re.run_without_using_ocr(app, idx, text, img)
            # re.run_without_check_visibility(app, idx, text, img)
            # re.run_without_verify_visibility(app, idx, text, img)
            # re.run_with_bare_llm(app, idx, text, img)
        except Exception as e:
            logger.warning(f"Analysis for Report #{idx} failed -- {e}")


if __name__ == "__main__":
    main()
