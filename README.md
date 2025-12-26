### InconHunter: Human Cognitive Pattern Simulation for Crowdsourced Test Report Consistency Detection

---

**0. Dataset Prepare**

The dataset we construct is in dataset/dataset.txt.

Call method `download_dataset()` in `main.py` to download images of the dataset.

```python
def download_dataset():
    dataset = load_reports()
    for item in dataset:
        download_img_from_url(item["image_url"], item["index"])
```

---

**1. Environment Setup**

$ pip install -r requirements.txt

---

**2. LLM API Settings**

Set OpenAI API key in Environment Variable `OPENAI_API_KEY`.

Make sure the "gpt-4o" and "gpt-4o-mini" model is available by running `python llm.py`.

Change the LLM service in `llm.py` is also feasible.

---

**3. Run & Evaluate**

$ python main.py

$ python result_analysis.py <log_file_path>

---

**4. Code Explanation**

main.py: implementation of automated crowdsourced test report consistency detection

result_analysis.py: implementation of the evaluation of consistency detection

llm.py: managing the LLM querying service

ocr_detect.py: implementation of OCR for text extraction from images

---

**5. Summary of Prompt Templates**（detailed prompts are shown in `main.py`）

| Prompt ID    | Component                               | Purpose                                                      | Core Reasoning Logic                                         |
| ------------ | --------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Prompt 1** | Bug Visibility Analysis                 | Determines whether a reported issue has observable visual manifestations. | Evaluates if the issue description implies visible UI anomalies (e.g., missing elements, layout issues, error messages) or is inherently non-visual. |
| **Prompt 2** | UI Text-Based Semantics Matching        | Determines whether the issue can be validated purely through on-screen text. | Checks if the issue description can be directly confirmed by textual UI elements without visual reasoning. |
| **Prompt 3** | UI Text-Based Semantics Matching        | Validates consistency between the issue description and OCR-extracted UI text. | Infers expected textual features from the report and verifies their presence in OCR results. |
| **Prompt 4** | Visual Feature-Based Semantics Matching | Verifies whether the screenshot visually supports the reported issue. | Uses vision-based reasoning to match described visual anomalies with screenshot content. |
| **Prompt 5** | Bug Visibility Analysis                 | Re-evaluates “invisible” bugs using OCR signals.             | Determines whether supposedly invisible issues actually surface through textual error messages or alerts. |
| **Prompt 6** | Contextual Coherence Reasoning          | Produces a concise textual description of the app state from OCR text. | Summarizes extracted UI text into a coherent description of the current interface state. |
| **Prompt 7** | Visual Context Matching                 | Checks whether the visual app state can plausibly lead to the reported issue. | Assesses if a single user action from the shown UI could trigger the reported bug. |
| **Prompt 8** | Textual Context Matching                | Performs contextual reasoning using a textual description of the app state. | Evaluates logical consistency between the described app state and the reported issue without relying on images. |
