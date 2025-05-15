### InconHunter: Human Cognitive Pattern Simulation for Crowdsourced Test Report Consistency Detection

**0. Dataset Prepare**

The dataset we construct is in dataset/dataset.txt.

Call method `download_dataset()` in `main.py` to download images of the dataset.

```python
def download_dataset():
    dataset = load_reports()
    for item in dataset:
        download_img_from_url(item["image_url"], item["index"])
```

**1. Environment Setup**

$ pip install -r requirements.txt

**2. LLM API Settings**

Set OpenAI API key in Environment Variable `OPENAI_API_KEY`.

Make sure the "gpt-4o" and "gpt-4o-mini" model is available by running `python llm.py`.

Change the LLM service in `llm.py` is also feasible.

**3. Run & Evaluate**

$ python main.py

$ python result_analysis.py <log_file_path>

**4. Code Explanation**

main.py: implementation of automated crowdsourced test report consistency detection

result_analysis.py: implementation of the evaluation of consistency detection

llm.py: managing the LLM querying service

ocr_detect.py: implementation of OCR for text extraction from images