import re
import sys

from utils import get_labels_true

def logic_chain_triggering_analysis(log_file: str):
    result = {}
    with open(log_file, mode="r", encoding="utf-8") as f:
        pattern1 = r'\{(["\'])result(["\']):\s*(True|False),\s*(["\'])reason(["\']):\s*(["\'])(.*?)(["\'])\}'
        pattern2 = "Report #(\d+) Consistent?"
        temp = []
        for line in f.readlines():
            match1 = re.search(pattern1, line)
            match2 = re.search(pattern2, line)
            if "WARNING" in line:
                temp.clear()
            if match1:
                temp.append(match1.group(3) == "True")
            elif match2:
                key = "-".join(str(e) for e in temp)
                if key not in result.keys():
                    result[key] = 1
                else:
                    result[key] += 1
                temp.clear()
    total = sum(v for v in result.values())
    for k, v in result.items():
        print(k, v, v / total)

def classification_analysis(log_file: str):
    true_dict = get_labels_true()
    pred_dict = {}
    with open(log_file, mode="r", encoding="utf-8") as f:
        pattern = "Report #(\d+) Consistent?"
        for line in f.readlines():
            match = re.search(pattern, line)
            if match:
                report_id = match.group(1)
                consistent = True if line.strip().endswith("True") else False
                pred_dict[report_id] = consistent
    tp, tn, fp, fn = 0, 0, 0, 0
    for k, v in pred_dict.items():
        pred, true = v, true_dict[k]
        if pred and true:
            tp += 1
        elif not pred and not true:
            tn += 1
        elif pred and not true:
            fp += 1
        elif not pred and true:
            fn += 1
    print(f"tp = {tp}\ntn = {tn}\nfp = {fp}\nfn = {fn}\ntotal = {tp + tn + fp + fn}")
    print(f"accuracy = {(tp + tn) / (tp + tn + fp + fn)}")
    print(f"precision = {tp / (tp + fp)}")
    print(f"recall = {tp / (tp + fn)}")
    print(f"f1-score = {2 * tp / (2 * tp + fp + fn)}")

def cost_analysis(log_file: str):
    total = 0
    money_usage, token_usage = [], []
    with open(log_file, mode="r", encoding="utf-8") as f:
        pattern1 = r"Input token: (\d+) \(\$(\d+\.\d+)\); Output token: (\d+) \(\$(\d+\.\d+)\)"
        pattern2 = "Report #(\d+) Consistent?"
        for line in f.readlines():
            match1 = re.search(pattern1, line)
            match2 = re.search(pattern2, line)
            if match1:
                input_token = int(match1.group(1))
                input_price = float(match1.group(2))
                output_token = int(match1.group(3))
                output_price = float(match1.group(4))
                money_usage.append(input_price + output_price)
                token_usage.append(input_token + output_token)
            if match2:
                total += 1
    print(f"avg money cost: {sum(money_usage) / total}")
    print(f"avg token cost: {sum(token_usage) / total}")


if __name__ == "__main__":
    log_file_path = sys.argv[1]
    cost_analysis(log_file_path)
    classification_analysis(log_file_path)
    logic_chain_triggering_analysis(log_file_path)
