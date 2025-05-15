import re

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
    yy, nn, yn, ny = 0, 0, 0, 0
    for k, v in pred_dict.items():
        pred, true = v, true_dict[k]
        if pred and true:
            yy += 1
        elif not pred and not true:
            nn += 1
        elif pred and not true:
            ny += 1
        elif not pred and true:
            yn += 1
    print(f"yy = {yy}\nnn = {nn}\nyn = {yn}\nny = {ny}\ntotal = {yy+nn+ny+yn}")
    print(f"accuracy = {(yy + nn) / (yy + nn + ny + yn)}")
    print(f"precision = {yy / (yy + yn)}")
    print(f"recall = {yy / (yy + ny)}")
    print(f"f1-score = {2 * yy / (2 * yy + ny + yn)}")

def cost_analysis(log_file: str):
    total = 0
    money_usage, token_usage = [], []
    with open(log_file, mode="r", encoding="utf-8") as f:
        pattern1 = r"Input token: (\d+) \(\$(\d+\.\d+)\); Output token: (\d+) \(\$(\d+\.\d+)\)"
        pattern2 = "Report #(\d+) from App-(\d+) Consistent?"
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
    cost_analysis("")
    classification_analysis("")
    logic_chain_triggering_analysis("")
