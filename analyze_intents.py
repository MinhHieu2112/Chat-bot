import json
import nltk
from collections import Counter
import matplotlib.pyplot as plt
import os

# --- Cấu hình ban đầu ---
nltk.download('punkt')
INTENTS_PATH = "intents.json"
REPORT_PATH = "intents_report.txt"


def load_data():
    """Đọc dữ liệu từ intents.json"""
    if not os.path.exists(INTENTS_PATH):
        raise FileNotFoundError(f"Không tìm thấy file {INTENTS_PATH}")
    with open(INTENTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["intents"]


def basic_statistics(intents):
    """Thống kê cơ bản"""
    num_intents = len(intents)
    total_patterns = sum(len(i["patterns"]) for i in intents)
    avg_patterns = total_patterns / num_intents
    return num_intents, total_patterns, avg_patterns


def token_analysis(intents):
    """Phân tích tần suất từ"""
    all_words = []
    for intent in intents:
        for pattern in intent["patterns"]:
            words = nltk.word_tokenize(pattern.lower())
            all_words.extend(words)
    counter = Counter(all_words)
    return counter


def plot_intent_distribution(intents):
    """Vẽ biểu đồ số lượng pattern của mỗi intent"""
    tags = [i["tag"] for i in intents]
    counts = [len(i["patterns"]) for i in intents]

    plt.figure(figsize=(10, 5))
    plt.bar(tags, counts)
    plt.xticks(rotation=45, ha="right")
    plt.title("Số lượng câu patterns trong mỗi intent")
    plt.xlabel("Tên intent")
    plt.ylabel("Số lượng patterns")
    plt.tight_layout()
    plt.savefig("intent_distribution.png")
    print("Đã lưu biểu đồ: intent_distribution.png")


def save_report(intents, counter, stats):
    """Xuất báo cáo phân tích ra file txt"""
    num_intents, total_patterns, avg_patterns = stats
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("===== BÁO CÁO PHÂN TÍCH INTENTS.JSON =====\n\n")
        f.write(f"Số lượng intents: {num_intents}\n")
        f.write(f"Tổng số câu patterns: {total_patterns}\n")
        f.write(f"Số mẫu trung bình / intent: {avg_patterns:.2f}\n\n")

        f.write("Số mẫu mỗi intent:\n")
        for intent in intents:
            f.write(f"- {intent['tag']}: {len(intent['patterns'])} patterns\n")
        f.write("\n")

        f.write("Top 15 từ xuất hiện nhiều nhất:\n")
        for word, count in counter.most_common(15):
            f.write(f"{word}: {count}\n")

    print(f"Báo cáo đã lưu tại: {REPORT_PATH}")


def main():
    print("Đang phân tích intents.json ...")
    intents = load_data()

    stats = basic_statistics(intents)
    counter = token_analysis(intents)

    plot_intent_distribution(intents)
    save_report(intents, counter, stats)
    print("Hoàn thành phân tích dữ liệu.")


if __name__ == "__main__":
    main()
