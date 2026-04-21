from cup_utils import get_decision_value
from inference import load_model

if __name__ == "__main__":
    model = load_model("C:\Users\ADMIN\Documents\cup_disk_ratio_api\model_weight\svm_model_v4.pkl")
    get_decision_value()