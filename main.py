import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression

# Load dữ liệu
def load_data():
    return pd.read_csv("data.csv")

# Lưu dữ liệu mới
def save_data(new_row):
    df = load_data()
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv("data.csv", index=False)

# Huấn luyện model
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

def load_data():
    """Tải và làm sạch dữ liệu từ data.csv"""
    try:
        df = pd.read_csv("data.csv")
    except FileNotFoundError:
        raise FileNotFoundError("File data.csv không tồn tại. Vui lòng kiểm tra đường dẫn.")

    # Kiểm tra các cột cần thiết
    required_columns = ["toan", "van", "anh", "hoc_luc", "nganh", "ket_qua", "CNTT", "KinhTe", "SuPham"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"File data.csv thiếu các cột: {missing_columns}")

    # Xóa cột không cần thiết (nếu có)
    if "truong" in df.columns:
        df = df.drop(columns=["truong"])

    # Kiểm tra và xử lý NaN trong ket_qua
    if df["ket_qua"].isna().any():
        print("Cảnh báo: Cột ket_qua chứa NaN. Tạo nhãn giả dựa trên tổng điểm.")
        df["ket_qua"] = (df["toan"] + df["van"] + df["anh"] >= 18).astype(int)
    
    # Kiểm tra NaN trong các cột features
    feature_columns = ["toan", "van", "anh", "hoc_luc", "CNTT", "KinhTe", "SuPham"]
    if df[feature_columns].isna().any().any():
        print("Cảnh báo: Các cột features chứa NaN. Điền NaN bằng giá trị trung bình cho cột số.")
        imputer = SimpleImputer(strategy="mean")
        df[feature_columns[:4]] = imputer.fit_transform(df[feature_columns[:4]])  # Chỉ áp dụng cho cột số

    return df

def train_model():
    """Huấn luyện mô hình và lưu vào model.pkl"""
    # Tải dữ liệu
    df = load_data()

    # Tách features và target
    X = df[["toan", "van", "anh", "hoc_luc", "CNTT", "KinhTe", "SuPham"]]
    y = df["ket_qua"]

    # Kiểm tra NaN lần cuối
    if X.isna().any().any():
        raise ValueError("Features (X) vẫn chứa NaN sau khi xử lý.")
    if y.isna().any():
        raise ValueError("Nhãn (y) vẫn chứa NaN sau khi xử lý.")

    # Huấn luyện mô hình
    model = LogisticRegression(random_state=42)
    model.fit(X, y)

    # Lưu mô hình
    try:
        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)
        print("Đã huấn luyện và lưu mô hình vào model.pkl")
    except Exception as e:
        raise Exception(f"Lỗi khi lưu model.pkl: {e}")

    return model

# Dự đoán
def predict(toan, van, anh, hoc_luc, nganh):
    nganh_map = {"CNTT": [1,0,0], "KinhTe": [0,1,0], "SuPham": [0,0,1]}
    X = np.array([[toan, van, anh, hoc_luc] + nganh_map[nganh]])
    model = pickle.load(open("model.pkl", "rb"))
    return model.predict(X)[0], model.predict_proba(X)[0][1]

# Giao diện
st.title("Dự đoán trúng tuyển đại học")

toan = st.slider("Điểm Toán", 0.0, 10.0, 5.0)
van = st.slider("Điểm Văn", 0.0, 10.0, 5.0)
anh = st.slider("Điểm Anh", 0.0, 10.0, 5.0)
hoc_luc_str = st.selectbox("Học lực", ["Trung bình", "Khá", "Giỏi"])
nganh = st.selectbox("Ngành đăng ký", ["CNTT", "KinhTe", "SuPham"])

hoc_luc_map = {"Trung bình": 0, "Khá": 1, "Giỏi": 2}
hoc_luc = hoc_luc_map[hoc_luc_str]

if st.button("Dự đoán"):
    label, prob = predict(toan, van, anh, hoc_luc, nganh)
    st.success(f"Xác suất đậu: {prob*100:.2f}%. Kết quả: {'Đậu' if label == 1 else 'Rớt'}")

if st.button("Lưu dữ liệu mới"):
    nganh_map = {"CNTT": [1,0,0], "KinhTe": [0,1,0], "SuPham": [0,0,1]}
    new_data = {
        "toan": toan,
        "van": van,
        "anh": anh,
        "hoc_luc": hoc_luc,
        "nganh": nganh,
        "CNTT": nganh_map[nganh][0],
        "KinhTe": nganh_map[nganh][1],
        "SuPham": nganh_map[nganh][2],
        "ket_qua": None  # Không biết kết quả, để huấn luyện sau
    }
    save_data(new_data)
    st.success("Đã lưu dữ liệu mới. Hãy thêm nhiều dữ liệu có kết quả để huấn luyện!")

if st.button("Huấn luyện lại mô hình"):
    # Chạy hàm
    try:
        train_model()
    except Exception as e:
        print(f"Lỗi: {e}")
    st.success("Đã huấn luyện lại mô hình với dữ liệu hiện tại.")
