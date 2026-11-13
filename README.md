# Medical Image Segmentation: Polyp Segmentation (Kvasir-SEG)

Dự án Bài tập lớn môn **Học sâu (Deep Learning)** - Phân đoạn vùng tổn thương (polyp nội soi) trên ảnh y tế mức điểm ảnh, so sánh kiến trúc Encoder–Decoder CNN (U-Net) với các kiến trúc dựa trên Transformer (SegFormer, TransUNet).

---

## 📁 Cấu trúc thư mục dự án

```text
BTL/
├── configs/                     # Cấu hình quản lý bởi Hydra
│   ├── config.yaml              # Cấu hình chính (chọn dataset, model, hyperparameters)
│   ├── dataset/                 # Cấu hình chi tiết cho từng bộ dữ liệu
│   │   └── kvasir.yaml          # Đường dẫn, kích thước ảnh, batch size,...
│   └── model/                   # Cấu hình chi tiết cho từng kiến trúc mô hình
│       ├── unet.yaml            # U-Net baseline & pretrained encoder
│       ├── segformer.yaml       # SegFormer (Transformer-based)
│       └── transunet.yaml       # TransUNet (Hybrid CNN-Transformer)
│
├── dataset/                     # Dữ liệu phục vụ huấn luyện (được ignore trên git)
│   └── kvasir-seg/
│       └── Kvasir-SEG/
│           ├── images/          # 1000 ảnh nội soi đường tiêu hóa
│           ├── masks/           # 1000 mặt nạ phân đoạn nhị phân tương ứng
│           └── kavsir_bboxes.json
│
├── src/                         # Mã nguồn chính của dự án
│   ├── datasets/                # DataLoader & Pipeline tiền xử lý/tăng cường ảnh
│   │   └── kvasir_dataset.py
│   ├── models/                  # Các mô hình mạng nơ-ron
│   │   ├── unet.py              # U-Net tự cài đặt từ đầu & pretrained encoder
│   │   ├── segformer.py         # SegFormer
│   │   └── transunet.py         # TransUNet
│   ├── losses/                  # Các hàm mất mát (Dice, BCE, Focal, Tversky Loss)
│   └── utils/                   # Các hàm hỗ trợ (Metrics: Dice, IoU; Visualizer; Logger)
│       └── metrics.py
│
├── test_env.py                  # Script kiểm tra môi trường, nạp config Hydra & forward pass
├── train.py                     # Script huấn luyện chính (Tuần 2+)
├── evaluate.py                  # Script đánh giá mô hình trên tập test
├── requirements.txt             # Danh sách thư viện phụ thuộc
├── .gitignore                   # Loại trừ các file dữ liệu lớn, checkpoints, venv, log
└── README.md                    # Tài liệu hướng dẫn dự án
```

---

## ⚙️ Cài đặt môi trường

### 1. Trên máy cá nhân (Local - CPU / GPU)

```bash
# 1. Tạo và kích hoạt môi trường ảo
python -m venv .venv
# Trên Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Trên Linux/macOS:
source .venv/bin/activate

# 2. Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### 2. Trên Kaggle / Google Colab (Huấn luyện với GPU)

Trên Kaggle notebook (chọn Accelerator GPU T4 hoặc P100):
```python
# Clone repo
!git clone https://github.com/<username>/<repo-name>.git
%cd <repo-name>

# Cài đặt thư viện bổ sung
!pip install -r requirements.txt -q
```

---

## 🚀 Chạy thử nghiệm với Hydra

Dự án sử dụng **Hydra** để quản lý tham số linh hoạt từ dòng lệnh mà không cần sửa code:

### 1. Chạy kiểm tra môi trường (Sanity Check)
```bash
python test_env.py
```

### 2. Ghi đè tham số qua dòng lệnh (CLI Override)
- **Chạy trên CPU với batch size nhỏ:**
  ```bash
  python test_env.py train.device=cpu dataset.batch_size=2
  ```

- **Chạy trên Kaggle (trỏ tới dataset của Kaggle và dùng GPU):**
  ```bash
  python test_env.py dataset.root_dir="/kaggle/input/kvasir-seg/Kvasir-SEG" train.device=cuda
  ```

