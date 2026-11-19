# Polyp Segmentation — trạng thái hiện tại

Repository này là base code cho bài tập lớn Học sâu về phân đoạn polyp trên Kvasir-SEG. Hiện tại framework train/evaluate đã có và model demo là TransUNet tự cài bằng PyTorch. Các model còn lại của nhóm, như U-Net baseline, U-Net pretrained encoder hoặc SegFormer, **chưa được thêm**.

Notebook `dl-uet-polyp.ipynb` chỉ là bản demo ban đầu trên Kaggle, không phải entrypoint chính. Mọi thí nghiệm tiếp theo nên đi qua `train.py` và config Hydra.

## Hiện đã có

- Kvasir-SEG được tải bằng `kagglehub.dataset_download("datnd004/kvasir-seg")` nếu chưa truyền data root local.
- Data split cố định trong `kvasir-seg.json`: 800 train, 100 validation, 100 test.
- Pipeline train/validation/test, checkpoint theo validation IoU, early stopping, AMP CUDA, prediction export.
- Loss config: BCE, Dice, BCE + Dice, Focal, Tversky, Focal-Tversky.
- Optimizer: AdamW, SGD. Scheduler: ReduceLROnPlateau, cosine annealing, none.
- W&B logger ở offline mode mặc định; đổi sang online bằng biến môi trường `WANDB_MODE`.
- TransUNet tại `src/models/transunet.py` là model duy nhất đã có config sẵn.

Chưa có kết quả train, checkpoint, test suite hoặc model baseline nào trong repo.

## Cấu trúc đang dùng

```text
configs/
  dataset/kvasir.yaml       Data module, KaggleHub, transform factories
  model/transunet.yaml      Model mặc định
  loss/*.yaml               Chọn loss
  optimizer/*.yaml          Chọn optimizer
  scheduler/*.yaml          Chọn scheduler
  logger/*.yaml             W&B hoặc none
  trainer/default.yaml      Training loop options
  config.yaml               File compose tất cả config group
src/
  data/                     Dataset và transforms
  losses/                   Loss implementations
  models/                   Model implementations
  training/                 Trainer và scheduler factories
  utils/                    Metrics, seed, W&B adapter
train.py                    Train entrypoint
evaluate.py                 Test entrypoint
test_env.py                 Kiểm tra data/model/loss một batch
kvasir-seg.json            Split chuẩn của nhóm
```

## Cài môi trường

Tạo môi trường Python, cài PyTorch đúng bản CPU/GPU của máy từ [PyTorch](https://pytorch.org/get-started/locally/), rồi cài các dependency còn lại:

```powershell
pip install -r requirements.txt
```

Nếu dùng W&B online, đăng nhập một lần:

```powershell
wandb login
```

## Kiểm tra trước khi train

Lệnh này tải/tìm dataset, kiểm tra split, tạo một batch, khởi tạo model và chạy một forward pass:

```powershell
python test_env.py
```

Nếu dataset đã có sẵn trên máy, truyền thư mục chứa `images/` và `masks/`:

```powershell
python test_env.py dataset.root_dir='D:/data/Kvasir-SEG'
```

## Train

Train bằng config mặc định: TransUNet + BCE/Dice + AdamW + ReduceLROnPlateau.

```powershell
python train.py
```

Một số override Hydra hữu ích:

```powershell
python train.py dataset.batch_size=4 dataset.image_size=[256,256] model.input_size=256
python train.py loss=focal_tversky scheduler=cosine optimizer=sgd
$env:WANDB_MODE='online'; python train.py
python train.py logger=none trainer.device=cpu
```

Checkpoint tốt nhất được lưu ở `outputs/checkpoints/<run_name>/best.ckpt`. Mọi quyết định chọn checkpoint đều theo validation metric, không theo test metric.

## Evaluate

Đánh giá checkpoint trên test split cố định và lưu prediction mask PNG:

```powershell
python evaluate.py evaluation.checkpoint_path='outputs/checkpoints/<run_name>/best.ckpt'
```

Để không lưu prediction masks:

```powershell
python evaluate.py evaluation.checkpoint_path='outputs/checkpoints/<run_name>/best.ckpt' evaluation.save_predictions=false
```

## Cách thêm model

Mỗi model mới cần đúng hai phần:

1. Tạo class `torch.nn.Module` trong `src/models/`. Input là `[B, 3, H, W]`, output là binary logits `[B, 1, H, W]`.
2. Tạo YAML cùng tên trong `configs/model/`, ví dụ:

```yaml
_target_: src.models.unet.UNet
name: unet
in_channels: 3
out_channels: 1
base_channels: 32
```

Sau đó train bằng:

```powershell
python train.py model=unet
```

Không sửa `train.py` hay trainer để chọn model. Hydra dùng `_target_` để khởi tạo model/loss/data module/logger; optimizer, scheduler và transform dùng `_partial_` để trainer truyền dependency runtime vào factory.

## Quy ước làm việc

- Không đổi `kvasir-seg.json` nếu chưa thống nhất với cả nhóm.
- Không chọn hyperparameter/checkpoint theo test score.
- Không commit `dataset/`, `outputs/`, `wandb/` hoặc checkpoint; chúng đã được ignore.
- Mỗi run cần lưu config, validation metric tốt nhất, test metric cuối cùng và ảnh dự đoán để dùng cho báo cáo.
