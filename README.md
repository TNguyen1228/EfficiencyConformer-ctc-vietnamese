# Vietnamese CTC Whisper ASR

Hệ thống nhận dạng giọng nói tiếng Việt hiệu suất cao dựa trên CTC (Connectionist Temporal Classification) với PhoWhisper encoder.

## Tổng Quan Dự Án

Model này được thiết kế đặc biệt để nhận dạng giọng nói tiếng Việt với hiệu suất vượt trội so với Whisper gốc của OpenAI. Sử dụng kiến trúc CTC thay vì autoregressive decoder, model đạt được tốc độ inference nhanh hơn 3-5 lần và độ chính xác cao hơn cho tiếng Việt.

## ⚡ Những Khác Biệt Chính So Với Whisper Gốc

### 🏗️ Kiến Trúc Model

| Thành Phần | Whisper Gốc | Model Này | Lợi Ích |
|------------|-------------|-----------|---------|
| **Decoder** | Autoregressive Transformer | CTC Decoder | ⚡ Nhanh hơn 3-5x |
| **Memory** | Cao (decoder cache) | Thấp (không cache) | 💾 Tiết kiệm 40% memory |
| **Stability** | Error propagation | Independent prediction | 🎯 Ổn định hơn |

### 🧠 Encoder Cải Tiến

- **PhoWhisper Encoder**: Được pre-train đặc biệt cho tiếng Việt
- **ALiBi Attention**: Xử lý audio dài tốt hơn, hỗ trợ streaming
- **Optimized Context**: Không bị giới hạn 30 giây như Whisper gốc

### 🎯 CTC Decoder Thông Minh

- **Prefix Beam Search**: Chính xác hơn standard beam search
- **Label Smoothing**: Giảm overfitting, cải thiện generalization
- **Length Normalization**: Cân bằng giữa độ dài và chất lượng

### 📝 Tokenizer Chuyên Biệt

- **SentencePiece BPE**: 1024 vocab size tối ưu cho tiếng Việt
- **Subword Handling**: Xử lý từ ghép và từ mới tiếng Việt hiệu quả
- **Compact Vocabulary**: Nhỏ gọn nhưng hiệu quả hơn tokenizer đa ngôn ngữ

### 🎵 Xử Lý Audio Thông Minh

- **Flexible Length**: Không giới hạn độ dài audio
- **Real-world Noise**: Inject tiếng ồn thực tế

## 📊 Chuẩn Bị Dữ Liệu

### 📁 Cấu Trúc Thư Mục

```text

whisper_asr/
├── metadata.csv              # File metadata chính
├── datatest/                 # Thư mục chứa file audio
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
└── weights/                  # Model weights và tokenizer
    ├── phowhisper_small_encoder.pt
    └── tokenizer_spe_bpe_v1024_pad/
        └── tokenizer.model
```

### 📝 Format Dữ Liệu

Model sử dụng **một file CSV duy nhất** với format đơn giản:

```csv
path|text
./datatest/audio1.wav|chụp cộng hưởng từ phát hiện u tuyến yên kích thước 8mm.
./datatest/audio2.wav|khi mặt trời ló rạng, sương đọng trên lá bỗng long lanh hơn.
./datatest/audio3.wav|bệnh nhân nhập viện với chẩn đoán viêm cơ tim cấp do virus.
```

### 🔧 Tính Năng Dữ Liệu

- **Auto Train/Val Split**: Tự động chia 95% train, 5% validation
- **Text-only Filtering**: Lọc theo độ dài text (1-60 ký tự)
- **No Duration Limit**: Không giới hạn độ dài audio
- **Reproducible Split**: Sử dụng random seed để đảm bảo kết quả nhất quán

### 📋 Yêu Cầu Dữ Liệu

- **Audio Format**: WAV, MP3, FLAC (khuyến nghị WAV 16kHz)
- **Text Quality**: Chú ý dấu câu và chính tả tiếng Việt
- **File Size**: Không giới hạn (model xử lý được audio dài)
- **Minimum**: Ít nhất 100 samples để test

## ⚙️ Cài Đặt

### 📦 Yêu Cầu Hệ Thống

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (khuyến nghị)
- RAM: 8GB+ (16GB khuyến nghị)
- GPU: 6GB+ VRAM

### 🛠️ Cài Đặt Dependencies

```bash
# Clone repository
git clone https://github.com/iamdinhthuan/vietnamese-ctc-whispe
cd vietnamese-ctc-whisper

# Cài đặt dependencies
pip install -r requirements.txt

# Cài đặt PyTorch (cho CUDA 11.8)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```

### 📥 Download Model Weights

```bash
wget https://huggingface.co/dinhthuan/phowhisper_small_encoder/resolve/main/phowhisper_small_encoder.pt -O weights/phowhisper_small_encoder.pt
```

## 🚀 Sử Dụng

### 1️⃣ Training Cơ Bản

```bash
# Train với config mặc định
python run.py

# Train với config tùy chỉnh
python run.py --config config.json

# Train với parameters override
python run.py --batch-size 16 --learning-rate 2e-4 --max-epochs 30
```

### 2️⃣ Config Tùy Chỉnh

```python
from config import get_config

# Load config mặc định
config = get_config()

# Tùy chỉnh parameters
config.training.batch_size = 32
config.training.learning_rate = 1e-4
config.model.dropout = 0.15

# Lưu config
config.save("my_config.json")

# Train với config mới
# python run.py --config my_config.json
```

### 3️⃣ Inference

```bash
# Inference cơ bản
python inference.py --checkpoint checkpoints/best.ckpt --audio audio.wav

# Với beam search
python inference.py --checkpoint checkpoints/best.ckpt --audio audio.wav --beam_search

# Batch inference
python inference.py --checkpoint checkpoints/best.ckpt --audio_dir audio_folder/ --output results.txt
```

## ⚙️ Cấu Hình Chi Tiết

### 🎵 Audio Processing

```json
{
  "audio": {
    "sample_rate": 16000,     // Tần số lấy mẫu
    "n_mels": 80,            // Số mel filters
    "n_fft": 400,            // FFT window size
    "hop_length": 160        // Hop length cho STFT
  }
}
```

### 🧠 Model Architecture

```json
{
  "model": {
    "n_state": 768,          // Dimension của model
    "n_head": 12,            // Số attention heads
    "n_layer": 12,           // Số transformer layers
    "vocab_size": 1024,      // Kích thước vocabulary
    "dropout": 0.1,          // Dropout rate
    "label_smoothing": 0.1   // Label smoothing factor
  }
}
```

### 🏋️ Training Parameters

```json
{
  "training": {
    "batch_size": 16,              // Batch size
    "learning_rate": 1e-4,         // Learning rate ban đầu
    "max_epochs": 50,              // Số epochs tối đa
    "precision": "bf16-mixed",     // Mixed precision training
    "gradient_clip_val": 1.0,      // Gradient clipping
    "num_sanity_val_steps": 0      // Tắt sanity checking
  }
}
```

### 📊 Data Configuration

```json
{
  "data": {
    "metadata_file": "metadata.csv",     // File metadata
    "train_val_split": 0.95,             // Tỷ lệ train/val
    "min_text_len": 1,                   // Độ dài text tối thiểu
    "max_text_len": 60,                  // Độ dài text tối đa
    "enable_augmentation": true,         // Bật data augmentation
    "augmentation_prob": 0.8,            // Xác suất augmentation
    "random_seed": 42                    // Random seed
  }
}
```

## 📈 Tips Training Hiệu Quả

### 🚀 Cho Thí Nghiệm Nhanh

```python
config.model.n_state = 512        # Model nhỏ hơn
config.model.n_layer = 6          # Ít layers hơn
config.training.batch_size = 8     # Batch nhỏ
config.training.max_epochs = 10    # Training ngắn
```

### 🎯 Cho Chất Lượng Cao

```python
config.model.n_state = 768        # Model đầy đủ
config.training.batch_size = 32    # Batch lớn
config.training.max_epochs = 100   # Training dài
config.data.augmentation_prob = 0.9  # Augmentation nhiều
```

### 🏭 Cho Production

```python
config.training.precision = "bf16-mixed"     // Mixed precision
config.training.accumulate_grad_batches = 2  // Gradient accumulation
config.training.num_workers = 8              // Parallel data loading
```

## 📊 Monitoring Training

### 📈 TensorBoard

```bash
# Khởi động TensorBoard
tensorboard --logdir checkpoints

# Xem tại http://localhost:6006
```

### 🔍 Metrics Quan Trọng

- **Training Loss**: Giảm đều theo epochs
- **Validation WER**: Word Error Rate (càng thấp càng tốt)
- **Learning Rate**: Theo OneCycle schedule
- **Step Time**: Thời gian mỗi step training

## 🎯 Strategies Inference

### ⚡ Greedy Decoding

```python
# Nhanh nhất, chất lượng tốt
inference = CTCInference(checkpoint_path, config)
result = inference.transcribe_single(audio_path, use_beam_search=False)
```

### 🎯 Prefix Beam Search

```python
# Chậm hơn nhưng chính xác nhất
result = inference.transcribe_single(
    audio_path, 
    use_beam_search=True,
    beam_size=10,
    length_penalty=0.3
)
```

## 🐛 Troubleshooting

### ❌ Memory Issues

```python
# Giảm batch size
config.training.batch_size = 8

# Sử dụng gradient accumulation
config.training.accumulate_grad_batches = 4

# Tắt cache audio
config.data.enable_caching = False
```

### 🐌 Training Chậm

```python
# Sử dụng mixed precision
config.training.precision = "bf16-mixed"

# Tắt sanity checking
config.training.num_sanity_val_steps = 0

# Single-threaded data loading (Windows)
config.training.num_workers = 0
```

### 💥 NaN Loss

```python
# Giảm learning rate
config.training.learning_rate = 5e-5

# Tăng gradient clipping
config.training.gradient_clip_val = 0.5

# Kiểm tra data quality
```

## 📚 Examples Thực Tế

### 🎙️ Training với Medical Data

```python
# Config cho y tế
config = get_config()
config.data.metadata_file = "medical_data.csv"
config.model.dropout = 0.2  # Tăng dropout cho domain-specific
config.training.max_epochs = 200  # Training lâu hơn
config.save("medical_config.json")
```

### 📞 Training với Call Center Data

```python
# Config cho call center
config.data.bg_noise_path = ["./noise/call_center/"]
config.data.augmentation_prob = 0.95  # Augmentation mạnh
config.audio.sample_rate = 8000  # Tần số thấp
config.save("callcenter_config.json")
```

## 🤝 Đóng Góp

### 📝 Báo Lỗi

- Mở issue trên GitHub với thông tin chi tiết
- Bao gồm config, logs và error message
- Mô tả steps để reproduce

### 💡 Đề Xuất Tính Năng

- Mô tả rõ use case và lợi ích
- Đưa ra implementation approach nếu có thể
- Thảo luận trong Discussions trước khi code

### 🔧 Pull Requests

- Fork repo và tạo feature branch
- Viết tests cho code mới
- Đảm bảo code style consistency
- Update documentation nếu cần

## 📄 License

MIT License - Xem file LICENSE để biết chi tiết.

## 📞 Liên Hệ

- **GitHub Issues**: Báo bugs và feature requests
- **Email**: <bpyphuthien115@gmail.com>

## 🙏 Acknowledgments

- **PhoWhisper Team**: Cung cấp pre-trained encoder cho tiếng Việt
- **OpenAI**: Whisper architecture inspiration
- **PyTorch Lightning**: Framework training mạnh mẽ
- **SentencePiece**: Tokenization library hiệu quả

---

⭐ **Nếu project này hữu ích, hãy star repo để ủng hộ nhé!** ⭐
