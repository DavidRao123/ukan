# Attention-MBA-UKAN: Biomedical Image Super-Resolution

## 📌 Overview
**Attention-MBA-UKAN** is a U-Net-based super-resolution (SR) framework for biomedical imaging. It integrates:  

- **Kolmogorov–Arnold Networks (KANs)** for nonlinear feature modeling  
- **Multi-Basis Adaptation (MBA)** with B-spline, Chebyshev, and Hermite basis functions  
- **Attention modules** to enhance spatial and channel dependencies  

This framework is designed to reconstruct high-resolution microscopy images from low-resolution data, tested on **Clathrin-Coated Pits (CCPs)** and **Microtubules**.  

---

## 📂 Dataset
The dataset can be downloaded from Zenodo:  
👉 [Zenodo Record 7115540](https://zenodo.org/records/7115540)  

- Includes **CCPs** and **Microtubules** microscopy data.  
- After downloading, both datasets can be merged, using suffixes to distinguish them.  
- Preprocessing is handled by scripts in the `dataset/` folder, which support `.mrc` image formats.  

---

## ⚙️ Project Structure
```
Attention-MBA-UKAN/
│── .idea/                     # IDE config files (can be ignored)
│── data/                      # Training & validation datasets
│── dataset/                   # Scripts for handling MRC data
│── logs/                      # Training logs
│── models/                    # Model architectures
│── utils/                     # Utility functions
│
│── .gitignore                 # Git ignore rules
│── README.md                  # Project description
│── requirements.txt           # Python dependencies
│
│── run_train.sh               # Shell script to launch training
│── run_test.sh                # Shell script to launch evaluation
│── train_biosr.py             # Main training script
│── test_biosr.py              # Evaluation script
│── utest_biosr.py             # Unit tests
```

---

## 🚀 Installation
Clone the repo and install dependencies:
```bash
git clone https://github.com/DavidRao123/ukan.git
cd ukan
pip install -r requirements.txt
```

Dependencies are listed in `requirements.txt`.

---

## 🏋️ Training
Example command for training **Attention-MBA-UKAN**:
```bash
python train_biosr.py   --train_dir ./data/train/   --val_dir ./data/val/   --Num_epoch 100   --batch_size 32   --model_name AttenMSA_UKAN   --exp_name exp_attenmsaukan
```

Logs are saved in `logs/`, and checkpoints are stored under `outputs/checkpoints/`.

Alternatively, you can use the shell script:
```bash
sh run_train.sh
```

---

## 🔍 Evaluation
To evaluate a trained model on the test set, run:  
```bash
python utest_biosr.py   --data_dir ./data/CM/test   --model_name AttenMSA_UKAN   --save_pred_image   --model_weights ./logs/train_log
```

### Key Arguments
- `--data_dir` : Test dataset path  
- `--model_name` : Model architecture name (e.g., `UKAN`, `MBA_UKAN`, `AttenMSA_UKAN3`)  
- `--model_weights` : Path to the trained model checkpoint (`.pth`)  
- `--save_pred_image` : If set, saves the predicted super-resolution images  

Alternatively, you can also run:
```bash
sh run_test.sh
```

---

## 📦 Dependencies
Install all required dependencies with:
```bash
pip install -r requirements.txt
```

---

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request.  
