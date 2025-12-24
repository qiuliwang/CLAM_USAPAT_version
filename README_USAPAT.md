# 🧠 USAPAT: Unsupervised Stain-Aware Pixel-Adversarial Transfer Learning for Virtual Immunohistochemical Staining

无监督掩码引导的虚拟IHC染色生成。

---

## 🚀 快速开始

### 1. 安装
```bash
git clone https://github.com/yourname/your-repo.git
cd your-repo
pip install -r requirements.txt
```

### 2. 数据目录
```bash
datasets/BCI_example/
├── TrainValAB/
│   ├── trainA/
│   ├── trainB/
│   ├── valA/
│   └── valB/
```

### 3. 训练 / 测试
### 流程测试（128×128，3 epoch，快速验证）
```bash
python train.py --name transformer_mask_BCI --model cycle_gan --gpu_ids 0 --batch_size 1 --epoch_count 0 --n_epochs 2 --n_epochs_decay 1 --load_size 128 --crop_size 128 --num_threads 1 --display_freq 1 --netG resnet_6blocks_vit --dataroot ./datasets/BCI_example/TrainValAB --maskroot ./datasets/BCI_example/TrainValAB --no_flip
```


### 正式训练（1024×1024，200 epoch）
```bash
python train.py --name transformer_mask_BCI --model cycle_gan --gpu_ids 0 --batch_size 1 --epoch_count 0 --n_epochs 200 --n_epochs_decay 100 --load_size 1024 --crop_size 1024 --num_threads 8 --display_freq 1000 --netG resnet_6blocks_vit --dataroot ./datasets/BCI_example/TrainValAB --maskroot ./datasets/BCI_example/TrainValAB --no_flip
```


### 测试（生成 fake 图）
```bash
python test_onlyFake.py --dataroot ./datasets/BCI_example/TrainValAB/valA --name transformer_mask_BCI --model test --no_dropout --results_dir ./runs --load_size 1024 --crop_size 1024 --num_test 20000 --epoch 300 --gpu_ids 0 --netG resnet_6blocks_vit
```
### 结果保存在 runs/transformer_mask_BCI/test_300/images/

### 常用参数
```bash
| 参数                                | 说明                            |
| --------------------------------- | ----------------------------- |
| `--name`                          | 实验名称                          |
| `--load_size`                     | 加载图像短边尺寸                      |
| `--crop_size`                     | 训练裁剪尺寸                        |
| `--n_epochs` / `--n_epochs_decay` | 训练与衰减阶段 epoch 数               |
| `--netG`                          | 生成器结构，默认 `resnet_6blocks_vit` |
```

### 断点续训
```bash
python train.py ... --continue_train --epoch_count <last_epoch>
```

### 许可证
```bash
MIT
```