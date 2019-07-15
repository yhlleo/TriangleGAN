### Metrics

-------

 - [x] Mean Square Error (MSE)
 - [x] Peak Signal-toNoise Ratio (PSNR)
 - [x] Inception Score (IS)
 - [x] Fréchet Inception Distance (FID)
 - [x] Fréchet Resnet Distance (FRD) [Matlab verison](https://github.com/Ha0Tang/GestureGAN)
 - [x] Precision Recall Distance (PRD)

The lower, the better: MSE, FID, FRD

The higher, the better: PSNR, IS

## Usage

```
python3 eval.py \
  --metric_mode <mode> \
  --model_name <model_name> \
  --suffix_pred fake_B \
  --gpu_id <gpu_id>
```

 