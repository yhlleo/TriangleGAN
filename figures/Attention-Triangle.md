### Attention Map with Triangle & Boundary

 - NTU

|-|Source Image|Ground Truth|PG2|Ours|
|----|:----:|:----:|----|----|
|Attention|x|x|![](./ntu_full/pg2gan/P1-G2-6-AB-P1-G9-8_diff_map.png)|![](./ntu_full/rollinggan/P1-G2-6-AB-P1-G9-8_fake_B2_mask.png)|
|Output|![](./ntu_full/source-image/P1-G2-6-AB-P1-G9-8_real_A.png)|![](./ntu_full/ground-truth/P1-G2-6-AB-P1-G9-8_real_B.png)|![](./ntu_full/pg2gan/P1-G2-6-AB-P1-G9-8_fake_B2.png)|![](./ntu_full/rollinggan/P1-G2-6-AB-P1-G9-8_fake_B2_masked.png)|
|Attention|x|x|![](./ntu_full/pg2gan/P10-G1-4-AB-P10-G3-9_diff_map.png)|![](./ntu_full/rollinggan/P10-G1-4-AB-P10-G3-9_fake_B2_mask.png)|
|Output|![](./ntu_full/source-image/P10-G1-4-AB-P10-G3-9_real_A.png)|![](./ntu_full/ground-truth/P10-G1-4-AB-P10-G3-9_real_B.png)|![](./ntu_full/pg2gan/P10-G1-4-AB-P10-G3-9_fake_B2.png)|![](./ntu_full/rollinggan/P10-G1-4-AB-P10-G3-9_fake_B2_masked.png)|

 - Senz3D

|-|Source Image|Ground Truth|PG2|Ours-triangle|Ours-boundary|
|----|:----:|:----:|----|----|----|
|Attention|x|x|![](./senz3d_full/pg2gan/S1-G1-16-color-AB-S1-G4-26-color_diff_map.png)|![](./senz3d_full/rollinggan-t/S1-G1-16-color-AB-S1-G4-26-color_fake_B2_mask.png)|![](./senz3d_full/rollinggan-b/S1-G1-16-color-AB-S1-G4-26-color_fake_B2_mask.png)|
|Output|![](./senz3d_full/source-image/S1-G1-16-color-AB-S1-G4-26-color_real_A.png)|![](./senz3d_full/ground-truth/S1-G1-16-color-AB-S1-G4-26-color_real_B.png)|![](./senz3d_full/pg2gan/S1-G1-16-color-AB-S1-G4-26-color_fake_B2.png)|![](./senz3d_full/rollinggan-t/S1-G1-16-color-AB-S1-G4-26-color_fake_B2_masked.png)|![](./senz3d_full/rollinggan-b/S1-G1-16-color-AB-S1-G4-26-color_fake_B2_masked.png)|
|Attention|x|x|![](./senz3d_full/pg2gan/S3-G1-17-color-AB-S3-G2-30-color_diff_map.png)|![](./senz3d_full/rollinggan-t/S3-G1-17-color-AB-S3-G2-30-color_fake_B2_mask.png)|![](./senz3d_full/rollinggan-b/S3-G1-17-color-AB-S3-G2-30-color_fake_B2_mask.png)|
|Output|![](./senz3d_full/source-image/S3-G1-17-color-AB-S3-G2-30-color_real_A.png)|![](./senz3d_full/ground-truth/S3-G1-17-color-AB-S3-G2-30-color_real_B.png)|![](./senz3d_full/pg2gan/S3-G1-17-color-AB-S3-G2-30-color_fake_B2.png)|![](./senz3d_full/rollinggan-t/S3-G1-17-color-AB-S3-G2-30-color_fake_B2_masked.png)|![](./senz3d_full/rollinggan-b/S3-G1-17-color-AB-S3-G2-30-color_fake_B2_masked.png)|

