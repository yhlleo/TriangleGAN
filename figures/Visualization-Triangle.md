### Rolling with Triangle & Boundary

 Visualizing the distribution of extracted features

 - NTU - full

 |Source Image|Cond Map|Ground Truth| w/o rolling| w/ rolling|
|----|----|----|----|----|
|![](./ntu_full/source-image/P1-G2-6-AB-P1-G9-8_real_A.png)|![](./ntu_full/cond-map/P1-G2-6-AB-P1-G9-8_cond_B.png)|![](./ntu_full/ground-truth/P1-G2-6-AB-P1-G9-8_real_B.png)|![](./ntu_full/rollinggan-kp/P1-G2-6-AB-P1-G9-8_fake_B1_masked.png)|![](./ntu_full/rollinggan-kp/P1-G2-6-AB-P1-G9-8_fake_B2_masked.png)|
|![](./ntu_full/source-image/P2-G5-7-AB-P2-G9-9_real_A.png)|![](./ntu_full/cond-map/P2-G5-7-AB-P2-G9-9_cond_B.png)|![](./ntu_full/ground-truth/P2-G5-7-AB-P2-G9-9_real_B.png)|![](./ntu_full/rollinggan-kp/P2-G5-7-AB-P2-G9-9_fake_B1_masked.png)|![](./ntu_full/rollinggan-kp/P2-G5-7-AB-P2-G9-9_fake_B2_masked.png)|
|![](./ntu_full/source-image/P10-G1-4-AB-P10-G3-9_real_A.png)|![](./ntu_full/cond-map/P10-G1-4-AB-P10-G3-9_cond_B.png)|![](./ntu_full/ground-truth/P10-G1-4-AB-P10-G3-9_real_B.png)|![](./ntu_full/rollinggan-kp/P10-G1-4-AB-P10-G3-9_fake_B1_masked.png)|![](./ntu_full/rollinggan-kp/P10-G1-4-AB-P10-G3-9_fake_B2_masked.png)|


 - Senz3D - full - Triangle


|Source Image|Cond Map|Ground Truth | w/o rolling| w/ rolling|
|----|----|----|----|----|
|![](./senz3d_full/source-image/S1-G1-16-color-AB-S1-G4-26-color_real_A.png)|![](./senz3d_full/cond-map-t/S1-G1-16-color-AB-S1-G4-26-color_cond_B.png)|![](./senz3d_full/ground-truth/S1-G1-16-color-AB-S1-G4-26-color_real_B.png)|![](./senz3d_full/rollinggan-t-kp/S1-G1-16-color-AB-S1-G4-26-color_fake_B1_masked.png)|![](./senz3d_full/rollinggan-t-kp/S1-G1-16-color-AB-S1-G4-26-color_fake_B2_masked.png)|
|![](./senz3d_full/source-image/S1-G4-18-color-AB-S1-G10-26-color_real_A.png)|![](./senz3d_full/cond-map-t/S1-G4-18-color-AB-S1-G10-26-color_cond_B.png)|![](./senz3d_full/ground-truth/S1-G4-18-color-AB-S1-G10-26-color_real_B.png)|![](./senz3d_full/rollinggan-t-kp/S1-G4-18-color-AB-S1-G10-26-color_fake_B1_masked.png)|![](./senz3d_full/rollinggan-t-kp/S1-G4-18-color-AB-S1-G10-26-color_fake_B2_masked.png)|
|![](./senz3d_full/source-image/S3-G1-17-color-AB-S3-G2-30-color_real_A.png)|![](./senz3d_full/cond-map-t/S3-G1-17-color-AB-S3-G2-30-color_cond_B.png)|![](./senz3d_full/ground-truth/S3-G1-17-color-AB-S3-G2-30-color_real_B.png)|![](./senz3d_full/rollinggan-t-kp/S3-G1-17-color-AB-S3-G2-30-color_fake_B1_masked.png)|![](./senz3d_full/rollinggan-t-kp/S3-G1-17-color-AB-S3-G2-30-color_fake_B2_masked.png)|
|![](./senz3d_full/source-image/S2-G2-22-color-AB-S2-G9-30-color_real_A.png)|![](./senz3d_full/cond-map-t/S2-G2-22-color-AB-S2-G9-30-color_cond_B.png)|![](./senz3d_full/ground-truth/S2-G2-22-color-AB-S2-G9-30-color_real_B.png)|![](./senz3d_full/rollinggan-t-kp/S2-G2-22-color-AB-S2-G9-30-color_fake_B1_masked.png)|![](./senz3d_full/rollinggan-t-kp/S2-G2-22-color-AB-S2-G9-30-color_fake_B2_masked.png)|

 - Senz3D - full - Boundary


|Source Image|Cond Map|Ground Truth | w/o rolling| w/ rolling|
|----|----|----|----|----|
|![](./senz3d_full/source-image/S1-G1-16-color-AB-S1-G4-26-color_real_A.png)|![](./senz3d_full/cond-map-b/S1-G1-16-color-AB-S1-G4-26-color_cond_B.png)|![](./senz3d_full/ground-truth/S1-G1-16-color-AB-S1-G4-26-color_real_B.png)|![](./senz3d_full/rollinggan-b-kp/S1-G1-16-color-AB-S1-G4-26-color_fake_B1_masked.png)|![](./senz3d_full/rollinggan-b-kp/S1-G1-16-color-AB-S1-G4-26-color_fake_B2_masked.png)|
|![](./senz3d_full/source-image/S1-G4-18-color-AB-S1-G10-26-color_real_A.png)|![](./senz3d_full/cond-map-b/S1-G4-18-color-AB-S1-G10-26-color_cond_B.png)|![](./senz3d_full/ground-truth/S1-G4-18-color-AB-S1-G10-26-color_real_B.png)|![](./senz3d_full/rollinggan-b-kp/S1-G4-18-color-AB-S1-G10-26-color_fake_B1_masked.png)|![](./senz3d_full/rollinggan-b-kp/S1-G4-18-color-AB-S1-G10-26-color_fake_B2_masked.png)|
|![](./senz3d_full/source-image/S3-G1-17-color-AB-S3-G2-30-color_real_A.png)|![](./senz3d_full/cond-map-b/S3-G1-17-color-AB-S3-G2-30-color_cond_B.png)|![](./senz3d_full/ground-truth/S3-G1-17-color-AB-S3-G2-30-color_real_B.png)|![](./senz3d_full/rollinggan-b-kp/S3-G1-17-color-AB-S3-G2-30-color_fake_B1_masked.png)|![](./senz3d_full/rollinggan-b-kp/S3-G1-17-color-AB-S3-G2-30-color_fake_B2_masked.png)|
|![](./senz3d_full/source-image/S2-G2-22-color-AB-S2-G9-30-color_real_A.png)|![](./senz3d_full/cond-map-b/S2-G2-22-color-AB-S2-G9-30-color_cond_B.png)|![](./senz3d_full/ground-truth/S2-G2-22-color-AB-S2-G9-30-color_real_B.png)|![](./senz3d_full/rollinggan-b-kp/S2-G2-22-color-AB-S2-G9-30-color_fake_B1_masked.png)|![](./senz3d_full/rollinggan-b-kp/S2-G2-22-color-AB-S2-G9-30-color_fake_B2_masked.png)|