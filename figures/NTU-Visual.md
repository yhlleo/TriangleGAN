
### NTU Hand Gesture

We randomly selected several samples.

 - Part 

|Source Image|Cond Map|Ground Truth|GANimation|StarGAN|PG2|Gesture|Ours|
|----|----|----|----|----|----|----|----|
|![](./ntu_part/source-image/P1-G1-3-AB-P1-G5-6_real_A.png)|![](./ntu_part/cond-map/P1-G1-3-AB-P1-G5-6_cond_B.png)|![](./ntu_part/ground-truth/P1-G1-3-AB-P1-G5-6_real_B.png)|![](./ntu_part/ganimation/P1-G1-3-AB-P1-G5-6_fake_B_masked.png)|![](./ntu_part/stargan/P1-G1-3-AB-P1-G5-6_fake_B.png)|![](./ntu_part/pg2gan/P1-G1-3-AB-P1-G5-6_fake_B2.png)|![](./ntu_part/gesturegan-raw/P1-G1-3-AB-P1-G5-6_fake_B.png)|![](./ntu_part/rollinggan/P1-G1-3-AB-P1-G5-6_fake_B2_masked.png)|
|![](./ntu_part/source-image/P2-G1-4-AB-P2-G2-7_real_A.png)|![](./ntu_part/cond-map/P2-G1-4-AB-P2-G2-7_cond_B.png)|![](./ntu_part/ground-truth/P2-G1-4-AB-P2-G2-7_real_B.png)|![](./ntu_part/ganimation/P2-G1-4-AB-P2-G2-7_fake_B_masked.png)|![](./ntu_part/stargan/P2-G1-4-AB-P2-G2-7_fake_B.png)|![](./ntu_part/pg2gan/P2-G1-4-AB-P2-G2-7_fake_B2.png)|![](./ntu_part/gesturegan-raw/P2-G1-4-AB-P2-G2-7_fake_B.png)|![](./ntu_part/rollinggan/P2-G1-4-AB-P2-G2-7_fake_B2_masked.png)|
|![](./ntu_part/source-image/P3-G8-9-AB-P3-G3-1_real_A.png)|![](./ntu_part/cond-map/P3-G8-9-AB-P3-G3-1_cond_B.png)|![](./ntu_part/ground-truth/P3-G8-9-AB-P3-G3-1_real_B.png)|![](./ntu_part/ganimation/P3-G8-9-AB-P3-G3-1_fake_B_masked.png)|![](./ntu_part/stargan/P3-G8-9-AB-P3-G3-1_fake_B.png)|![](./ntu_part/pg2gan/P3-G8-9-AB-P3-G3-1_fake_B2.png)|![](./ntu_part/gesturegan-raw/P3-G8-9-AB-P3-G3-1_fake_B.png)|![](./ntu_part/rollinggan/P3-G8-9-AB-P3-G3-1_fake_B2_masked.png)|
|![](./ntu_part/source-image/P4-G6-1-AB-P4-G9-10_real_A.png)|![](./ntu_part/cond-map/P4-G6-1-AB-P4-G9-10_cond_B.png)|![](./ntu_part/ground-truth/P4-G6-1-AB-P4-G9-10_real_B.png)|![](./ntu_part/ganimation/P4-G6-1-AB-P4-G9-10_fake_B_masked.png)|![](./ntu_part/stargan/P4-G6-1-AB-P4-G9-10_fake_B.png)|![](./ntu_part/pg2gan/P4-G6-1-AB-P4-G9-10_fake_B2.png)|![](./ntu_part/gesturegan-raw/P4-G6-1-AB-P4-G9-10_fake_B.png)|![](./ntu_part/rollinggan/P4-G6-1-AB-P4-G9-10_fake_B2_masked.png)|
|![](./ntu_part/source-image/P5-G3-5-AB-P5-G10-8_real_A.png)|![](./ntu_part/cond-map/P5-G3-5-AB-P5-G10-8_cond_B.png)|![](./ntu_part/ground-truth/P5-G3-5-AB-P5-G10-8_real_B.png)|![](./ntu_part/ganimation/P5-G3-5-AB-P5-G10-8_fake_B_masked.png)|![](./ntu_part/stargan/P5-G3-5-AB-P5-G10-8_fake_B.png)|![](./ntu_part/pg2gan/P5-G3-5-AB-P5-G10-8_fake_B2.png)|![](./ntu_part/gesturegan-raw/P5-G3-5-AB-P5-G10-8_fake_B.png)|![](./ntu_part/rollinggan/P5-G3-5-AB-P5-G10-8_fake_B2_masked.png)|
|![](./ntu_part/source-image/P7-G4-1-AB-P7-G1-8_real_A.png)|![](./ntu_part/cond-map/P7-G4-1-AB-P7-G1-8_cond_B.png)|![](./ntu_part/ground-truth/P7-G4-1-AB-P7-G1-8_real_B.png)|![](./ntu_part/ganimation/P7-G4-1-AB-P7-G1-8_fake_B_masked.png)|![](./ntu_part/stargan/P7-G4-1-AB-P7-G1-8_fake_B.png)|![](./ntu_part/pg2gan/P7-G4-1-AB-P7-G1-8_fake_B2.png)|![](./ntu_part/gesturegan-raw/P7-G4-1-AB-P7-G1-8_fake_B.png)|![](./ntu_part/rollinggan/P7-G4-1-AB-P7-G1-8_fake_B2_masked.png)|

OurGAN uses triangle as conditional map:

|1|2|3|4|5|6|
|----|----|----|----|----|----|
|![](./ntu_part/rollinggan-triangle/P1-G1-3-AB-P1-G5-6_fake_B2_masked.png)|![](./ntu_part/rollinggan-triangle/P2-G1-4-AB-P2-G2-7_fake_B2_masked.png)|![](./ntu_part/rollinggan-triangle/P3-G8-9-AB-P3-G3-1_fake_B2_masked.png)|![](./ntu_part/rollinggan-triangle/P4-G6-1-AB-P4-G9-10_fake_B2_masked.png)|![](./ntu_part/rollinggan-triangle/P5-G3-5-AB-P5-G10-8_fake_B2_masked.png)|![](./ntu_part/rollinggan-triangle/P7-G4-1-AB-P7-G1-8_fake_B2_masked.png)|

 - Full

|Source Image|Cond Map|Ground Truth|PG2|Gesture|Ours|
|----|----|----|----|----|----|
|![](./ntu_full/source-image/P1-G2-6-AB-P1-G9-9_real_A.png)|![](./ntu_full/cond-map/P1-G2-6-AB-P1-G9-9_cond_B.png)|![](./ntu_full/ground-truth/P1-G2-6-AB-P1-G9-9_real_B.png)|![](./ntu_full/pg2gan/P1-G2-6-AB-P1-G9-9_fake_B1.png)|![](./ntu_full/gesturegan-raw/P1-G2-6-AB-P1-G9-9_fake_B.png)|![](./ntu_full/rollinggan/P1-G2-6-AB-P1-G9-9_fake_B2_masked.png)|
|![](./ntu_full/source-image/P4-G2-5-AB-P4-G6-10_real_A.png)|![](./ntu_full/cond-map/P4-G2-5-AB-P4-G6-10_cond_B.png)|![](./ntu_full/ground-truth/P4-G2-5-AB-P4-G6-10_real_B.png)|![](./ntu_full/pg2gan/P4-G2-5-AB-P4-G6-10_fake_B1.png)|![](./ntu_full/gesturegan-raw/P4-G2-5-AB-P4-G6-10_fake_B.png)|![](./ntu_full/rollinggan/P4-G2-5-AB-P4-G6-10_fake_B2_masked.png)|
|![](./ntu_full/source-image/P5-G2-10-AB-P5-G4-9_real_A.png)|![](./ntu_full/cond-map/P5-G2-10-AB-P5-G4-9_cond_B.png)|![](./ntu_full/ground-truth/P5-G2-10-AB-P5-G4-9_real_B.png)|![](./ntu_full/pg2gan/P5-G2-10-AB-P5-G4-9_fake_B1.png)|![](./ntu_full/gesturegan-raw/P5-G2-10-AB-P5-G4-9_fake_B.png)|![](./ntu_full/rollinggan/P5-G2-10-AB-P5-G4-9_fake_B2_masked.png)|






