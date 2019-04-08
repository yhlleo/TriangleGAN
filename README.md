## TriangleGAN

A new gesture-to-gesture translation framework.


### 1.Dataset preparing

 - Original Dataset
   - [NTU Hand Gesture Dataset](https://drive.google.com/file/d/1f8tUHid1KmnwbgskGMXmobOxMfbxIgHM/view)
   - [Senz3d Gesture Dataset](http://lttm.dei.unipd.it/downloads/gesture/#senz3d)

[More details >>>](./datasets/README.md)

### 2.Installation

For Conda users, we provide a installation script `./scripts/conda_deps.sh`. Alternatively, you can create a new Conda environment using `conda env create -f environment.yml`.

### 3.Train/Test

 1.Download dataset and copy them into `./datasets`
 
 2.Modify the scripts to train/test:

```
sh ./scripts/train_trianglegan_ntu.sh


sh ./scripts/test_trianglegan_ntu.sh
```

### 4.Visual Results

[More Details >>>](./figures/README.md)