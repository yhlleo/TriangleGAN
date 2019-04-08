DATAROOT=./datasets
DATA_NAME=ntu_dataset
RESTORE_NAME=ntu_part_trianglegan_skeleton
MODEL_MODE=triangle_gan
DATASET_MODE=gesture_part
COND_TYPE=skeleton
IMAGE_NAME=images-part
TRAIN_NAME=train.part.lst

INPUT_NC=3
OUTPUT_NC=3
VDIM=10 # gesture
COND_DIM=1
NGF=64
BATCH_SIZE=4
LOAD_SIZE=256
NITER=10
NITER_DECAY=10
CONTINUE_TRAIN=0

NORM=instance
CHECKPOINTS_DIR=./checkpoints

GPU_IDS=0
python3.5 ./train.py \
  --dataroot ${DATAROOT} \
  --data_name ${DATA_NAME} \
  --name ${RESTORE_NAME} \
  --model ${MODEL_MODE} \
  --dataset_mode ${DATASET_MODE} \
  --cond_type ${COND_TYPE} \
  --image_name ${IMAGE_NAME} \
  --train_name ${TRAIN_NAME} \
  --input_nc ${INPUT_NC} \
  --output_nc ${OUTPUT_NC} \
  --ngf ${NGF} \
  --vdim ${VDIM} \
  --cond_dim ${COND_DIM} \
  --batch_size ${BATCH_SIZE} \
  --load_size ${LOAD_SIZE} \
  --norm ${NORM} \
  --netG resnet_6blocks \
  --checkpoints_dir ${CHECKPOINTS_DIR} \
  --gpu_ids ${GPU_IDS} \
  --niter ${NITER} \
  --niter_decay ${NITER_DECAY} \
  --continue_train ${CONTINUE_TRAIN}