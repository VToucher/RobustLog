CUDA_VISIBLE_DEVICES=1 \
python train.py \
--base_file data/ano_aug_1007/ano_aug_scene_1_normal_1007.csv \
--batch_size 16 \
--epochs 1000 \
--lr 1e-4 \
--num_class 2