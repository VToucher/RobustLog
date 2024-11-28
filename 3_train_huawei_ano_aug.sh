CUDA_VISIBLE_DEVICES=2 \
python train.py \
--base_file data/ano_aug_1010/ano_aug_scene_1_normal_200.csv \
--add_file data/ano_aug_1015/ano_aug_scene_1_mid_intp_dist_min_64.csv \
--batch_size 16 \
--epochs 1000 \
--lr 1e-4 \
--num_class 2