CUDA_VISIBLE_DEVICES=5 \
python train.py \
--base_file data/ano_aug_1010/ano_aug_scene_1_normal_200.csv \
--add_file data/ano_aug_1015/ano_aug_scene_1_double_cfg_dist_min_64.csv \
--pretrain_model model_outputs/aug_scene_1_fix_test_1k3_noraml_1e-4/20241015_034352_double_cfg_dist_add/best_epoch_737.bin \
--batch_size 16 \
--epochs 1000 \
--lr 1e-5 \
--num_class 2