CUDA_VISIBLE_DEVICES=1 \
python test.py \
--base_file data/ano_aug_1007/ano_aug_scene_1_normal_1007.csv \
--extra_test_file data/ano_mine_1016/all_scene_normal_1016.csv \
--checkpoint model_outputs/20241015_084448_double_cfg_ft_best/best_epoch_43.bin \
--split extra_test \
--batch_size 463 \
--num_class 2