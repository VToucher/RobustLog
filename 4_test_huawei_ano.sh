CUDA_VISIBLE_DEVICES=1 \
python test.py \
--base_file data/ano_aug_1007/ano_aug_scene_1_normal_1007.csv \
--checkpoint model_outputs/aug_scene_1_fix_test_1k3_noraml_1e-4/20241008_084929_base/best_epoch_74.bin \
--batch_size 1 \
--num_class 2