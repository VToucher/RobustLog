CUDA_VISIBLE_DEVICES=1 \
python train.py \
--base_file data/AGNews_train_1000.csv \
--batch_size 64 \
--epochs 1000 \
--lr 1e-3 \
--num_class 4