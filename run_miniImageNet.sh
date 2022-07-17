echo "5-way 1-shot miniImageNet meta-subspace"
python miniimagenet_train_subspace.py --epoch 100 --n_way 5 --num_subspace 40 --k_spt 1 --k_qry 15 --log_dir ./save_final/miniImageNet/subspace/5way1shot

