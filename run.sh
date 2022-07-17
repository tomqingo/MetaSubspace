echo "5-way 1-shot cifarfs maml"
python cifarfs_train_subspace.py --epoch 20000 --n_way 5 --num_subspace 40 --k_spt 1 --k_qry 15 --log_dir ./cifarfssub/5way1shot/maml/lab_1

echo "5-way 5-shot cifarfs maml"
python cifarfs_train_subspace.py --epoch 20000 --n_way 5 --num_subspace 40 --k_spt 5 --k_qry 15 --log_dir ./cifarfssub/5way5shot/maml/lab_1


