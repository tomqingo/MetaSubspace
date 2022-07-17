echo "10-way 1-shot cifarfs maml"
python cifarfs_train_subspace.py --epoch 20000 --n_way 10 --num_subspace 40 --k_spt 1 --k_qry 15 --log_dir ./cifarfssub/10way1shot/maml/lab_1

echo "10-way 5-shot cifarfs maml"
python cifarfs_train_subspace.py --epoch 20000 --n_way 10 --num_subspace 40 --k_spt 5 --k_qry 15 --log_dir ./cifarfssub/10way5shot/maml/lab_1
