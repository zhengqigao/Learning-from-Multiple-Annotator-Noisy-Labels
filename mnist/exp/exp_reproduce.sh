for((i=0;i<=4;i++));
do
  # download our provided data in Google drive, and unzip and put it under the data folder
  python main_train_cvpr.py    --gpu 0 --num_epochs 40 --expert_type 0 --num_experts 3 --hyper_parameter 0.01 >> ./logs/log_cvpr.txt  &
  python main_train_ours.py    --gpu 1 --num_epochs 40 --expert_type 0 --num_experts 3 --num_basis 20 >> ./logs/log_ours.txt  &
  python main_train_wdn.py     --gpu 3 --num_epochs 40 --expert_type 0 --num_experts 3 >> ./logs/log_wdn.txt   &
  python main_train_mjv.py     --gpu 4 --num_epochs 40 --expert_type 0 --num_experts 3 >> ./logs/log_mjv.txt   &
  python main_train_mbem.py    --gpu 5 --num_epochs 20 --expert_type 0 --num_experts 3 --iter 2 >> ./logs/log_ebem.txt   &
  python main_train_w_each.py  --gpu 2 --num_epochs 40 --expert_type 0 --num_experts 3 >> ./logs/log_weach.txt &
  wait
  echo $i experiment finishes;
done