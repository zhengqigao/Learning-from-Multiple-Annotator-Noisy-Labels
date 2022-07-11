
# This shell script demonstrates the overall running flow of how we report the results shown in Table I of our paper.

# [Important Note]: Running preprocess.py will (i) first load the original MNIST data, (ii) synthesize multiple experts' labels, (iii) and then save the data to the $pj_dir/data folder. However, due to the randomness of loading data in step (i), when you run this shell script, it is possible yielding completely different results from ours. In short, it is because the 'index_experts' variable defined in preprocess.py will correspond to different images in your and our case due to randomness of data loading. To this end, to justify our results, we provide one set of our generated data so that you could exactly reproduce our results. For more details, please refer to $pj_dir/data/data.md.


cd ../

for((i=0;i<=4;i++));
do
 python ./utils/preprocess.py --seed $i --expert_type 0 --num_experts 3 --expert_threshold 30 & # generate data
 wait
 echo successfully generate data for the $i experiment;
 python main_train_cvpr.py    --gpu 0 --num_epochs 40 --expert_type 0 --num_experts 3 --hyper_parameter 0.01 >> ./logs_final/logvarythre1_cvpr.txt  &
 python main_train_ours.py    --gpu 1 --num_epochs 40 --expert_type 0 --num_experts 3 --num_basis 20 >> ./logs_final/logvarythre1_ours.txt  &
 python main_train_wdn.py     --gpu 3 --num_epochs 40 --expert_type 0 --num_experts 3 >> ./logs_final/logvarythre1_wdn.txt   &
 python main_train_mjv.py     --gpu 4 --num_epochs 40 --expert_type 0 --num_experts 3 >> ./logs_final/logvarythre1_mjv.txt   &
 python main_train_mbem.py    --gpu 5 --num_epochs 20 --expert_type 0 --num_experts 3 --iter 2 >> ./logs_final/logvarythre1_ebem.txt   &
 python main_train_w_each.py  --gpu 2 --num_epochs 40 --expert_type 0 --num_experts 3 >> ./logs_final/logvarythre1_weach.txt &
 wait
 echo $i experiment finishes;
done

for((i=0;i<=4;i++));
do
  python ./utils/preprocess.py --seed $i --expert_type 0 --num_experts 3 --expert_threshold 31 & # generate data
  wait
  echo successfully generate data for the $i experiment;
  python main_train_cvpr.py    --gpu 0 --num_epochs 40 --expert_type 0 --num_experts 3 --hyper_parameter 0.01 >> ./logs_final/logvarythre2_cvpr.txt  &
  python main_train_ours.py    --gpu 1 --num_epochs 40 --expert_type 0 --num_experts 3 --num_basis 20 >> ./logs_final/logvarythre2_ours.txt  &
  python main_train_wdn.py     --gpu 3 --num_epochs 40 --expert_type 0 --num_experts 3 >> ./logs_final/logvarythre2_wdn.txt   &
  python main_train_mjv.py     --gpu 4 --num_epochs 40 --expert_type 0 --num_experts 3 >> ./logs_final/logvarythre2_mjv.txt   &
  python main_train_mbem.py    --gpu 5 --num_epochs 20 --expert_type 0 --num_experts 3 --iter 2 >> ./logs_final/logvarythre2_ebem.txt   &
  python main_train_w_each.py  --gpu 2 --num_epochs 40 --expert_type 0 --num_experts 3 >> ./logs_final/logvarythre2_weach.txt &
  wait
  echo $i experiment finishes;
done

for((i=0;i<=4;i++));
do
  python ./utils/preprocess.py --seed $i --expert_type 0 --num_experts 3 --expert_threshold 32 & # generate data
  wait
  echo successfully generate data for the $i experiment;
  python main_train_cvpr.py    --gpu 0 --num_epochs 40 --expert_type 0 --num_experts 3 --hyper_parameter 0.01 >> ./logs_final/logvarythre3_cvpr.txt  &
  python main_train_ours.py    --gpu 1 --num_epochs 40 --expert_type 0 --num_experts 3 --num_basis 20 >> ./logs_final/logvarythre3_ours.txt  &
  python main_train_wdn.py     --gpu 3 --num_epochs 40 --expert_type 0 --num_experts 3 >> ./logs_final/logvarythre3_wdn.txt   &
  python main_train_mjv.py     --gpu 4 --num_epochs 40 --expert_type 0 --num_experts 3 >> ./logs_final/logvarythre3_mjv.txt   &
  python main_train_mbem.py    --gpu 5 --num_epochs 20 --expert_type 0 --num_experts 3 --iter 2 >> ./logs_final/logvarythre3_ebem.txt   &
  python main_train_w_each.py  --gpu 2 --num_epochs 40 --expert_type 0 --num_experts 3 >> ./logs_final/logvarythre3_weach.txt &
  wait
  echo $i experiment finishes;
done

for((i=0;i<=4;i++));
do
  python ./utils/preprocess.py --seed $i --expert_type 0 --num_experts 3 --expert_threshold 33 & # generate data
  wait
  echo successfully generate data for the $i experiment;
  python main_train_cvpr.py    --gpu 0 --num_epochs 40 --expert_type 0 --num_experts 3 --hyper_parameter 0.01 >> ./logs_final/logvarythre4_cvpr.txt  &
  python main_train_ours.py    --gpu 1 --num_epochs 40 --expert_type 0 --num_experts 3 --num_basis 20 >> ./logs_final/logvarythre4_ours.txt  &
  python main_train_wdn.py     --gpu 3 --num_epochs 40 --expert_type 0 --num_experts 3 >> ./logs_final/logvarythre4_wdn.txt   &
  python main_train_mjv.py     --gpu 4 --num_epochs 40 --expert_type 0 --num_experts 3 >> ./logs_final/logvarythre4_mjv.txt   &
  python main_train_mbem.py    --gpu 5 --num_epochs 20 --expert_type 0 --num_experts 3 --iter 2 >> ./logs_final/logvarythre4_ebem.txt   &
  python main_train_w_each.py  --gpu 2 --num_epochs 40 --expert_type 0 --num_experts 3 >> ./logs_final/logvarythre4_weach.txt &
  wait
  echo $i experiment finishes;
done

for((i=0;i<=4;i++));
do
  python ./utils/preprocess.py --seed $i --expert_type 0 --num_experts 3 --expert_threshold 34 & # generate data
  wait
  echo successfully generate data for the $i experiment;
  python main_train_cvpr.py    --gpu 0 --num_epochs 40 --expert_type 0 --num_experts 3 --hyper_parameter 0.01 >> ./logs_final/logvarythre5_cvpr.txt  &
  python main_train_ours.py    --gpu 1 --num_epochs 40 --expert_type 0 --num_experts 3 --num_basis 20 >> ./logs_final/logvarythre5_ours.txt  &
  python main_train_wdn.py     --gpu 3 --num_epochs 40 --expert_type 0 --num_experts 3 >> ./logs_final/logvarythre5_wdn.txt   &
  python main_train_mjv.py     --gpu 4 --num_epochs 40 --expert_type 0 --num_experts 3 >> ./logs_final/logvarythre5_mjv.txt   &
  python main_train_mbem.py    --gpu 5 --num_epochs 20 --expert_type 0 --num_experts 3 --iter 2 >> ./logs_final/logvarythre5_ebem.txt   &
  python main_train_w_each.py  --gpu 2 --num_epochs 40 --expert_type 0 --num_experts 3 >> ./logs_final/logvarythre5_weach.txt &
  wait
  echo $i experiment finishes;
done

for((i=0;i<=4;i++));
do
  python ./utils/preprocess.py --seed $i --expert_type 0 --num_experts 3 --expert_threshold 35 & # generate data
  wait
  echo successfully generate data for the $i experiment;
  python main_train_cvpr.py    --gpu 0 --num_epochs 40 --expert_type 0 --num_experts 3 --hyper_parameter 0.01 >> ./logs_final/logvarythre6_cvpr.txt  &
  python main_train_ours.py    --gpu 1 --num_epochs 40 --expert_type 0 --num_experts 3 --num_basis 20 >> ./logs_final/logvarythre6_ours.txt  &
  python main_train_wdn.py     --gpu 3 --num_epochs 40 --expert_type 0 --num_experts 3 >> ./logs_final/logvarythre6_wdn.txt   &
  python main_train_mjv.py     --gpu 4 --num_epochs 40 --expert_type 0 --num_experts 3 >> ./logs_final/logvarythre6_mjv.txt   &
  python main_train_mbem.py    --gpu 5 --num_epochs 20 --expert_type 0 --num_experts 3 --iter 2 >> ./logs_final/logvarythre6_ebem.txt   &
  python main_train_w_each.py  --gpu 2 --num_epochs 40 --expert_type 0 --num_experts 3 >> ./logs_final/logvarythre6_weach.txt &
  wait
  echo $i experiment finishes;
done