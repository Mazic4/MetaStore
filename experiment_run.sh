#bin/bash!

logger_path="./experiment_result_test_model_single_query_test"
device_number=0
analyze_file_path="./main.py"

mkdir -p ${logger_path}

#for experiment_type in "model"
#do
#for method in "ted" "naive"
#do
##for precision_mode in "quant"
##do
##for float_precision in "torch.qint8" "torch.quint8" "torch.qint32"
##do
#for dataset in "cifar10"
#do
#for num_samples in 2000
#do
#for num_query in 5
#do
#  echo $method
#  echo $dataset
#  echo $num_samples
#  CUDA_VISIBLE_DEVICES=${device_number} python3 $analyze_file_path --output_path ${logger_path} --experiment_type $experiment_type --method $method --dataset $dataset --num_samples $num_samples --num_query $num_query > $logger_path/analyze_${dataset}_${experiment_type}_${method}_${num_samples}_${num_query}.out
##done
##done
#done
#done
#done
#done
#done

experiment_type="batch"

for method in "recon" "half"
do
for dataset in "AGNews" "cifar10"
do
for num_samples in 100
do
for num_query in 1
do
  echo $experiment_type
  echo $method
  echo $dataset
  echo $num_query
  CUDA_VISIBLE_DEVICES=${device_number} python3 $analyze_file_path --output_path ${logger_path}_${num_samples} --dataset $dataset --experiment_type $experiment_type --num_samples $num_samples --num_query $num_query --method $method> $logger_path/analyze_${dataset}_${experiment_type}_${num_samples}_${num_query}.out
done
done
done
done

#experiment_type="model"
#
#for method in "naive" "reproduce" "ted"
#do
#for dataset in "imagenet"
#do
#for num_samples in 500 1000 2000
#do
#for num_query in 10
#do
#  echo $experiment_type
#  echo $method
#  echo $dataset
#  echo $num_query
#  CUDA_VISIBLE_DEVICES=${device_number} python3 $analyze_file_path --output_path ${logger_path}_${num_samples} --dataset $dataset --experiment_type $experiment_type --num_samples $num_samples --num_query $num_query --method ${method}> $logger_path/analyze_${dataset}_${experiment_type}_${num_samples}_${num_query}.out
#done
#done
#done
#done

exit 0