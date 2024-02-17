#bin/bash!

device_number=0
analyze_file_path="./main.py"
config_path="./config.yaml"

mkdir -p ${logger_path}

for method in "ted" "half" "recon" "naive"
do
for dataset in "cifar10" "AGNews" "imagenet"
do
for num_samples in 500 1000 2000
do
for num_query in 20
do
  echo $method
  echo $dataset
  echo $num_samples
  if [ "$dataset" = "cifar10" ]; then
    target_model="VGG16"
  elif [ "$dataset" = "AGNews" ]; then
    target_model="Bert"
  elif [ "$dataset" = "imagenet" ]; then
    target_model="ResNet50"
  fi
  CUDA_VISIBLE_DEVICES=${device_number} python3 $analyze_file_path --config $config_path --method $method --target_model $target_model \
  --data $dataset --num_analyzed_samples $num_samples --num_query $num_query
done
done
done
done


exit 0