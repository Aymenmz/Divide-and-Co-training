#!/bin/bash

# Setup PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/workspace}/Divide-and-Co-training"

# Define data directory
data_dir="/workspace/dataset"

# # Select dataset function
# select_dataset() {
#     local choice=$1
#     case ${choice} in
#         0 )
#             echo "cifar10 128 32 300"
#             ;;
#         1 ) 
#             echo "cifar100 128 32 300"
#             ;;
#         2 )
#             echo "imagenet 256 224 120"
#             ;;
#         3 )
#             echo "svhn 128 32 200"
#             ;;
#         * )
#             echo "Invalid choice" >&2
#             exit 1 
#             ;;
#     esac
# }

# # Prompt for dataset choice
# echo -e "\n0  --  cifar10"
# echo "1  --  cifar100"
# echo "2  --  ImageNet"
# echo "3  --  svhn"
# echo -n "Choose the dataset: "
# read dataset_choice

# # Set dataset variables
# read dataset batch_size crop_size epochs <<< $(select_dataset $dataset_choice)
data="${data_dir}/cifar"

# # If ImageNet is selected, change cot_weight_warm_up_epochs
# cot_weight_warm_up_epochs=40
# [ "${dataset}" = "imagenet" ] && cot_weight_warm_up_epochs=60

# Define other variables
arch=resnet50
workers=16
split_factor=1
cot_weight=0.5
is_cot_loss=1
cot_loss_choose='js_divergence'
is_diff_data_train=1
is_cot_weight_warm_up=1
is_div_wd=0
lr_mode=cos
is_cutout=1
erase_p=0.5
is_mixup=1
is_autoaugment=1
world_size=1
rank=0
dist_url='tcp://127.0.0.1:6066'
optimizer=SGD
work_dir="/workspace/Divide-and-Co-training"
resume=None
iters_to_accumulate=1

# Training function
train_model() {
    local num=$1
    local model_arch=$2
    local model_dir="/workspace/Divide-and-Co-training/model/${model_arch}_split${split_factor}_${dataset}_${batch_size}_${num}"
    
    # Check and create model directory
    [ -d "${model_dir}" ] || mkdir -p "${model_dir}"

    # Train the model
    python3 ${work_dir}/train_split_1.py \
        --dist_url ${dist_url} \
        --multiprocessing_distributed \
        --world_size ${world_size} \
        --rank ${rank} \
        --data ${data} \
        --resume ${resume} \
        --dataset "cifar10" \
		--crop_size 32 \
        --batch_size 128 \
        --model_dir ${model_dir} \
        --arch "resnet110" \
        --proc_name ${model_arch}_split${split_factor}_${dataset}_${batch_size}_${num} \
        --split_factor ${split_factor} \
        --is_cot_loss ${is_cot_loss} \
        --cot_weight ${cot_weight} \
        --is_diff_data_train ${is_diff_data_train} \
        --is_cutout ${is_cutout} \
        --erase_p ${erase_p} \
        --lr_mode ${lr_mode} \
        --is_mixup ${is_mixup} \
        --workers ${workers} \
        --cot_loss_choose ${cot_loss_choose} \
        --is_autoaugment ${is_autoaugment} \
        --is_cot_weight_warm_up ${is_cot_weight_warm_up} \
        --is_syncbn 0 \
        --is_div_wd ${is_div_wd} \
        --is_amp 0 \
        --iters_to_accumulate ${iters_to_accumulate} \
        --optimizer ${optimizer}
	

	
}

# Loop to handle multiple training runs
for num in 07; do
    arch=resnet50  # default architecture
    [ "${num}" = "07" ] && arch="se_resnet50_B"
    [ "${num}" = "12" ] && arch="efficientnetb1" && lr_mode="exponential" && optimizer="RMSpropTF"
    
    train_model "${num}" "${arch}"
done

echo "Training Finished!!!"
