#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/lib
export CUDA_VISIBLE_DEVICES="0"
# run the tensorboard command periodly
echo ""
echo "1 -- splitnet directory"
echo -n "choose the directory: "
read dir_choose

echo ""
echo -n "input the port:"
read port
logdir="../model/se_resnet50_B_split1___07" 

# set the logdir
# case ${dir_choose} in
# 	1 )
# 		logdir="/workspace/models"
# 		;;
# 	* )
# 		echo "The choice of the directory is illegal!" se_resnet50_B_split1___07 http://172.17.0.2/
# 		exit 1
# 		;;
# esac

# docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' container_id 74a3186c073b


# sleep time, hours
sleep_t=6
times=0

# while loop
while true
do
	# https://stackoverflow.com/questions/40106949/unable-to-open-tensorboard-in-browser
	tensorboard --bind_all --logdir=${logdir} --port=${port} &
	last_pid=$!
	sleep ${sleep_t}h
	kill -9 ${last_pid}
	times=`expr ${times} + 1`
	echo "Restart tensorboard ${times} times."
done

echo "tensorboard is stopped!"
