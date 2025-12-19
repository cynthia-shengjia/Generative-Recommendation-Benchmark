# kill all zombie nvidia processes for NNI 2.10 BUG
# Usage: ./kill_nvidia.sh

# show all processes that are using nvidia devices
fuser -v /dev/nvidia*

# kill all processes that are using nvidia devices
fuser -v /dev/nvidia* |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sh

# NOTE: you should ensure that all important processes are not using nvidia devices before running this script