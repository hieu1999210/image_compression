#!/bin/bash
set -x

TEMP_FILE=.tmp
PROJECT_NAME=$(basename "`pwd`")
echo "Run training for project $PROJECT_NAME"

/bin/cat <<EOM >$TEMP_FILE
*__pycache__*
cache/*
logs/*
weights/*
rsna_subs/*
notebooks/*
*.txt
*.ipynb
*runs/*
apex/*
*MICCAI*

*egg*
*.pth
*.csv
EOM

if [ "$1" == "workstation" ]; then
    echo "Push code to workstation (1x2080Ti)"
    REMOTE_HOME="/mnt/DATA/fixmatch"

elif [ "$1" == "laptop" ]; then
    echo "Push code to workstation (1x1050Ti)"
    REMOTE_HOME="/mnt/DATA/personal_projects/Fixmatch"

elif [ "$1" == "tung" ]; then
    echo "Push code to workstation (1x2080Ti)"
    REMOTE_HOME="/home/tungthanhlee/hieu/Fixmatch"

elif [ "$1" == "medical" ]; then
    echo "Push code to medical (4x2080Ti)"
    REMOTE_HOME="/home/dev/hieunt/Fixmatch"

elif [ "$1" == "dgx3" ]; then
    echo "Push code to dgx3 (8xV100)"
    REMOTE_HOME="/home/vinbdi/hieu/Fixmatch"

else
    echo "Unknown server"
    exit
fi
# config jump server
# if [ "$1" == "medical" ] || [ "$1" == "dgx1" ]; then
#     JUMP=""
# else
#     echo "dgx2 server, need to use dgx1 as proxy"
#     JUMP="-J dgx1"
# if [ "$1" == "dgx1" ]; then
#     JUMP="-J dev@localhost:69"
# elif [ "$1" == "dgx2" ]; then 
#     JUMP="-J dev@localhost:69,dev@10.100.53.77:8008"
# fi
# push code to server
# rsync -vr -P -e "ssh -p$PORT $JUMP" --exclude-from $TEMP_FILE "$PWD" $USER@$IP:$REMOTE_HOME/nhan/
rsync -vr -P --exclude-from $TEMP_FILE "$PWD/" $1:$REMOTE_HOME
# rsync -vr -P --exclude-from $TEMP_FILE tungthanhlee@10.208.209.66:/home/tungthanhlee/brats/ /home/ad/mammo_code/aNhan/
# rsync -vr -P --exclude-from $TEMP_FILE nhannt@10.208.209.77:/home/nhannt/workspace/lib/ /home/ad/mammo_code/aNhan/
# rsync -vr -P --exclude-from $TEMP_FILE /mnt/DATA/mammo/source/ workstation:/home/ad/mammo_detection/source/
# pull model weights and log files from server
# rsync -vr -P -e "ssh -p$PORT $JUMP" $USER@$IP:$REMOTE_HOME/nhan/$PROJECT_NAME/logs/*.txt ./logs/
# rsync -vr -P -e "ssh -p$PORT $JUMP" $USER@$IP:$REMOTE_HOME/nhan/$PROJECT_NAME/weights/best* ./weights/
# remove temp. file
rm $TEMP_FILE