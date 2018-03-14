#!/bin/bash

#顶层脚本  训练和测试测试集
stage=0

CONF_FILE='./conf/ctc_model_setting.conf'
TIMIT_dir='/home/fan/Audio_data/TIMIT'

if [ ! -z $1 ]; then
    stage=$1
fi

if [ $stage -le 0 ]; then
    echo =======================================================
    echo "               TIMIT Data Processing                "
    echo =======================================================
    
    bash ./timit_data_prep.sh $TIMIT_dir || exit 1
fi

if [ $stage -le 1 ]; then
    echo ========================================================
    echo "                    Training                          "
    echo ========================================================

    python train.py --conf $CONF_FILE || exit 1;
fi

if [ $stage -le 2 ]; then
    echo ========================================================
    echo "                    Decoding                          " 
    echo ========================================================

    python test.py --conf $CONF_FILE || exit 1
fi

###############################################################
#将字符作为标签训练CTC的声学模型在TIMIT上测试集的识别率为:
#Greedy decoder:    61.4831%
#Beam decoder  :    62.1029%




