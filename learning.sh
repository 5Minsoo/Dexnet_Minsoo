#!/bin/bash
cd /home/minsoo/Dexnet_Minsoo
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "✅ PYTHONPATH가 설정되었습니다: $PYTHONPATH"
datapath="/home/minsoo/Dexnet_Minsoo/grasp_dataset_big1.zarr"
python3 Minsoo_net/model/train.py --data $datapath --thresh 0.002 2>&1 --epoch 40 "$@"
# python3 Minsoo_net/model/train.py --data $datapath --thresh 0.002 --alpha 0.4 0.6 --loss "FL" 2>&1 --epochs $epoch
# python3 Minsoo_net/model/train.py --data $datapath --thresh 0.034 --alpha 0.3 0.7 --loss "FL" 2>&1 --epochs $epoch