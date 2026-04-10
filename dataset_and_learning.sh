#!/bin/bash

# 1. 현재 디렉토리를 PYTHONPATH에 추가 (Dexnet_Minsoo 폴더 내에서 실행 기준)
cd /home/minsoo/Dexnet_Minsoo
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "✅ PYTHONPATH가 설정되었습니다: $PYTHONPATH"

# 2. 멀티 오브젝트 데이터셋 생성 실행
echo "🚀 데이터셋 생성을 시작합니다..."
python3 Minsoo_net/database/dataset_generation_multy_objects.py

# 3. 데이터셋 생성 성공 여부 확인 후 학습 시작
if [ $? -eq 0 ]; then
    echo "✅ 데이터셋 생성 완료. 모델 학습을 시작합니다..."
    python3 Minsoo_net/model/train.py --thresh 0.002 --loss "CE" 2>&1
else
    echo "❌ 데이터셋 생성 중 오류가 발생하여 학습을 중단합니다."
    exit 1
fi