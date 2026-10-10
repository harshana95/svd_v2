#!/bin/bash
cd /home/wweligam/svd_v2/playground/vjepa_blur_layers
PY=/home/wweligam/home_env/bin/python
R=/scratch/gilbreth/wweligam/vjepa_blur_layers/full2
L=/scratch/gilbreth/wweligam/vjepa_blur_layers/logs
mkdir -p $R $L
$PY extract.py --split test --n 200 --out $R > $L/k010_extract_test.log 2>&1
$PY followup.py --root $R --part movement > $L/k010_movement.log 2>&1
until [ -f $R/train/meta.npz ]; do sleep 20; done
$PY attribute.py --root $R --model ft --n 80 > $L/k010_attr_ft.log 2>&1
$PY followup.py --root $R --part patch --model ft > $L/k010_fu_ft.log 2>&1
echo CHAIN_DONE > $L/k010_chain.done
