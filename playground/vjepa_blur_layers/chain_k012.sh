#!/bin/bash
cd /home/wweligam/svd_v2/playground/vjepa_blur_layers
PY=/home/wweligam/home_env/bin/python
R=/scratch/gilbreth/wweligam/vjepa_blur_layers/full2
L=/scratch/gilbreth/wweligam/vjepa_blur_layers/logs
mkdir -p $R $L
$PY extract.py --split train --n 400 --out $R > $L/k012_extract_train.log 2>&1
until [ -f $R/test/meta.npz ]; do sleep 20; done
$PY probes.py --root $R > $L/k012_probes.log 2>&1
$PY attribute.py --root $R --model frozen --n 80 > $L/k012_attr_frozen.log 2>&1
$PY followup.py --root $R --part patch --model frozen > $L/k012_fu_frozen.log 2>&1
echo CHAIN_DONE > $L/k012_chain.done
