#!/bin/bash
cd /home/wweligam/svd_v2/playground/vjepa_blur_layers
PY=/home/wweligam/home_env/bin/python
R=/scratch/gilbreth/wweligam/vjepa_blur_layers/full2; L=/scratch/gilbreth/wweligam/vjepa_blur_layers/logs
$PY extract_global.py --split train --n 300 --out $R > $L/k012_glob_train.log 2>&1
until [ -f $R/glob_test/meta.npz ]; do sleep 20; done
$PY heads_global.py --root $R > $L/k012_heads_global.log 2>&1
$PY attribute.py --root $R --model frozen --head mlp --n 80 > $L/k012_attr_frozen_mlp.log 2>&1
$PY followup.py --root $R --part patch --model frozen --head mlp > $L/k012_fu_frozen_mlp.log 2>&1
echo DONE > $L/k012_chain2.done
