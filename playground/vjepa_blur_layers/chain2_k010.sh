#!/bin/bash
cd /home/wweligam/svd_v2/playground/vjepa_blur_layers
PY=/home/wweligam/home_env/bin/python
R=/scratch/gilbreth/wweligam/vjepa_blur_layers/full2; L=/scratch/gilbreth/wweligam/vjepa_blur_layers/logs
$PY extract_global.py --split test --n 150 --out $R > $L/k010_glob_test.log 2>&1
$PY probes.py --root $R --head mlp > $L/k010_probes_mlp.log 2>&1
$PY attribute.py --root $R --model ft --head mlp --n 80 > $L/k010_attr_ft_mlp.log 2>&1
$PY followup.py --root $R --part patch --model ft --head mlp > $L/k010_fu_ft_mlp.log 2>&1
echo DONE > $L/k010_chain2.done
