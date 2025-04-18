#!/usr/bin/env bash

[ $# -ne 1 ] && echo Need the script name main-stdio.py or main-sse.py && exit

# load conda functions (adjust path to match your install)
source ~/anaconda3/etc/profile.d/conda.sh

conda activate devenv
DEBUG=1 python $1
#main-stdio.py
#main-sse.py
conda deactivate
