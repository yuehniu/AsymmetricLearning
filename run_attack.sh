gpu="0"
lr=0.002
noise=0.4
logdir="./log/attack/resnet18-noise${noise}-lr${lr}"
if [ -d $logdir ]; then
  rm -r $logdir
fi

CUDA_VISIBLE_DEVICES=$gpu python miAttacker.py \
  --log-dir=$logdir --lr=${lr} --with-residual --noise=$noise