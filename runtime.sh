python=python

# --> training hparams
dataset=cifar100
model=resnet18
bs=32
lr=0.1
wd=2e-4
regcoeff=8e-4
regmethod="l2"
gpu="0"

svd=1
rank=4
dct=1
quant=1
noise=0.0
alpha=1.0
moreopts=" "

logdir="./log/${dataset}_${model}/orig/lr${lr}_${regmethod}_${wd}"
if [ $dataset == "imagenet"  ]; then
  moreopts+=" --epochs=100 --batch-size=512"
fi

if [ $svd -eq 1 ]; then
  moreopts+=" --svd --rank=${rank}"
  if [ $regmethod == "l2" ]; then
    logdir="./log/${dataset}_${model}/svd/lr${lr}_${regmethod}_${wd}_svd${rank}_botneck_3x3_bn_expand8"
  else
    logdir="./log/${dataset}_${model}/svd/lr${lr}_${regmethod}_${wd}-${regcoeff}_svd${rank}_botneck_3x3_bn_expand8"
  fi
fi

if [ $dct -eq 1 ]; then
  moreopts+=" --dct"
  if [ $regmethod == "l2" ]; then
    logdir="./log/${dataset}_${model}/dct/lr${lr}_${regmethod}_${wd}_svd${rank}_dct28-14-prelu-bb_expand12"
  else
    logdir="./log/${dataset}_${model}/dct/lr${lr}_${regmethod}_${wd}-${regcoeff}_svd${rank}_dct28-14-prelu-bb_expand12"
  fi
fi

if [ $quant -eq 1 ]; then
  moreopts+="  --quant --noise=${noise}"
  if [ $regmethod == "l2" ]; then
    logdir="./log/runtime/${dataset}_${model}/asym/svd${rank}_dct32-16_M2large"
  else
    logdir="./log/runtime/${dataset}_${model}/asym/svd${rank}_dct32-16_M2large"
  fi
fi

if [ -d $logdir ]; then
    rm -r $logdir
fi
mkdir -p $logdir

CUDA_VISIBLE_DEVICES=$gpu $python runtime.py  ${moreopts} \
  --cuda \
  --model=$model --dataset=$dataset --batch-size=$bs \
  --lr=$lr --wd=$wd \
  --reg=$regmethod --regcoeff=$regcoeff \
  --logdir=$logdir