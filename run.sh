python=python

# --> training hparams
dataset=cifar100
model=resnet18
lr=0.1
wd=2e-4
regcoeff=1e-3
regmethod="orth"
seed=3
gpu="0"

svd=1
rank=8
dct=1
quant=1
epsilon=1000
dpdata=0
p=0.001
moreopts=" "

logdir="./log/acc/${dataset}_${model}/orig/lr${lr}_${regmethod}_${wd}"
savedir="./checkpoints/running/${dataset}_${model}"
if [ $dataset == "imagenet"  ]; then
  moreopts+=" --epochs=100 --batch-size=512"
fi

if [ $dpdata -eq 1 ]; then
  moreopts+=" --dpdata --p=${p}"
  logdir="./log/acc/${dataset}_${model}/dpdata/lr${lr}_${regmethod}_${wd}"
fi

if [ $svd -eq 1 ]; then
  moreopts+=" --svd --rank=${rank}"
  if [ $regmethod == "l2" ]; then
    logdir="./log/acc/${dataset}_${model}/svd/lr${lr}_${regmethod}_${wd}_svd${rank}_botneck_3x3_bn_expand8"
  else
    logdir="./log/acc/${dataset}_${model}/svd/lr${lr}_${regmethod}_${wd}-${regcoeff}_svd${rank}_botneck_3x3_bn_expand8"
  fi
fi

if [ $dct -eq 1 ]; then
  moreopts+=" --dct"
  if [ $regmethod == "l2" ]; then
    logdir="./log/acc/${dataset}_${model}/dct/lr${lr}_${regmethod}_${wd}_svd${rank}_dct28-14-prelu-bb_expand12"
  else
    logdir="./log/acc/${dataset}_${model}/dct/lr${lr}_${regmethod}_${wd}-${regcoeff}_svd${rank}_dct28-14-prelu-bb_expand12"
  fi
fi

if [ $quant -eq 1 ]; then
  moreopts+="  --quant --epsilon=${epsilon} --p=${p}"
  if [ $regmethod == "l2" ]; then
    logdir="./log/acc/${dataset}_${model}/asym/NsyGrad0.02_${lr}_${regmethod}_${wd}_svd${rank}_dct32-16_quant${noise}_M2large"
  else
    logdir="./log/acc/${dataset}_${model}/asym/BBfreeze_${regmethod}_${wd}-${regcoeff}_svd${rank}_ep${epsilon}_M2-1_seed${seed}"
  fi
fi

if [ -d $logdir ]; then
    rm -r $logdir
fi
if [ ! -d $savedir ]; then
    mkdir -p $savedir
fi
mkdir -p $logdir

CUDA_VISIBLE_DEVICES=$gpu $python main.py  ${moreopts} \
  --cuda \
  --model=$model --dataset=$dataset \
  --lr=$lr --wd=$wd --seed=${seed} \
  --reg=$regmethod --regcoeff=$regcoeff \
  --logdir=$logdir --save-dir=${savedir}