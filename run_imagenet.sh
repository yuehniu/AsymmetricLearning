python=python

# --> training hparams
dataset=imagenet
model=resnet18
lr=0.1
wd=1e-4
regcoeff=5e-4
regmethod="l2"
seed=2
gpu="0,1,2,3"

svd=1
rank=8
dct=1
quant=1
p=0.0002
epsilon=2
moreopts=" "

logdir="./log/acc/${dataset}_${model}/orig/lr${lr}_${regmethod}_${wd}"
if [ $dataset == "imagenet"  ]; then
  moreopts+=" --epochs=100 --batch-size=256"
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
    logdir="./log/acc/${dataset}_${model}/asym/BBfreeze_${regmethod}_${wd}_svd${rank}_ep${epsilon}_M2-1_seed${seed}"
  else
    logdir="./log/acc/${dataset}_${model}/asym/BBfreeze_${regmethod}_${wd}-${regcoeff}_svd${rank}_ep${epsilon}_M2-1_seed${seed}"
  fi
fi

if [ -d $logdir ]; then
    rm -r $logdir
fi
mkdir -p $logdir

CUDA_VISIBLE_DEVICES=$gpu $python main.py  ${moreopts} \
  --cuda \
  --model=$model --dataset=$dataset \
  --lr=$lr --wd=$wd --seed=${seed}\
  --reg=$regmethod --regcoeff=$regcoeff \
  --logdir=$logdir --save-dir=${savedir}