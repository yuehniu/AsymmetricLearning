python=python

# --> training hparams
dataset=imagenet
model=resnet18
lr=0.1
wd=2e-4
regcoeff=5e-4
regmethod="l2"
gpu="4,5,6,7"

svd=1
rank=8
dct=0
quant=0
noise=0.4
moreopts=" "

logdir="./log/${dataset}_${model}/orig/lr${lr}_${regmethod}_${wd}"
if [ $dataset == "imagenet"  ]; then
  moreopts+=" --epochs=100 --batch-size=512"
fi

if [ $svd -eq 1 ]; then
  moreopts+=" --svd --rank=${rank}"
  logdir="./log/${dataset}_${model}/svd/lr${lr}_${regmethod}_${wd}-${regcoeff}_svd${rank}_botneck_3x3_bn"
fi

if [ $dct -eq 1 ]; then
  moreopts+=" --dct"
  logdir="./log/${dataset}_${model}/dct/lr${lr}_${regmethod}_${wd}-${regcoeff}_svd${rank}_dct32-16-prelu-64"
fi

if [ $quant -eq 1 ]; then
  moreopts+="  --quant --noise=${noise}"
  logdir="./log/${dataset}_${model}/asym/lr${lr}_${regmethod}_${wd}-${regcoeff}_svd${rank}_dct32-16_quant${noise}"
fi

if [ -d $logdir ]; then
    rm -r $logdir
fi
mkdir -p $logdir

CUDA_VISIBLE_DEVICES=$gpu $python main.py  ${moreopts} \
  --cuda \
  --model=$model --dataset=$dataset \
  --lr=$lr --wd=$wd \
  --reg=$regmethod --regcoeff=$regcoeff \
  --logdir=$logdir