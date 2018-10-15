NUM=0
GPU=3

EPOCH=100
BATCH=128

EMBED=200
UNITS=200

DIR=expr/$NUM
if [ ! -d $DIR ];then
	mkdir $DIR
fi
python train.py --embed $EMBED --units $UNITS --num $NUM --epoch $EPOCH --gpu $GPU --batch $BATCH --save $DIR | tee $DIR/train_log.txt
