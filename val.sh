NUM=1
GPU=2
EMBED=64
UNITS=64


DIR=expr/$NUM
python val.py --embed $EMBED --units $UNITS --num $NUM --gpu $GPU | tee $DIR/val_log.txt
