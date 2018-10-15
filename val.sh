NUM=0
GPU=1
EMBED=200
UNITS=200


DIR=expr/$NUM
python val.py --embed $EMBED --units $UNITS --num $NUM --gpu $GPU | tee $DIR/val_log.txt
