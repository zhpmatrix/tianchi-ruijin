import bilstm_crf_model
import argparse
from utils import *
from keras.callbacks import ModelCheckpoint

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TRAIN')
    parser.add_argument('--num', type=int)
    parser.add_argument('--embed', type=int)
    parser.add_argument('--units', type=int)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--save', type=str)
    parser.add_argument('--batch', type=int, default=2)
    
    args = parser.parse_args()
    gpu_config(args.gpu)
    

    model, (train_x, train_y), (test_x, test_y) = bilstm_crf_model.create_model(args.embed, args.units)
    # used for multi checkpoints to vote
    #filepath = args.save+'/weights-improvement-{epoch:02d}-{val_acc:.4f}.h5'
    
    # only get the best single model
    filepath = args.save+'/model.h5'
    
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    model.fit(train_x, train_y,batch_size=args.batch,epochs=args.epoch, validation_data=[test_x, test_y],callbacks=[checkpoint])
    
