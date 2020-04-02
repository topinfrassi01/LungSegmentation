# $1 is nepochs
# $2 is batchsize

RESULT_PATH=$(pwd)/output_$(date +%Y%m%d%H%M%S)

#mkdir $RESULT_PATH

python3 /Users/francistoupin/Documents/SYS843/nn/r2unet/main.py \
                --xtrain '/Users/francistoupin/Documents/SYS843/nn/r2unet/data/images_train.npy' \
                --xtest '/Users/francistoupin/Documents/SYS843/nn/r2unet/data/images_test.npy' \
                --ytrain '/Users/francistoupin/Documents/SYS843/nn/r2unet/data/masks_train.npy' \
                --ytest '/Users/francistoupin/Documents/SYS843/nn/r2unet/data/masks_test.npy' \
                --output $RESULT_PATH \
                --nepochs $1 \
                --batchsize $2
                

