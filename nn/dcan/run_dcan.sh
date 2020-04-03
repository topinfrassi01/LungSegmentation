# $1 is nepochs
# $2 is batchsize

RESULT_PATH=$(pwd)/output_$(date +%Y%m%d%H%M%S)

#mkdir $RESULT_PATH

python3 /Users/francistoupin/Documents/SYS843/nn/dcan/main.py \
                --xtrain '/Users/francistoupin/Documents/SYS843/nn/dcan/data/images_train.npy' \
                --xtest '/Users/francistoupin/Documents/SYS843/nn/dcan/data/images_test.npy' \
                --contourstrain '/Users/francistoupin/Documents/SYS843/nn/dcan/data/contours_train.npy' \
                --contourstest '/Users/francistoupin/Documents/SYS843/nn/dcan/data/contours_test.npy' \
                --maskstrain '/Users/francistoupin/Documents/SYS843/nn/dcan/data/masks_train.npy' \
                --maskstest '/Users/francistoupin/Documents/SYS843/nn/dcan/data/masks_test.npy' \
                --output $RESULT_PATH \
                --nepochs $1 \
                --batchsize $2