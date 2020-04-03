# $1 is nepochs
# $2 is batchsize

RESULT_PATH=$(pwd)/output_$(date +%Y%m%d%H%M%S)

#mkdir $RESULT_PATH

python3 /Users/francistoupin/Documents/SYS843/preprocess/debug_processed.py \
                --xtrain '/Users/francistoupin/Documents/SYS843/data/prepared/images_train.npy' \
                --xtest '/Users/francistoupin/Documents/SYS843/data/prepared/images_test.npy' \
                --ytrain '/Users/francistoupin/Documents/SYS843/data/prepared/masks_train.npy' \
                --ytest '/Users/francistoupin/Documents/SYS843/data/prepared/masks_test.npy' \
                --batchsize $1
                

