rm -rf /Users/francistoupin/Documents/SYS843/data/prepared
mkdir /Users/francistoupin/Documents/SYS843/data/prepared

rm -rf /Users/francistoupin/Documents/SYS843/data/images/.DS_Store
rm -rf /Users/francistoupin/Documents/SYS843/data/masks_left_lung/.DS_Store
rm -rf /Users/francistoupin/Documents/SYS843/data/masks_right_lung/.DS_Store

python3 /Users/francistoupin/Documents/SYS843/preprocess/preprocess_images.py \
    --imgpath "/Users/francistoupin/Documents/SYS843/data/images" \
    --maskspath "/Users/francistoupin/Documents/SYS843/data/masks_left_lung" "/Users/francistoupin/Documents/SYS843/data/masks_right_lung" \
    --output "/Users/francistoupin/Documents/SYS843/nn/unet/data" \
    --grayscale \
    --testsize=0.2