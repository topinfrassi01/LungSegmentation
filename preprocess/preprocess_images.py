import argparse
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split

def preprocess_images(image_path, masks_paths, output, one_hot, grayscale, test_size, binary, add_contours):

    if not os.path.exists(image_path):
        raise ValueError("imgpath : {0} doesn't exist".format(image_path))

    for mask_path in masks_paths:
        if not os.path.exists(mask_path):
            raise ValueError("maskpath : {0} doesn't exist".format(mask_path))

    if not os.path.exists(output):
        raise ValueError("output : {0} doesn't exist".format(output))

    first_pass = True
    image_files = list(sorted(os.listdir(image_path)))
    size = len(image_files)

    for i, file in enumerate(image_files):
        img = cv2.imread(os.path.join(image_path, file), cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
        
        if first_pass:
            images = np.zeros((size,) + img.shape + (1,), dtype=np.uint8)
            first_pass = False
        
        # Fill the black boxes in images with mean value to hopefully help training.
        np.place(img, img == 0, np.mean(img))
        images[i,:,:,0] = img

    masks = np.zeros((size,) + img.shape[0:2] + (len(masks_paths) if one_hot else 1,), dtype=np.uint8)

    if add_contours:
        contours = np.zeros_like(masks)

    for c, mask_path in enumerate(masks_paths):

        masks_files = list(sorted(os.listdir(mask_path)))

        for i, file in enumerate(masks_files):

            if not file[0:file.find(".")] == image_files[i][0:image_files[i].find(".")]:
                print(file)
                print(image_files[i])
                raise ValueError("Masks should have the same names as their corresponding images.")

            mask = cv2.imread(os.path.join(mask_path, file), cv2.IMREAD_GRAYSCALE)
            
            ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            mask = mask.astype(np.uint8)

            if add_contours:
                ctrs, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
                contours_img = np.zeros_like(mask)

                if len(ctrs) > 1:
                    print(file)
                    raise ValueError("Mask shouldn't have more than one connection region.")

                cv2.drawContours(contours_img, ctrs, -1, (255), 1) 

            if one_hot:
                if add_contours:
                    contours[i,:,:,c] = contours_img

                masks[i,:,:,c] = mask
            else:
                binary_factor = 1 if binary else c + 1

                masks[i,:,:,0] = masks[i,:,:,0] + ((mask / np.max(mask)) * binary_factor)

                if add_contours:
                    contours[i,:,:,0] = contours[i,:,:,0] + ((contours_img / np.max(contours_img)) * binary_factor)


    if not one_hot:
        binary_factor = 1 if binary else c + 1

        masks = np.clip(masks, None, binary_factor)

        if add_contours:
            contours = np.clip(contours, None, binary_factor)


    train_size = size // (1 - test_size)
    
    images_train, images_test, masks_train, masks_test = train_test_split(images, masks, test_size=test_size, random_state=7)
    
    images_train = images_train.astype(np.float32)
    images_test = images_test.astype(np.float32)

    np.save(os.path.join(output, "images_train.npy"), images_train)
    np.save(os.path.join(output, "images_test.npy"), images_test)

    np.save(os.path.join(output, "masks_train.npy"), masks_train)
    np.save(os.path.join(output, "masks_test.npy"), masks_test)        

    print("Created images_train.npy with shape {0} with type {1}".format(images_train.shape, images_train.dtype))
    print("Created images_test.npy with shape {0} with type {1}".format(images_test.shape, images_test.dtype))
    print("Created masks_train.npy with shape {0} with type {1}".format(masks_train.shape, masks_train.dtype))
    print("Created masks_test.npy with shape {0} with type {1}".format(masks_test.shape, masks_test.dtype))

    if add_contours:
        _, __, contours_train, contours_test = train_test_split(images, contours, test_size=test_size, random_state=7)

        np.save(os.path.join(output, "contours_train.npy"), contours_train)
        np.save(os.path.join(output, "contours_test.npy"), contours_test)             

        print("Created contours_train.npy with shape {0} with type {1}".format(contours_train.shape, contours_train.dtype))
        print("Created contours_test.npy with shape {0} with type {1}".format(contours_test.shape, contours_test.dtype))


def main():
    parser = argparse.ArgumentParser(description='Pre-process images before feeding them to various DCNN.')
    parser.add_argument('--imgpath', type=str, help='Path where images are.')
    parser.add_argument('--maskspath', type=str, nargs='+', help='Path(s) where masks are. Masks are expected to have the same name as their corresponding image. Masks are expected to be separated by class.')
    parser.add_argument('--output', type=str, help="Output path for the processed data.")
    parser.add_argument('--grayscale', action='store_true', help="Converts images to grayscale. Default False")
    parser.add_argument('--onehot', action='store_true', help="Outputs the labels to one-hot matrix. Default is label-encoded.")
    parser.add_argument('--testsize', type=float, help="% of the test size. Default zero.")
    parser.add_argument('--binary', action="store_true", help="If set, all masks will have the same value. Cannot be set with --onehot.")
    parser.add_argument('--contours', action="store_true", help="Generates contours.")

    args = parser.parse_args()

    if args.onehot and args.binary:
        raise argparse.ArgumentError("Cannot have --binary and --onehot at the same time")

    preprocess_images(args.imgpath, args.maskspath, args.output, args.onehot, args.grayscale, args.testsize, args.binary, args.contours)
    

if __name__ == "__main__":
    main()