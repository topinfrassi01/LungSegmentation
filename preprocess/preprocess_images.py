import argparse
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def preprocess_images(image_path, masks_paths, output, test_size):

    if not os.path.exists(image_path):
        raise ValueError("imgpath : {0} doesn't exist".format(image_path))

    for mask_path in masks_paths:
        if not os.path.exists(mask_path):
            raise ValueError("maskpath : {0} doesn't exist".format(mask_path))

    if not os.path.exists(output):
        os.mkdir(output)
        print("Created :" + output)

    first_pass = True
    image_files = list(sorted(os.listdir(image_path)))
    size = len(image_files)

    for i, file in enumerate(image_files):
        img = cv2.imread(os.path.join(image_path, file), cv2.IMREAD_GRAYSCALE)
        
        if first_pass:
            images = np.zeros((size,) + img.shape + (1,), dtype=np.uint8)
            first_pass = False
        
        # Fill the black boxes in images with mean value to hopefully help training.
        np.place(img, img == 0, np.mean(img))
        images[i,:,:,0] = img

    masks = np.zeros((size,) + img.shape[0:2] + (1,), dtype=np.uint8)

    contours = np.zeros_like(masks)

    for c, mask_path in enumerate(masks_paths):
        class_id = c + 1
        print("Processing {0}".format(mask_path))
        masks_files = list(sorted(os.listdir(mask_path)))

        for i, file in enumerate(masks_files):

            if not file[0:file.find(".")] == image_files[i][0:image_files[i].find(".")]:
                print(file)
                print(image_files[i])
                raise ValueError("Masks should have the same names as their corresponding images.")

            mask = cv2.imread(os.path.join(mask_path, file), cv2.IMREAD_GRAYSCALE)

            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            mask = mask.astype(np.uint8)

            ctrs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            contours_img = np.zeros_like(mask)

            cv2.drawContours(contours_img, ctrs, -1, (255), 1) 

            masks[i,:,:,0] = masks[i,:,:,0] + ((mask / np.max(mask)) * class_id)

            contours[i,:,:,0] = contours[i,:,:,0] + ((contours_img / np.max(contours_img)) * class_id)

    masks = np.clip(masks, None, class_id)
    contours = np.clip(contours, None, class_id)
    
    masks = to_categorical(masks)
    contours = to_categorical(contours)

    images_train, images_test, masks_train, masks_test = train_test_split(images, masks, test_size=test_size, random_state=7)
    _, __, contours_train, contours_test = train_test_split(images, contours, test_size=test_size, random_state=7)    
    
    images_train = images_train.astype(np.float32)
    images_test = images_test.astype(np.float32)

    save_np_array_as_directory(images_train, os.path.join(output, "train\\images"))
    save_np_array_as_directory(images_test, os.path.join(output, "test\\images"))
    save_np_array_as_directory(masks_train, os.path.join(output, "train\\masks"))
    save_np_array_as_directory(masks_test, os.path.join(output, "test\\masks"))
    save_np_array_as_directory(contours_train, os.path.join(output, "train\\contours"))
    save_np_array_as_directory(contours_test, os.path.join(output, "test\\contours"))

    np.save(os.path.join(output, "images_train.npy"), images_train)
    np.save(os.path.join(output, "images_test.npy"), images_test)

    np.save(os.path.join(output, "masks_train.npy"), masks_train)
    np.save(os.path.join(output, "masks_test.npy"), masks_test)        

    print("Created images_train.npy with shape {0} with type {1}".format(images_train.shape, images_train.dtype))
    print("Created images_test.npy with shape {0} with type {1}".format(images_test.shape, images_test.dtype))
    print("Created masks_train.npy with shape {0} with type {1}".format(masks_train.shape, masks_train.dtype))
    print("Created masks_test.npy with shape {0} with type {1}".format(masks_test.shape, masks_test.dtype))

    _, __, contours_train, contours_test = train_test_split(images, contours, test_size=test_size, random_state=7)

    np.save(os.path.join(output, "contours_train.npy"), contours_train)
    np.save(os.path.join(output, "contours_test.npy"), contours_test)             

    print("Created contours_train.npy with shape {0} with type {1}".format(contours_train.shape, contours_train.dtype))
    print("Created contours_test.npy with shape {0} with type {1}".format(contours_test.shape, contours_test.dtype))


def save_np_array_as_directory(array, directory):
    path = os.path.join(os.getcwd(), directory)
    if not os.path.exists(path):
        os.makedirs(path)

    for i in range(array.shape[0]):
        success = cv2.imwrite(os.path.join(path, "{0}.png".format(i)), array[i,:,:,:])

        if not success:
            raise Error("Img write didn't work")

    print("Successfully saved {0} images to directory {1}".format(array.shape[0], directory))

def main():
    imgpath = "C:\\Users\\Francis\\Documents\\Repositories\\LungSegmentation\\data\\images"
    maskspath = ["C:\\Users\\Francis\\Documents\\Repositories\\LungSegmentation\\data\\lungs", "C:\\Users\\Francis\\Documents\\Repositories\\LungSegmentation\\data\\masks_heart"]
    output = "C:\\Users\\Francis\\Documents\\Repositories\\LungSegmentation\\data\\prepared"
    testsize = 0.2
    
    preprocess_images(imgpath, maskspath, output, testsize)
    

if __name__ == "__main__":
    main()