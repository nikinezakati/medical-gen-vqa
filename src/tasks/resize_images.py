import os
import argparse
from PIL import Image
from src.constants import IMGS_DIR, IMGS_RESIZED_DIR


def resize_image(image, size):

    return image.resize(size, Image.ANTIALIAS)


def resize_images(input_dir, output_dir, size):

    iter=0
    for dir in os.listdir(input_dir):
        if dir != '.DS_Store':
            image_id = int(dir.replace('xmlab', ''))
            img_dir = IMGS_DIR + dir
            filepath = os.path.join(img_dir, 'source.jpg')
            try:
                with open(filepath, 'r+b') as f:
                    with Image.open(f) as img:
                        img = resize_image(img, size)
                        img.save(output_dir + str(image_id) + '.png')
            except(IOError, SyntaxError) as e:
                pass
            if (iter) % 50 == 0:
                print("[{}/{}] Resized the images and saved into '{}'.png"
                        .format(dir, str(image_id), output_dir+str(image_id)))
            iter+=1    
            
    
if __name__ == '__main__':
    input_dir = IMGS_DIR
    output_dir = IMGS_RESIZED_DIR
    image_size = [224, 224]
    resize_images(input_dir, output_dir, image_size)
