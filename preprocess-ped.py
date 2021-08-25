#!/usr/bin/env python

import glob
import json
import os
import shutil

import cv2

in_json_path = "../dhd_traffic/annotations/dhd_traffic_train.json"
out_dir_top = "../drive/MyDrive/peddet/ped-yolov5"

num_train = 1600
num_val = 400
num_test = 400

num_total = num_train + num_val + num_test
eqhist = True
denoising = True
mark_label = False
scale = False

#if os.path.exists(out_dir_top):
#    shutil.rmtree(out_dir_top)

class Image:
    def __init__(self):
        self.annotations = []

dat = json.load(open(in_json_path))

categories = [d["name"] for d in dat["categories"]]

# Create dictionary of image_id -> file_name
imgid2fname_dict = {}
for image in dat["images"]:
    imgid2fname_dict[image["id"]] = image["file_name"]

# Read annotations
images_dict = {}
for annotation in dat["annotations"]:
    #if annotation["category_id"] == 1 and annotation["ignore"] == 0:
    if annotation["ignore"] == 0:
        image_id = annotation["image_id"]
        if image_id in images_dict:
            image = images_dict[image_id]
        else:
            image = Image()
            images_dict[image_id] = image
        image.annotations.append(annotation)

# Convert dictionary to array
images = []
for image_id, image in images_dict.items():
    images.append(image)

# Extract a part of images which contains pedestrian
ex_images = [] # extracted images
num_extracted = 0
for i, image in enumerate(images):
    if True:#i % 5 == 0:
        contains_pedestrian = False
        for annotation in image.annotations:
            if annotation["category_id"] == 1:
                contains_pedestrian = True
                break
        if contains_pedestrian:
            ex_images.append(image)
            num_extracted += 1
            if len(ex_images) == num_total:
                break
if num_extracted < num_total:
    raise ValueError("Insufficient images: %d < %d" % (num_extracted, num_total))
images = None

orig_w = 1624
orig_h = 1200
scale_ratio = 1
for i, image in enumerate(ex_images):
    for annotation in image.annotations:
        lt_x_p = annotation["bbox"][0] # left-top x in pixel
        lt_y_p = annotation["bbox"][1] # left-top y in pixel
        w_p = annotation["bbox"][2]    # width in pixel
        h_p = annotation["bbox"][3]    # height in pixel
        c_x_p = lt_x_p + (w_p / 2)     # center x in pixel
        c_y_p = lt_y_p + (h_p / 2)     # center y in pixel
        c_x_r = float(c_x_p) / orig_w  # center x in ratio
        c_y_r = float(c_y_p) / orig_h  # center y in ratio
        w_r = float(w_p) / orig_w
        w_h = float(h_p) / orig_h
        rel_bbox = [c_x_r, c_y_r, w_r, w_h]
        annotation["rel_bbox"] = rel_bbox
        image_id = annotation["image_id"]
        fname = imgid2fname_dict[image_id]
        #print(fname)
        #print(annotation["bbox"])
        #print(annotation["rel_bbox"])

# Generate or clean up output directories
out_dir_train = out_dir_top + "/train"
out_dir_valid = out_dir_top + "/valid"
out_dir_test  = out_dir_top + "/test"
out_dir_train_image = out_dir_train + "/images"
out_dir_train_label = out_dir_train + "/labels"
out_dir_valid_image = out_dir_valid + "/images"
out_dir_valid_label = out_dir_valid + "/labels"
out_dir_test_image  = out_dir_test  + "/images"
out_dir_test_label  = out_dir_test  + "/labels"
def remove_files_under(dir_path):
    dir_path = dir_path.rstrip('/')
    file_paths = glob.glob(dir_path + "/*")
    for file_path in file_paths:
        os.remove(file_path)
def clean_gen_dir(dir_path):
    if os.path.exists(dir_path):
        remove_files_under(dir_path)
    else:
        os.makedirs(dir_path)
out_dirs = [out_dir_train_image, out_dir_train_label, out_dir_valid_image, out_dir_valid_label, out_dir_test_image, out_dir_test_label]
for out_dir in out_dirs:
    clean_gen_dir(out_dir)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
for i, image in enumerate(ex_images):
    image_id = image.annotations[0]["image_id"]
    img_fname = imgid2fname_dict[image_id]
    stem = os.path.splitext(img_fname)[0]
    txt_fname = stem + ".txt"
    print("%5d %s" % (i, fname))
    src_path = "../dhd_traffic/images/train/" + img_fname
    if i < num_train:
        img_dst_path = out_dir_train_image + "/" + img_fname
        txt_dst_path = out_dir_train_label + "/" + txt_fname
    elif i < num_train + num_val:
        img_dst_path = out_dir_valid_image + "/" + img_fname
        txt_dst_path = out_dir_valid_label + "/" + txt_fname
    elif i < num_total:
        img_dst_path = out_dir_test_image + "/" + img_fname
        txt_dst_path = out_dir_test_label + "/" + txt_fname
    else:
        break

    if eqhist or denoising:
        img = cv2.imread(src_path)
        if eqhist:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            lab_planes = cv2.split(lab)
            lab_planes[0] = clahe.apply(lab_planes[0])
            lab = cv2.merge(lab_planes)
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        if denoising:
            img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
        cv2.imwrite(img_dst_path, img)
    else:
        shutil.copy2(src_path, img_dst_path)

    with open(txt_dst_path, "w") as fout:
        for annotation in image.annotations:
            cat_id = annotation["category_id"] # 1: ped, 2: cyc, 3: car, 4: truck, 5: van
            if True:#cat_id == 1:
                rbb = annotation["rel_bbox"]
                fout.write("%d %f %f %f %f\n" % (cat_id-1, rbb[0], rbb[1], rbb[2], rbb[3]))

yaml_path = out_dir_top + "/data.yaml"
categories_with_quote = ["'"+cat+"'" for cat in categories]
with open(yaml_path, "w") as f:
    f.write("train: " + out_dir_top + "/train/images\n")
    f.write("val: "   + out_dir_top + "/valid/images\n")
    f.write("\n")
    f.write("nc: %d\n" % len(categories))
    f.write("names: [" + ",".join(categories_with_quote) + "]\n")
