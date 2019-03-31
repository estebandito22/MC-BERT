import glob
import numpy as np
import os
import sys
import random
import csv

#this script takes the Pinterest metadata files and downloaded files
#and moves them into a more managable directory structure
#as well as creates new csv metadata files 

print("got" ,len(sys.argv), "options")

if len(sys.argv) == 4:
    metadata_dir = sys.argv[1]
    src_image_dir = sys.argv[2].split(',')
    dst_image_dir = sys.argv[3].split(',')
else:
    metadata_dir = "."
    src_image_dir = ["."]
    dst_image_dir = ["."]
    
print("Loading metadata files from", metadata_dir)
print("Loading images from", src_image_dir)
print("Moving images to", src_image_dir)

metafiles = glob.glob(os.path.join(metadata_dir,"*.npy"))

dirmap_file = open('dirmap.csv', 'a+', newline='')
dirmap_writer = csv.writer(dirmap_file)

allimgs_file = open('allimgs.csv', 'a+', newline='')
allimgs_writer = csv.writer(allimgs_file)



for file in metafiles:
    print("loading:", file, "... ", end = ' ')
    meta = np.load(file, encoding="bytes")

    meta_label = file.rsplit('.', maxsplit=1)[0].rsplit(os.sep, maxsplit=1)[-1]
    
    dest_dir = random.choice(dst_image_dir)
    
    full_dest_dir = os.path.join(dest_dir, meta_label)
    
    print(meta_label, ":", full_dest_dir)

    if not os.path.exists(full_dest_dir):
        os.makedirs(full_dest_dir)
    
    count = 0
    
    for i in range(len(meta)):
        
        imagename = meta[i][b'image_name'].decode('utf-8')
        
        image_src_file = None
        for imagesrc in src_image_dir:
            f = os.path.join(imagesrc, imagename)
            if os.path.exists(f):
                image_src_file = f
                break;
                
        if (image_src_file):
            newfile = os.path.join(full_dest_dir, imagename)
            #print("moving ", image_src_file, "to",os.path.join(full_dest_dir, imagename) )
            os.rename(image_src_file, newfile)
            allimgs_writer.writerow([imagename, newfile, meta[i][b'sentences'], meta[i][b'url']])
            count = count + 1
            
    if count > 0:
        dirmap_writer.writerow([meta_label, full_dest_dir])
        
    print("moved", count, "files.")

dirmap_file.close()
allimgs_file.close()
