from os import listdir
from PIL import Image

# import os
# os.chdir("/home/david/Documents/pistoldetection/dataset/handgun__person/")

# for filename in listdir('./'):
#     #print filename
#     if filename.endswith('.jpg'):
#         try:
#             img = Image.open("/home/david/Documents/pistoldetection/dataset/handgun__person/" + filename)  # open the image file
#             fact = img.verify()
#             print fact# verify that it is, in fact an image
#         except (IOError, SyntaxError) as e:
#             print('Bad file:', filename)  # print out the names of corrupt files
#




#
#
# import shutil
#
# def main():
#     path = "/home/david/Documents/pistoldetection/dataset/handgun__person"
#     newPath = "/home/david/Documents/pistoldetection/dataset/pistol"
#     count = 1
#
#     for root, dirs, files in os.walk(path):
#         for i in files:
#             if i.endswith('.jpg'):
#                 shutil.copy(os.path.join(root, i), os.path.join(newPath, "changed" + str(count) + ".jpg"))
#                 print i
#                 count += 1
#
# if __name__ == '__main__':
#     main()

__author__ = 'gerry'
#verify if an image is corrupt or not
#help from https://stackoverflow.com/questions/3964681/find-all-files-in-a-directory-with-extension-txt-in-python

img_dir="/home/david/Documents/pistoldetection/dataset/handgun__person/"

corrupt_img_dir="/home/david/Documents/pistoldetection/dataset/bad/"
good_img_dir="/home/david/Documents/pistoldetection/dataset/good/"

from PIL import Image
import os,time


def verify_image(img_file):
     #test image
     try:
        v_image = Image.open(img_file)
        v_image.verify()
        return True;
        #is valid
        #print("valid file: "+img_file)
     except OSError:
        return False;

count = 0
#main script
for root, dirs, files in os.walk(img_dir):
    for file in files:
        if file.endswith(".jpg"):
             currentFile=os.path.join(root, file)
             #test image
             if verify_image(currentFile):
                 new_file_name=good_img_dir+time.strftime("%Y%m%d%H%M%S_"+os.path.basename(currentFile))
                 print("good file, moving to dir: "+new_file_name)
                 try:
                     os.rename(currentFile, good_img_dir + str(count) + ".jpg")
                     count = count + 1
                 except WindowsError:
                     print("error moving file")
             else:
                 #Move to corrupt folder
                 #makefilename unique
                 new_file_name=corrupt_img_dir+time.strftime("%Y%m%d%H%M%S_"+os.path.basename(currentFile))
                 print("corrupt file")
                 os.rename(currentFile, new_file_name)