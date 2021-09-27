# import os
#
# w = open(r'VOCdevkit/VOC2007/ImageSets/Main/train.txt','w')
# path = r'VOCdevkit/VOC2007/Annotations'
# for name in os.listdir(path):
#     w.write(name.split('.')[0]+'\n')



# annpaths = r'VOCdevkit/VOC2007/Annotations'
# imgpaths = r'VOCdevkit/VOC2007/JPEGImages'
#
# count = 0
# for annname in os.listdir(annpaths):
#     imgname = annname.replace('xml','jpg')
#     ann = os.path.join(annpaths,annname)
#     img = os.path.join(imgpaths,imgname)
#
#     print(ann)
#     print(img)
#
#     new_ann = os.path.join(annpaths,str(count)+'.xml')
#     new_img = os.path.join(imgpaths,str(count)+'.jpg')
#
#     os.rename(ann,new_ann)
#     os.rename(img,new_img)
#     count+=1



# import xml.etree.ElementTree as ET
# import pickle
# import os
# from os import listdir, getcwd
# from os.path import join
#
#
# path = r'VOCdevkit/VOC2007/Annotations'
# classes = []
# for name in os.listdir(path):
#     infile = os.path.join(path,name)
#     in_file = open(infile)
#     tree = ET.parse(in_file)
#     root = tree.getroot()
#     size = root.find('size')
#     w = int(size.find('width').text)
#     h = int(size.find('height').text)
#
#     for obj in root.iter('object'):
#         difficult = obj.find('difficult').text
#         cls = obj.find('name').text
#         if cls not in classes:classes.append(cls)
#
# print(classes)
