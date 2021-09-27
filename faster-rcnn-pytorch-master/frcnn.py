import colorsys
import copy
import os
import time

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from nets.frcnn import FasterRCNN
from utils.utils import DecodeBox, get_new_img_size


#--------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和classes_path都需要修改！
#   如果出现shape不匹配
#   一定要注意训练时的NUM_CLASSES、
#   model_path和classes_path参数的修改
#--------------------------------------------#
class FRCNN(object):
    _defaults = {
        "model_path"    : 'logs/Epoch236-Total_Loss0.3896.pth',
        "classes_path"  : 'model_data/voc_classes.txt',
        "confidence"    : 0.9,
        "iou"           : 0.1,
        "backbone"      : "Swin",
        "cuda"          : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化faster RCNN
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.generate()

        self.mean = torch.Tensor([0, 0, 0, 0]).repeat(self.num_classes+1)[None]
        self.std = torch.Tensor([0.1, 0.1, 0.2, 0.2]).repeat(self.num_classes+1)[None]
        if self.cuda:
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()
            
        self.decodebox = DecodeBox(self.std, self.mean, self.num_classes)

    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    #---------------------------------------------------#
    #   载入模型
    #---------------------------------------------------#
    def generate(self):
        #-------------------------------#
        #   计算总的类的数量
        #-------------------------------#
        self.num_classes = len(self.class_names)

        #-------------------------------#
        #   载入模型与权值
        #-------------------------------#
        self.model = FasterRCNN(self.num_classes,"predict",backbone=self.backbone).eval()
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(self.model_path, map_location=device)
        self.model.load_state_dict(state_dict)
        
        if self.cuda:
            # self.model = nn.DataParallel(self.model)
            self.model = self.model.cuda()

        print('{} model, anchors, and classes loaded.'.format(self.model_path))

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
    
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        #-------------------------------------#
        #   转换成RGB图片，可以用于灰度图预测。
        #-------------------------------------#
        # image = image.convert("RGB")
        #
        # image_shape = np.array(np.shape(image)[0:2])
        # old_width, old_height = image_shape[1], image_shape[0]
        # old_image = copy.deepcopy(image)
        #
        # #---------------------------------------------------------#
        # #   给原图像进行resize，resize到短边为600的大小上
        # #---------------------------------------------------------#
        # width,height = get_new_img_size(old_width, old_height)
        # image = image.resize([width,height], Image.BICUBIC)
        #
        # #-----------------------------------------------------------#
        # #   图片预处理，归一化。
        # #-----------------------------------------------------------#
        # photo = np.transpose(np.array(image,dtype = np.float32)/255, (2, 0, 1))

        # resize image
        image = image.convert("RGB")
        iw, ih = image.size
        old_image = copy.deepcopy(image)
        h, w = 416,416
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image, np.float32)
        image_data = np.transpose(image_data / 255.0, [2, 0, 1])


        with torch.no_grad():
            images = torch.from_numpy(np.asarray([image_data]))
            if self.cuda:
                images = images.cuda()

            roi_cls_locs, roi_scores, rois, _ = self.model(images)
            #-------------------------------------------------------------#
            #   利用classifier的预测结果对建议框进行解码，获得预测框
            #-------------------------------------------------------------#
            outputs = self.decodebox.forward(roi_cls_locs[0], roi_scores[0], rois, height = h, width = w, nms_iou = self.iou, score_thresh = self.confidence)
            #---------------------------------------------------------#
            #   如果没有检测出物体，返回原图
            #---------------------------------------------------------#
            if len(outputs)==0:
                return old_image
            outputs = np.array(outputs)
            bbox = outputs[:,:4]
            label = outputs[:, 4]
            conf = outputs[:, 5]

            bbox[:, 0::2] = (bbox[:, 0::2]) / w * iw
            bbox[:, 1::2] = (bbox[:, 1::2]) / h * ih

        font = ImageFont.truetype(font='model_data/simhei.ttf',size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        thickness = max((np.shape(old_image)[0] + np.shape(old_image)[1]) // iw * 2, 1)
                
        image = old_image
        for i, c in enumerate(label):
            predicted_class = self.class_names[int(c)]
            score = conf[i]

            left, top, right, bottom = bbox[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[int(c)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[int(c)])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        
        return image

    def get_FPS(self, image, test_interval):
        #-------------------------------------#
        #   转换成RGB图片，可以用于灰度图预测。
        #-------------------------------------#
        image = image.convert("RGB")

        image_shape = np.array(np.shape(image)[0:2])
        old_width, old_height = image_shape[1], image_shape[0]
        
        #---------------------------------------------------------#
        #   给原图像进行resize，resize到短边为600的大小上
        #---------------------------------------------------------#
        width,height = get_new_img_size(old_width, old_height)
        image = image.resize([width,height], Image.BICUBIC)

        #-----------------------------------------------------------#
        #   图片预处理，归一化。
        #-----------------------------------------------------------#
        photo = np.transpose(np.array(image,dtype = np.float32)/255, (2, 0, 1))

        with torch.no_grad():
            images = torch.from_numpy(np.asarray([photo]))
            if self.cuda:
                images = images.cuda()

            roi_cls_locs, roi_scores, rois, _ = self.model(images)
            #-------------------------------------------------------------#
            #   利用classifier的预测结果对建议框进行解码，获得预测框
            #-------------------------------------------------------------#
            outputs = self.decodebox.forward(roi_cls_locs[0], roi_scores[0], rois, height = height, width = width, nms_iou = self.iou, score_thresh = self.confidence)
            #---------------------------------------------------------#
            #   如果没有检测出物体，返回原图
            #---------------------------------------------------------#
            if len(outputs)>0:
                outputs = np.array(outputs)
                bbox = outputs[:,:4]
                label = outputs[:, 4]
                conf = outputs[:, 5]

                bbox[:, 0::2] = (bbox[:, 0::2]) / width * old_width
                bbox[:, 1::2] = (bbox[:, 1::2]) / height * old_height
                    
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                roi_cls_locs, roi_scores, rois, _ = self.model(images)
                #-------------------------------------------------------------#
                #   利用classifier的预测结果对建议框进行解码，获得预测框
                #-------------------------------------------------------------#
                outputs = self.decodebox.forward(roi_cls_locs[0], roi_scores[0], rois, height = height, width = width, nms_iou = self.iou, score_thresh = self.confidence)
                #---------------------------------------------------------#
                #   如果没有检测出物体，返回原图
                #---------------------------------------------------------#
                if len(outputs)>0:
                    outputs = np.array(outputs)
                    bbox = outputs[:,:4]
                    label = outputs[:, 4]
                    conf = outputs[:, 5]

                    bbox[:, 0::2] = (bbox[:, 0::2]) / width * old_width
                    bbox[:, 1::2] = (bbox[:, 1::2]) / height * old_height
                
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time
