import math

from PIL import Image
import requests
import matplotlib.pyplot as plt

#import ipywidgets as widgets
#from IPython.display import display, clear_output

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
from hubconf import*
from util.misc import nested_tensor_from_tensor_list
torch.set_grad_enabled(False);

# COCO classes
CLASSES = [
    'cat', 'dog','background'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098]]

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()


def detect(im, model, transform):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.00001

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas[keep], bboxes_scaled

def predict(im, model, transform):
    # mean-std normalize the input image (batch-size: 1)
    anImg = transform(im)
    data = nested_tensor_from_tensor_list([anImg])

    # propagate through the model
    outputs = model(data) #输出100个索引，包含预测值和框值

    # keep only predictions with 0.7+ confidence
    # b = outputs['pred_logits'].softmax(-1)
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1] #softmax，并且去掉每个类别的最后一类

    keep = probas.max(-1).values > 0.7 #-1表示按行取最大值
    #print(probas[keep])

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas[keep], bboxes_scaled

#多标签预测
def predict_multi(im, model, transform):
    anImg = transform(im)
    data = nested_tensor_from_tensor_list([anImg])

    # propagate through the model
    outputs = model(data) #输出100个索引，包含预测值和框值

    # keep only predictions with 0.7+ confidence
    # b = outputs['pred_logits'].softmax(-1)
    probas_test = outputs['pred_logits'].sigmoid()[0, :, :-2]
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-2] #softmax，并且去掉每个类别的最后一类

    keep_test = probas_test.max(-1).values > 0.5 #-1表示按行取最大值
    # keep = probas.max(-1).values > 0.5 #-1表示按行取最大值
    #print(probas[keep])
    val = probas_test.unsqueeze(0)[0,keep_test]
    # val = outputs['pred_boxes'][0, keep]
    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep_test], im.size)
    return probas[keep_test], bboxes_scaled



if __name__ == "__main__":

    model=detr_resnet50(False,3);
    state_dict = torch.load("data/output_test/checkpoint.pth",map_location='cpu')
    model.load_state_dict(state_dict["model"])
    model.eval()

    im = Image.open('data/test/train2017/2.jpg')

    scores, boxes = predict_multi(im, model, transform)
    plot_results(im, scores, boxes)