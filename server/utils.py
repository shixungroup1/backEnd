from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import torch
import torch.nn as nn
from torchvision import transforms as T
from saliency_pytorch.ssrnet import SSRNet


# if __name__ == '__main__':
#     #获取图片
#     rootDir = './imgs/'
#     resultDir = './SSRN_DUTS_v2/'
#     img1 = Image.open(os.path.join(rootDir, "2092.jpg"))
#     img2 = Image.open(os.path.join(resultDir, "2092.png"))
#     plt.imshow(img1)

#     #使用模型
#     if True:
#         #sys.path.append("../..")
#         from saliency_pytorch.ssrnet import SSRNet
#         class ImageModel(nn.Module):
#             def __init__(self, pretrained = False):
#                 super(ImageModel, self).__init__()
#                 self.backbone = SSRNet(1, 16, pretrained=pretrained)

#             def forward(self, frame):
#                 seg = self.backbone(frame)
#                 return seg
#         device = torch.device('cuda:1')
#         model = ImageModel(pretrained=True)
#         # model_path = "./saliency_pytorch/image_miou_087.pth"
#         model_path = "../../saliency_pytorch/image_miou_087.pth"
#         model.load_state_dict(torch.load(model_path), strict = True)


class ImageModel(nn.Module):
    def __init__(self, pretrained = False):
        super(ImageModel, self).__init__()
        self.backbone = SSRNet(1, 16, pretrained=pretrained)

    def forward(self, frame):
        seg = self.backbone(frame)
        seg = torch.sigmoid(seg)
        return seg


"""获取显著性结果图地函数
参数：
ori_img   ：输入的原图片，一般是pil.image形式。
model     : 输入的模型
device    : cuda

示例
模型见上方
img1 = Image.open(os.path.join(rootDir, "119082.jpg"))
createSOD(img1, model, device)

返回
单通道的pil.image图片
"""
def createSOD(ori_img, model, device):
    # print(np.array(ori_img))
    w, h = ori_img.size
    ori_img = ori_img.convert('RGB')
    model.eval()
    model.to(device)
    preTransform = T.Compose([
        T.Resize(320),
        #T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.4924044, 0.47831464, 0.44143882), (0.25063434, 0.2492162,  0.26660094))
        ])
    img = preTransform(ori_img) 
    # img = torch.tensor(np.array(ori_img))
    # img = img.type(torch.FloatTensor)
    img = img.to(device)
    img = img.unsqueeze(0)
    #i mg = img.permute(0, 3, 1, 2)
    # print(img.shape)
    result = model(img)
    
    # result = result.permute(0, 2, 3, 1)
    result = result.squeeze().squeeze()
    
    result = result.detach().cpu().numpy()
    # print(result)
    result *= 255
    result = cv2.resize(result, dsize=(w, h))
    # plt.imshow(result)
    # result = Image.fromarray(np.uint8(result))
    return result
    


"""抠图
参数：
ori_img   ：输入的原图片，一般是pil.image形式。
sod_img   ：原图片的显著性分析结果，一般是pil.image形式。
threshold ：图片抠图阈值，阈值越大，被认为是前景的像素就越少，默认值为150。
smooth    ：是否开启边缘平滑，默认值为False。

返回
具有rgba通道的pil.image图片，可以存为png。

示例；
img1 = Image.open(os.path.join(rootDir, "119082.jpg"))
img2 = Image.open(os.path.join(resultDir, "119082.png"))
createMaskForPicture(img1, img2, 100, True);
"""
def createMaskForPicture(ori_img, sod_img, threshold=100, smooth = True):
    ori_img = ori_img.convert('RGBA')
    img1 = np.array(ori_img)
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGBA2BGRA)
    img2 = np.array(sod_img)
    if smooth:
        img2 =  cv2.GaussianBlur(img2, (25, 25), 0)
    # plt.imshow(img2)
    # img2 = imfill(img2, threshold)
    img2 = erodeDialate(img2,threshold, 5)
    
    cnt = 0
    maxl = 0
    for i in range(len(img2)):
        for j in range(len(img2[0])):
            if img2[i][j] < threshold:
                img1[i][j][3]=0 
            else:
                maxl = max(maxl,  img2[i][j])
                #cnt+=1
    
    #result = Image.fromarray(np.uint8(img1))
    result = img1
    # plt.imshow(result)
    return result


"""抠图
参数：
ori_img   ：输入的原图片，一般是pil.image形式。
sod_img   ：原图片的显著性分析结果，一般是pil.image形式。
threshold ：图片抠图阈值，阈值越大，被认为是前景的像素就越少，默认值为150。
smooth    ：是否开启边缘平滑，默认值为False。

返回
具有rgba通道的pil.image图片，可以存为png。

示例；
img1 = Image.open(os.path.join(rootDir, "119082.jpg"))
img2 = Image.open(os.path.join(resultDir, "119082.png"))
createMaskForPicture(img1, img2, 100, True);
"""
def barrage(ori_img, sod_img, threshold=100, smooth = True):
    img2 = np.array(sod_img)
    if smooth:
        img2 = cv2.GaussianBlur(img2, (25, 25), 0)
    # plt.imshow(img2)
    img2 = erodeDialate(img2,threshold, 5)
    #result = Image.fromarray(np.uint8(img1))
    result = img2
    # plt.imshow(result)
    return result

#背景虚化辅助函数，无需调用
def edgeBlur(img1,img2, threshold = 1):
    img1 = np.array(img1)
    th, img2 = cv2.threshold(np.array(img2), threshold, 255, cv2.THRESH_BINARY) 
    imgCanny = cv2.Canny(img2, 60, 80)
    img_copy = img1.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    #plt.imshow(imgCanny)
    img_dilate = cv2.dilate(imgCanny, kernel)
    #plt.imshow(img_dilate)
    img_dilate2 = cv2.dilate(imgCanny, kernel2)
    shape = img_dilate.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if img_dilate2[i, j] == 0:  # 二维定位到三维
                img1[i][j][0] = 0
                img1[i][j][1] = 0
                img1[i][j][2] = 0
    dst = cv2.GaussianBlur(img1, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if img_dilate[i, j] != 0:  # 二维定位到三维
                img_copy[i][j] = dst[i][j]
    #plt.imshow(img_copy)
    return img_copy

#背景虚化辅助函数，无需调用
def imfill(img, threshold=1):
    th, im = cv2.threshold(np.array(img), threshold, 255, cv2.THRESH_BINARY) 
    im_floodfill = im.copy()
    h, w = im.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    #plt.imshow(im_floodfill)
    
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    #plt.imshow(im_floodfill_inv)
    im_out = im | im_floodfill_inv
    #plt.imshow(im_out)
    return im_out

#背景虚化辅助函数，无需调用
def erodeDialate(img, threshold=1, kernel_size = 5):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size,  kernel_size))  # 矩形结构
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 椭圆结构
    #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))  # 十字形结构
    th, img = cv2.threshold(np.array(img), threshold, 255, cv2.THRESH_BINARY);
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    #plt.imshow(img)
    return img

"""背景虚化
参数
ori_img     ：输入的原图片，一般是pil.image形式。
sod_img     ：原图片的显著性分析结果，一般是pil.image形式。
thresholds  ：图片抠图阈值列表，对没达到对应阈值的像素使用对应的滤波器进行模糊，以达到渐变效果。
kernel_sizes：与thresholds对应的滤波器大小。
Lambda      ：细节加强系数，越大越强，一般在0-0.1间，默认为0

返回
rgb通道的pil.image图片

示例；
background_blur(img1, img2）
background_blur(img1, img2, [150], [15])
background_blur(img1, img2, [150], [7, 9])
"""
def background_blur(ori_img, sod_img, thresholds=[100], kernel_sizes = [7], Lambda = 0):
    ori_img = ori_img.convert('RGB')
    img1 = cv2.cvtColor(np.asarray(ori_img),cv2.COLOR_RGB2BGR)
    img2 = np.array(sod_img)
    result = img1.copy()

    fg_masks = [erodeDialate(img2, threshold)< threshold for threshold in thresholds]
    
    img1 = edgeBlur(img1, img2, 100)
    #print(fg_masks)
    for i, fg_mask in enumerate(fg_masks):
        kernel_size = kernel_sizes[i]
        blurred = cv2.GaussianBlur(img1, (kernel_size, kernel_size), 0)
        result[fg_mask] = blurred[fg_mask]
    shape = img1.shape
    if Lambda != 0:
        for i in range(shape[0]):
            for j in range(shape[1]):
                result[i][j][0] += Lambda * (result[i][j][0]-img1[i][j][0])
                result[i][j][1] += Lambda * (result[i][j][1]-img1[i][j][1])
                result[i][j][2] += Lambda * (result[i][j][2]-img1[i][j][2])
            
    #result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    #print(result)
    #plt.imshow(result)
    #result = Image.fromarray(np.uint8(result))
    return result
    
    
    
    
    
"""背景替换
参数
ori_img       ：输入的原图片，一般是pil.image形式。
sod_img       ：原图片的显著性分析结果，一般是pil.image形式。
background_img：背景图，一般是pil.image形式。
maxThreshold  ：图片抠图阈值。大于这个阈值认为是前景。默认值为180。
minThreshold  : 图像抠图最小阈值。小于这个阈值认为是背景。之间的将与使用前景图的像素和背景图融合。默认值为40。

返回
rgb通道的pil.image图片

示例；
img1 = Image.open(os.path.join(rootDir, "23025.jpg"))
img2 = Image.open(os.path.join(resultDir, "23025.png"))
img3 = Image.open(os.path.join(rootDir, "27059.jpg"))
background_substitution(img1, img2, img3, 180, 40);
"""
def background_substitution(ori_img, sod_img, background_img, maxThreshold = 180, minThreshold = 80):
    ori_img = ori_img.convert('RGB')
    img1 = cv2.cvtColor(np.asarray(ori_img),cv2.COLOR_RGB2BGR)
    img2 = np.array(sod_img)
    img3 = np.array(background_img.resize((len(img2[0]),len(img2)), Image.ANTIALIAS))
    img3 = cv2.cvtColor(np.asarray(img3),cv2.COLOR_RGB2BGR)
    #print(img1[0])
    result = img1.copy()
    
    img_dilate = erodeDialate(img2, maxThreshold)
    ran = maxThreshold - minThreshold
    if ran <= 0:
        return img1
    shape = img_dilate.shape
    hsv = cv2.cvtColor(img3, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    mean1 = np.mean(np.mean(hsv, axis = 0),axis = 0)
    mean2 = np.mean(np.mean(hsv2, axis = 0),axis = 0)
    print(mean1)
    hsv3 = hsv[:,:,2]
    hsv4 = hsv2[:,:,2]
    std1 = np.std(hsv3)
    std2 = np.std(hsv4)
    print(std1)
    for i in range(shape[0]):
        for j in range(shape[1]):
            hsv[i][j][2] = (hsv[i][j][2]-mean2[2]) / std2 * std1 + mean1[2]
    img1 = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)
            
    """for i in range(shape[0]):
        for j in range(shape[1]):
            mean1 += hsv[i][j][2]
            mean2 += hsv2[i][j][2]
    mean1 = mean1/(shape[0]*shape[1])
    mean2 = mean2/(shape[0]*shape[1])
    for i in range(shape[0]):
        for j in range(shape[1]):          
            std1 += (hsv[i][j][2] - mean1)**2
            std2 += (hsv2[i][j][2] - mean1)**2
    std1 = std1 ** 0.5"""    
    for i in range(shape[0]):
        for j in range(shape[1]):
            if img_dilate[i][j] == 0:
                if img2[i][j] < minThreshold:
                    img1[i][j][0] = img3[i][j][0]
                    img1[i][j][1] = img3[i][j][1]
                    img1[i][j][2] = img3[i][j][2]
                else:
                    p = (img2[i][j] - minThreshold)/ ran
                    #print(p)
                    img1[i][j][0] = (1 - p) * img3[i][j][0]  + p * img1[i][j][0]
                    img1[i][j][1] = (1 - p) * img3[i][j][1]  + p * img1[i][j][1]
                    img1[i][j][2] = (1 - p) * img3[i][j][2]  + p * img1[i][j][2]  
    
    
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    result = Image.fromarray(np.uint8(img1))
    return result


