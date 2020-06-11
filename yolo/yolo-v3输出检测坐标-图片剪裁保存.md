# yolo-v3输出检测坐标-图片剪裁保存



```python
# _*_coding:utf-8 _*_

# @Time　　   :2020/6/11   2:35 下午
# @Author　   : dashuai
# @ File　　  :get_word.py
# @Software   :PyCharm

# 通过模型获取坐标，剪裁图片上汉字

import cv2
from os import path
import os
import uuid
from darknet import load_net, load_meta, detect


net = load_net("/mengh.cfg".encode(), "/mengh_final.weights".encode(), 0)
meta = load_meta("/mengh.data".encode())


# 待切割图片集
base_dir = ""

# 切割完储存目录
target_dir = ""


def get_coor(img):
    r = detect(net, meta, img.encode())
    return r


def get_info():
    files = [i for i in os.listdir(base_dir) if i.endswith(".jpg")]
    success_count = 0
    print(f"---------共 {len(files)} 张图片----------")
    for f in files:
        img_path = os.path.join(base_dir, f)
        ress = get_coor(img_path)
        if len(ress) == 4:
            success_count += 1
            print(ress)
            seg_img(img_path, ress)
            break
    print(f"------处理完成，一共 {len(files)} 张，成功 {success_count} 张------")


def fix(xmin, ymin, xmax, ymax, w, h):
  	"""
  	修正坐标数据，保证坐标值在图片的宽高范围内
  	:param xmin: 最小x值
    :param ymin: 最小y值
    :param xmax: 最大x值
    :param ymax: 最大y值
    :param w: 图片宽
    :param h: 图片高
    :return: 
  	"""
    x1 = 0 if xmin < 0 else xmin
    y1 = 0 if ymin < 0 else ymin
    x2 = w if xmax > w else xmax
    y2 = h if ymax > h else ymax
    return x1, y1, x2, y2


def seg_img(img, rets):
    """
    先从模型获取坐标信息，再将目标切割保存成小图
    :param img: 待检测图片地址
    :param rets: yolo检测输出的信息
    :return: 
    """
    # 图片文件名为标注后名称：休息休息.jpg/休息休息_uuid.jpg
    img_base_name = os.path.basename(img)
    # 获取标注的汉字列表
    if not "_" in img_base_name:
        hanzi_list = list(img_base_name.split(".")[0])
    else:
        hanzi_list = list(img_base_name.split("_")[0])
    img = cv2.imread(img)
    i = 0
    shape = img.shape
    h = shape[0]
    w = shape[1]
    # 按照每个目标的x坐标值，从小大大排序，保证有序输出，这样才能和文件名一一对应
    rets.sort(key=lambda x: x[2][0])
    for res in rets:
        if res[1] > 0.5:
            # 将yolo输出的信息转换为坐标
            """
            [(b'h', 0.9784342646598816, (53.67627716064453, 20.701627731323242, 33.18819808959961, 37.7593994140625)), (b'h', 0.9729443192481995, (82.88678741455078, 19.116840362548828, 34.920902252197266, 34.44325256347656)), (b'h', 0.8594143390655518, (19.21661376953125, 23.809219360351562, 33.73884582519531, 37.02143478393555)), (b'h', 0.7929879426956177, (117.5282211303711, 16.859752655029297, 37.63766860961914, 36.864078521728516))]
            """
            x1 = int(res[2][0] - (res[2][2] / 2))
            y1 = int(res[2][1] - (res[2][3] / 2))
            x2 = x1 + int(res[2][2])
            y2 = y1 + int(res[2][3])
            # 对坐标进行修正，数值不能超过图片宽高
            x1, y1, x2, y2 = fix(x1, y1, x2, y2, w, h)
            try:
                hanzi_img = img[y1:y2, x1:x2]   # 将检测到的目标切割出来
                path = os.path.join(target_dir, f"{hanzi_list[i]}_{uuid.uuid1().hex}.jpg")
                cv2.imwrite(path, hanzi_img)    # 保存
                i += 1
            except Exception as e:
                print('#' * 20)
                print(e)
                print('存在不规则的图片')


if __name__ == '__main__':
    get_info()



```

