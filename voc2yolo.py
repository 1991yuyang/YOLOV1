import os
import xml.etree.ElementTree as ET

classes = ['crazing', 'patches', 'rolled-in_scale', 'inclusion', 'scratches', 'pitted_surface']

# 将x1, y1, x2, y2转换成yolov5所需要的x, y, w, h格式
def xyxy2xywh(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2 * dw
    y = (box[1] + box[3]) / 2 * dh
    w = (box[2] - box[0]) * dw
    h = (box[3] - box[1]) * dh
    return (x, y, w, h)         # 返回的都是标准化后的值


def voc2yolo(path, output_path):
    # 可以打印看看该路径是否正
    print(path)
    print(len(os.listdir(path)))
    # 遍历每一个xml文件
    for file in os.listdir(path):
        # xml文件的完整路径, 注意：因为是路径所以要确保准确，我是直接使用了字符串拼接, 为了保险可以用os.path.join(path, file)
        label_file = os.path.join(path, file)
        # 最终要改成的txt格式文件,这里我是放在voc2007/labels/下面
        # 注意: labels文件夹必须存在，没有就先创建，不然会报错
        print(path.replace('Annotations', 'labels') + file.replace('xml', 'txt'))
        out_file = open(os.path.join(output_path, file.replace('xml', 'txt')), 'w')
        # print(label_file)

        # 开始解析xml文件
        tree = ET.parse(label_file)
        root = tree.getroot()
        size = root.find('size')            # 图片的shape值
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            # 将名称转换为id下标
            cls_id = classes.index(cls)
            # 获取整个bounding box框
            bndbox = obj.find('bndbox')
            # xml给出的是x1, y1, x2, y2
            box = [float(bndbox.find('xmin').text), float(bndbox.find('ymin').text), float(bndbox.find('xmax').text),
                float(bndbox.find('ymax').text)]

            # 将x1, y1, x2, y2转换成yolov5所需要的x_center, y_center, w, h格式
            bbox = xyxy2xywh((w, h), box)
            # 写入目标文件中，格式为 id x y w h
            out_file.write(str(cls_id) + " " + " ".join(str(x) for x in bbox) + '\n')

if __name__ == '__main__':
	# 这里要改成自己数据集路径的格式
    xml_file_dir = r'H:\defeat_data\NEU-CLS-Data-master\NEU-DET\val_anno'
    output_yolo_format_dir = r"H:\defeat_data\NEU-CLS-Data-master\NEU-DET\val_yolo_anno"
    voc2yolo(xml_file_dir, output_yolo_format_dir)

