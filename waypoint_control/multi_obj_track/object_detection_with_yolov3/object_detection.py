import os
import cv2
import numpy as np
import scipy.io as sio
from tqdm import tqdm
import h5py

# 初始化YOLO模型
model = cv2.dnn.readNet("weights/yolov3.weights", "cfg/yolov3.cfg")
classes = [line.strip() for line in open("cfg/coco.names", "r").readlines()]
layers_names = model.getLayerNames()
output_layers = [layers_names[i - 1] for i in model.getUnconnectedOutLayers()]


def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.5:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids


def process_mat_files(data_path):
    # 获取所有.mat文件
    mat_files = [f for f in os.listdir(data_path) if f.endswith('.mat')]

    for mat_file in tqdm(mat_files, desc="Processing .mat files"):
        file_path = os.path.join(data_path, mat_file)
        try:
            # 加载.mat文件
            mat_data = sio.loadmat(file_path)
            if mat_data is None:
                print(f"Warning: Could not load {mat_file}, skipping...")
                continue

            # 检查datalog变量是否存在
            if 'datalog' not in mat_data:
                print(f"Warning: 'datalog' not found in {mat_file}, skipping...")
                continue

            datalog = mat_data['datalog']

            # scipy加载的数据
            camera_data = datalog[0][0][1]
            # 处理每个相机数据
            for i in range(len(camera_data[0])):
                img_path = os.path.join(data_path, camera_data[0][i][0][0])

                if not os.path.exists(img_path):
                    print(f"Warning: Image not found - {img_path}")
                    camera_data[0][i][3] = np.zeros((0, 4))
                    continue

                # 读取图片并进行检测
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Could not read image - {img_path}")
                    if isinstance(camera_data[i], dict):
                        camera_data[i]['Detections'] = np.zeros((0, 4))
                    else:
                        camera_data[0][i][3] = np.zeros((0, 4))
                    continue

                # YOLO前向传播
                height, width = img.shape[:2]
                blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                model.setInput(blob)
                outputs = model.forward(output_layers)

                # 获取检测结果
                boxes, confs, class_ids = get_box_dimensions(outputs, height, width)

                # 应用NMS
                indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)

                # 过滤掉交通灯检测结果
                filtered_boxes = []
                if len(indexes) > 0:
                    for i in indexes.flatten():
                        if classes[class_ids[i]] != "traffic light":
                            x, y, w, h = boxes[i]
                            filtered_boxes.append([x, y, w, h])
                detections = None
                # 保存检测结果
                if len(filtered_boxes) > 0:
                    detections = np.array(filtered_boxes)
                else:
                    camera_data[0][i][3] = np.zeros((0, 4))

                camera_data[0][i][3] = detections

            datalog[0][0][1] = camera_data
            sio.savemat(file_path, {'datalog': datalog}, do_compression=True)

        except Exception as e:
            print(f"Error processing {mat_file}: {str(e)}")
            continue

    print("All files processed successfully.")


if __name__ == "__main__":

    # data_path = r"C:\Users\ASUS\Desktop\multi_obj_track\Town10HD_Opt\test_data_junc1"
    current_dir = os.path.dirname(os.path.abspath(__file__))   # 收集的相机和雷达数据目录
    data_path = os.path.join(current_dir, "test_data_junc1")
    process_mat_files(data_path)