import os
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.decomposition import PCA
from PIL import Image
import joblib

def init_resnet50_extractor():
    """
    初始化并返回 ResNet50 模型、预处理流程和运行设备。
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载 ResNet50 预训练模型
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # 去掉全连接层，使其输出 2048 维的特征向量
    resnet50.fc = torch.nn.Identity()

    resnet50 = resnet50.to(device)
    resnet50.eval()  # 切换到评估模式

    # 定义预处理流程
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return resnet50, preprocess, device

def extract_feature(cropped_image_bgr, model, preprocess, device):
    """
    提取裁剪图像的特征向量
    """
    if cropped_image_bgr.size == 0:
        return None

    # BGR 转 RGB
    img_rgb = cv2.cvtColor(cropped_image_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    # 预处理并增加 Batch 维度: (1, C, H, W)
    input_tensor = preprocess(img_pil).unsqueeze(0).to(device)

    # 提取特征
    with torch.no_grad():
        feature = model(input_tensor)

    # 展平为 1D 数组 (长度为 2048)
    return feature.squeeze().cpu().numpy()

def train_pca_model(image_folder, n_components=32, save_path="pca_model.pkl"):
    """
    训练 PCA 模型并计算量化参数
    """
    print("初始化特征提取器...")
    model, preprocess, device = init_resnet50_extractor()

    features_list = []

    print(f"从 {image_folder} 提取 2048 维特征...")
    # 假设你的图片都放在 image_folder 目录下
    valid_extensions = ('.jpg', '.jpeg', '.png')
    for filename in os.listdir(image_folder):
        if not filename.lower().endswith(valid_extensions):
            continue

        img_path = os.path.join(image_folder, filename)
        img_bgr = cv2.imread(img_path)

        if img_bgr is not None and img_bgr.size > 0:
            # 提取 2048 维特征
            feat = extract_feature(img_bgr, model, preprocess, device)
            if feat is not None:
                features_list.append(feat)

    # 将列表转换为 NumPy 矩阵
    X = np.array(features_list)
    print(f"提取完毕，共获取了 {X.shape[0]} 个样本的特征，特征维度: {X.shape[1]}")

    if X.shape[0] < n_components:
        raise ValueError("样本数量太少！样本数必须大于预期降到的维度。")

    print(f"开始训练 PCA 模型 (降至 {n_components} 维)...")
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    print("PCA 训练完成！")

    print("计算 INT8 量化所需的动态范围 (Scale)...")
    max_val = np.percentile(np.abs(X_pca), 99.5)

    # 计算缩放比例
    scale = 127.0 / max_val
    print(f"降维特征的 99.5% 绝对值上限为: {max_val:.4f}")
    print(f"建议的 INT8 量化 Scale 乘数为: {scale:.4f}")

    print(f"保存模型到 {save_path} ...")
    # 我们把 pca 模型和算出来的 scale 一起打包保存
    model_data = {
        'pca_model': pca,
        'quantize_scale': scale
    }
    joblib.dump(model_data, save_path)
    print("保存成功！训练全部结束。")


# --- 运行训练 ---
if __name__ == "__main__":
    DATASET_FOLDER = "./PCAdata"
    train_pca_model(DATASET_FOLDER, n_components=32, save_path="v2x_pca_model.pkl")