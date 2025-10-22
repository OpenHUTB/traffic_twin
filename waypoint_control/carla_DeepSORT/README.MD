# Multi Vehicle tracking in carla simulator 

This repo host a comprehensive solution for multi-object tracking in combination with YOLO and DeepSORT in carla simulator .
YOLOV8 model is trained on Carla dataset , which is avaiable in kaggle (https://www.kaggle.com/datasets/alechantson/carladataset)

![ezgif com-optimize](https://github.com/Bsornapudi/Carla-YOLO-DeepSort-Multi-Object-Tracking/assets/48683074/c365a981-e314-4cae-b4aa-d234b3de5cfa)

Following software are required before installing required packages

1. Carla simulator : Download Carla simulator 0.9.14
2. CUDA : Download and install CUDA - https://developer.nvidia.com/cuda-downloads
3. CuDnn : Install Cuda DNN - https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
4. Anacoda : https://www.anaconda.com/download
5. SYSTEM: Linux20.04
6. Create and activate a new virtural conda env 

    ```
    conda create --name <env-name> python=3.8
    conda activate <env-name> 
        NOTE : <env-name> should be the name of your virtural env
    ```
7. Install required packages
    ```
    pip install  requirements.txt
    ```
8. Once the setup is done Run Calra.exe file to launch simulator
9. One commapnd prompt or launch jupyter from conda prompt
10. Run gt.py
11. Run evaluate.ipynb to generate MOTA and MOTP values

​       
