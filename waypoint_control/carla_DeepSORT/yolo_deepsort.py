from typing import Any
import torch
import numpy as np
import cv2
from time import perf_counter
from ultralytics import YOLO
from pathlib import Path
from types import SimpleNamespace
import carla
import random
from deep_sort.deep_sort import DeepSort
from deep_sort.utils.parser import get_config


class_id = [ 2, 3, 5, 7]
class_name = { 2: 'car', 3: 'motobike', 5: 'bus' ,7: 'truck'}

img_w = 256*4
img_h = 256*3
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
output_path = "output.mp4"

class main:

    def __init__(self):
        
        self.model = self.load_model()
        print('after load model')
        self.save_vid = True
        self.output_path = "output.mp4"
        
        self.cfg = get_config()
        self.cfg.merge_from_file('deep_sort/configs/deep_sort.yaml')
        self.deepsort_weights = "deep_sort/deep/checkpoint/ckpt.t7"

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.deepsort = DeepSort(
            self.deepsort_weights,
            max_age=70
        )
        
    def load_model(self):
        model = YOLO('weights/yolov8n.pt')
        return model

    def yolo_details(self, frame):
        results = self.model(frame)
        bbox_xyxy = []
        conf_score = []
        cls_id = []          
        for box in results:
            rows = [row for row in box.boxes.data.tolist() if int(row[5]) in class_id and row[5] != 0]
            bbox_xyxy.extend([[int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2, conf, id in rows])
            conf_score.extend([conf for x1, y1, x2, y2, conf, id in rows])
            cls_id.extend([int(id) for x1, y1, x2, y2, conf, id in rows])
        return frame , bbox_xyxy , conf_score , cls_id    

    def video(self,frame,path,fps,frame_width , frame_height):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path,fourcc,fps,(frame_width,frame_height)) 
        for f in frame:
            video_writer.write(f)
          
    
    def __call__(self):
        # The local Host for carla simulator is 2000
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)
        world = client.get_world() 

        '''
       climate = carla.WeatherParameters(
                    cloudiness=50.0,
                    precipitation=90.0,
                    sun_altitude_angle=70.0,
                    wetness = 50.0,
                    fog_density = 50.0)
        
        world.set_weather(climate)
        '''
        # settings = world.get_settings()
        # settings.synchronous_mode = True  # 开启同步模式
        # settings.fixed_delta_seconds = 0.05  # 每帧 0.05 秒，20 FPS
        # world.apply_settings(settings)

        spawn_points = world.get_map().get_spawn_points()
        vehicle_bp = world.get_blueprint_library().find('vehicle.lincoln.mkz_2020')
        vehicle_bp.set_attribute('role_name', 'ego')
        ego_vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

        spectator = world.get_spectator()
        transform = carla.Transform(ego_vehicle.get_transform().transform(carla.Location(x=-4, z=2.5)), carla.Rotation(yaw=-180, pitch=-90))
        spectator.set_transform(transform)

        for i in range(60):
            vehicle_bp = random.choice(world.get_blueprint_library().filter('vehicle'))
            npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
            
        if npc:
            for v in world.get_actors().filter('*vehicle*'):
                v.set_autopilot(True)
                   
        camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', f'{img_w}')
        camera_bp.set_attribute('image_size_y', f'{img_h}')
        camera_bp.set_attribute('fov', '110')
        
        camera_location = carla.Location(2,0,1)
        camera_rotation = carla.Rotation(0,180,0)

        camera_init_trans = carla.Transform(camera_location,camera_rotation)
        camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle , attachment_type=carla.AttachmentType.SpringArmGhost)
        
        def capture_image(image):
            image_data = np.array(image.raw_data)
            image_rgb = image_data.reshape((image.height, image.width, 4))[:, :, :3]
            return image_rgb

        image_w = camera_bp.get_attribute('image_size_x').as_int()
        image_h = camera_bp.get_attribute('image_size_y').as_int()

        camera_data = {'image': np.zeros((image_h, image_w, 4))}
        camera.listen(lambda image: camera_data.update({'image': capture_image(image)}))
        
        ego_vehicle.set_autopilot(True)
        v_frame=[]

        try:
            while True:
                frame = camera_data['image']
                frame , bbox_xyxy , conf_score , cls_id = self.yolo_details(frame)              
                outputs = self.deepsort.update(bbox_xyxy, conf_score, frame)

                frame = np.array(frame)
                if len(outputs) > 0:
                    for j, (output, conf , tracked_cls_id) in enumerate(zip(outputs, conf_score,cls_id)):
                            frame = self.draw_bbox(frame, output, conf, tracked_cls_id)
    
                frame = cv2.UMat(frame)
                cv2.imshow('deepSORT', frame)
                v_frame.append(frame)
                self.video(v_frame , self.output_path  , 14.0 , img_w , img_h)
                   
                if cv2.waitKey(1) == ord('q'):
                    break 
        finally:
            self.destroy_world(camera, ego_vehicle, world)
            print('all actors destroyed')
     
    def destroy_world(self,camera , vehicle ,world):
        try:
            cv2.destroyAllWindows()
            camera.stop()
            if camera is not None:
                camera.destroy()
                camera = None
            if vehicle is not None:
                vehicle.destroy()
                vehicle = None
            for npc in world.get_actors().filter('vehicle*'):

                if npc is not None:
                    npc.destroy()
        except Exception:
            pass

    def colour_label(self , label):

        label_colour = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
        return tuple(label_colour)    
    
    def draw_bbox(self, frame, output, conf, cls_id):
        x1, y1, x2, y2 = map(int,output[0:4])
        id = int(output[4])
        label = class_name.get(cls_id, '')
        frame = np.array(frame) if not isinstance(frame, np.ndarray) else frame
        
        colour = self.colour_label(id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c_id = f'{label} {id}'
        cv2.rectangle(frame, (x1, y1),(x2,y2), colour, 1)
        cv2.rectangle(frame, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), colour, -1)
        cv2.putText(frame, c_id, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 2)

        return frame
    
if __name__ == '__main__':
    run = main()