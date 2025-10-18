import carla
import numpy
import cv2
import json

image_w = 256*4
image_h = 256*3

print('in grounf truth file')

def point_in_canvas(pos, img_h, img_w):
    """Return true if point is in canvas"""
    if (pos[0] >= 0) and (pos[0] < img_w) and (pos[1] >= 0) and (pos[1] < img_h):
        return True
    return False

def get_image_point(loc, K, w2c):
    # Calculate 2D projection of 3D coordinate

    # Format the input coordinate (loc is a carla.Position object)
    point = np.array([loc.x, loc.y, loc.z, 1])
    # transform to camera coordinates
    point_camera = np.dot(w2c, point)

    point_camera = np.array([point_camera[1], -point_camera[2], point_camera[0]]).T

    point_img = np.dot(K, point_camera)
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img

def get_groundtruth(world , camera , vehicle , image , K, K_b):
    
    timestamp_sec = image.timestamp 
    world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
    for npc in world.get_actors().filter('*vehicle*'):
            if npc.id != vehicle.id:
                bb = npc.bounding_box
                dist = npc.get_transform().location.distance(vehicle.get_transform().location)
                if dist < 50:
                    forward_vec = vehicle.get_transform().get_forward_vector()
                    ray = npc.get_transform().location - vehicle.get_transform().location
                    if forward_vec.dot(ray) > 0:
                        verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                        points_image = []
                        for vert in verts:
                            ray0 = vert - camera.get_transform().location
                            cam_forward_vec = camera.get_transform().get_forward_vector()
                            if (cam_forward_vec.dot(ray0) > 0):
                                p = get_image_point(vert, K, world_2_camera)
                            else:
                                p = get_image_point(vert, K_b, world_2_camera)
                            points_image.append(p)
                        x_min, x_max = 10000, -10000
                        y_min, y_max = 10000, -10000
                        for edge in edges:
                            p1 = points_image[edge[0]]
                            p2 = points_image[edge[1]]
                            p1_in_canvas = point_in_canvas(p1, image_h, image_w)
                            p2_in_canvas = point_in_canvas(p2, image_h, image_w)
                            if not p1_in_canvas and not p2_in_canvas:
                                continue     
                            p1_temp, p2_temp = (p1.copy(), p2.copy())
                            if not (p1_in_canvas and p2_in_canvas):
                                p = [0, 0]
                                p_in_canvas, p_not_in_canvas = (p1, p2) if p1_in_canvas else (p2, p1)
                                k = (p_not_in_canvas[1] - p_in_canvas[1]) / (p_not_in_canvas[0] - p_in_canvas[0])
                                x = np.clip(p_not_in_canvas[0], 0, image.width)
                                y = k * (x - p_in_canvas[0]) + p_in_canvas[1]
                                if y >= image.height:
                                    p[0] = (image.height - p_in_canvas[1]) / k + p_in_canvas[0]
                                    p[1] = image.height - 1
                                elif y <= 0:
                                    p[0] = (0 - p_in_canvas[1]) / k + p_in_canvas[0]
                                    p[1] = 0
                                else:
                                    p[0] = image.width - 1 if x == image.width else 0
                                    p[1] = y
                                p1_temp, p2_temp = (p, p_in_canvas)
                            x_max = p1_temp[0] if p1_temp[0] > x_max else x_max
                            x_max = p2_temp[0] if p2_temp[0] > x_max else x_max
                            x_min = p1_temp[0] if p1_temp[0] < x_min else x_min
                            x_min = p2_temp[0] if p2_temp[0] < x_min else x_min
                            y_max = p1_temp[1] if p1_temp[1] > y_max else y_max
                            y_max = p2_temp[1] if p2_temp[1] > y_max else y_max
                            y_min = p1_temp[1] if p1_temp[1] < y_min else y_min
                            y_min = p2_temp[1] if p2_temp[1] < y_min else y_min
                        if (y_max - y_min) * (x_max - x_min) > 100 and (x_max - x_min) > 20:
                            if point_in_canvas((x_min, y_min), image_h, image_w) and point_in_canvas((x_max, y_max), image_h, image_w):
                                img = np.array(img, dtype=np.uint8)
                                cv2.line(img, (int(x_min),int(y_min)), (int(x_max),int(y_min)), (0,0,255, 255), 1)
                                cv2.line(img, (int(x_min),int(y_max)), (int(x_max),int(y_max)), (0,0,255, 255), 1)
                                cv2.line(img, (int(x_min),int(y_min)), (int(x_min),int(y_max)), (0,0,255, 255), 1)
                                cv2.line(img, (int(x_max),int(y_min)), (int(x_max),int(y_max)), (0,0,255, 255), 1)
                        gt_writer.addObject('vehicle' ,timestamp_sec , x_min, y_min, x_max, y_max)    
                        ground_truth_annotations.append({  "dco": True,
                                                  "height": y_max - y_min,
                                                    "width": x_max - x_min,
                                                      "id": "vehicle",  
                                                       "y": y_min,
                                                        "x": x_min})
                        
        annotations.append({
                                "timestamp": timestamp_sec,
                                "num": image.frame,
                                "class": "frame",
                                "annotations": ground_truth_annotations
                        })
                
        gt_output = {
                    "frames": annotations,
                    "class": "video",
                    "filename": "gt.json" 
                    }
        gt_writer.save('gt.xml')

        with open('gt.json', 'w') as json_file:
            json.dump(gt_output, json_file)
            
                             

    
    
    
    