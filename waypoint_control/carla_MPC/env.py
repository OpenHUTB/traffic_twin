import atexit
import json
import logging
import os
import random
import math
import datetime
import sys
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except IndexError:
    pass

import numpy as np
import carla
from pgconfig import *

try:
    import pygame
except ImportError:
    pygame = None

def draw_waypoints(world, waypoints, z=0.5, color=(255, 0, 0), life_time=100.0):
    color = carla.Color(r=color[0], g=color[1], b=color[2], a=255)
    for w in waypoints:
        t = w.transform
        begin = t.location + carla.Location(z=z)
        world.debug.draw_point(begin, size=0.1, color=color, life_time=life_time)

class Env:
    def __init__(self, host="localhost", port=2000, dt=0.05, display_method="spec", steer_ratio = 1 / 0.7):
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)

        self.world = self.client.get_world()
        # self.world = self.client.load_world("Town10")
        self.map = self.world.get_map()
        self.ego_vehicle = None
        self.actor_list = []
        self.display = None  
        self.camera_sensor = None  
        self.hud = None  
        self.clock = None  

        self.steer_ratio = steer_ratio
        self.dt = dt

        self.original_settings = self.world.get_settings()
        self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.dt))

        self.display_method = display_method

        if self.display_method == "spec":
            self.spectator = self.world.get_spectator()

        atexit.register(self.clean)
    def init_display(self, size=(1280, 720)):
        if pygame is None:
            raise RuntimeError("pygame is not installed or failed to import.")
        pygame.init()
        pygame.display.set_caption("MPC-Controller")
        self.display = pygame.display.set_mode(size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.hud = HUD(size[0], size[1])
        self.clock = pygame.time.Clock()
        # 添加摄像头
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute('image_size_x', str(self.hud.dim[0]))
        camera_bp.set_attribute('image_size_y', str(self.hud.dim[1]))

        vehicle = self.ego_vehicle
        bound_x = 0.5 + vehicle.bounding_box.extent.x
        bound_y = 0.5 + vehicle.bounding_box.extent.y
        bound_z = 0.5 + vehicle.bounding_box.extent.z
        spawn_point = carla.Transform(
            carla.Location(x=-3.0 * bound_x, y=0.0 * bound_y, z=3.0 * bound_z),
            carla.Rotation(pitch=8.0)
        )

        self.camera_sensor = self.world.spawn_actor(
            camera_bp, spawn_point, attach_to=vehicle,
            attachment_type=carla.AttachmentType.SpringArmGhost
        )
        self.camera_sensor.listen(lambda image: self.camera_callback(image))

    def camera_callback(self,image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))   
        self.display.blit(surface, (0, 0))

    def reset(self, spawn_point=None):
        if spawn_point is None:
            spawn_point = random.choice(self.map.get_spawn_points())

        blueprint_library = self.world.get_blueprint_library()
        bp = blueprint_library.filter('model3')[0]

        self.ego_vehicle = self.world.spawn_actor(bp, spawn_point)
        self.actor_list.append(self.ego_vehicle)

        self.world.tick()

    def get_cmd(self,action):
        acc_cmd, steer_cmd = action
        max_acc = 5.0 
        max_brake = 3.0 
        max_wheel_angle = 1.22  
        steer_norm = float(np.clip(steer_cmd / max_wheel_angle, -1.0, 1.0))

        if acc_cmd >= 0:
            throttle = float(np.clip(acc_cmd / max_acc, 0.0, 1.0))
            brake = 0.0
            reverse = False
        else:
            throttle = 0.0
            brake = float(np.clip(-acc_cmd / max_brake, 0.0, 1.0))
            reverse = False
        return throttle, steer_norm, brake, reverse

    def step(self, action):
        throttle, steer_norm, brake, reverse = self.get_cmd(action)
        self.ego_vehicle.apply_control(carla.VehicleControl(
            throttle=throttle,
            steer=steer_norm,
            brake=brake,
            reverse=reverse
        ))

        self.world.tick()

        if self.display_method == "spec":
            transform = self.ego_vehicle.get_transform()
            self.spectator.set_transform(carla.Transform(
                transform.location + carla.Location(z=30),
                carla.Rotation(pitch=-90)
            ))
        elif self.display_method == "pygame" and pygame is not None:
            transform = self.ego_vehicle.get_transform() 
    def check_quit(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
    
    def clean(self):
        self.world.apply_settings(self.original_settings)
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])

class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)

        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._info_text = []
        self._show_info = True
        self._server_clock = pygame.time.Clock()


    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, env, clock):
        if not self._show_info:
            return
        
        t = env.ego_vehicle.get_transform()
        v = env.ego_vehicle.get_velocity()
        c = env.ego_vehicle.get_control()
        
        vehicles = env.world.get_actors().filter('vehicle.*')
        
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(env.ego_vehicle, truncate=20),
            'Map:     % 20s' % env.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
        ]
        
        self._info_text += [
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
        ]
        
        
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
            ]
        
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != env.ego_vehicle.id]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))


    def toggle_info(self):
        self._show_info = not self._show_info

    def render(self, display):
        if self._show_info:
            if not hasattr(self, 'info_surface'):
                self.info_surface = pygame.Surface((220, self.dim[1]))
                self.info_surface.set_alpha(40) 
                display.blit(self.info_surface, (0, 0))  

            v_offset = 4
            bar_h_offset = 100
            bar_width = 106

            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):  
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]

                if item: 
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18

def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name