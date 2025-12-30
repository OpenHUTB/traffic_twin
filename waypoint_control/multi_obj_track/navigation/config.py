import carla


class IntersectionConfig:
    def __init__(self, ego_vehicle_position, camera_positions):
        self.ego_vehicle_position = ego_vehicle_position
        self.camera_positions = camera_positions


town_configurations = {
    "Town01": {
        "road_intersection_1": IntersectionConfig(
            carla.Transform(carla.Location(x=90, y=59, z=0.98), carla.Rotation(pitch=0, yaw=90, roll=0)),
            {
                "back_camera": carla.Transform(carla.Location(x=90, y=52, z=3.6),
                                               carla.Rotation(pitch=0, yaw=-90, roll=0)),
                "front_camera": carla.Transform(carla.Location(x=90, y=66, z=3.6),
                                                carla.Rotation(pitch=0, yaw=90, roll=0)),
                "right_camera": carla.Transform(carla.Location(x=86, y=52, z=3.6),
                                                carla.Rotation(pitch=0, yaw=-180, roll=0)),
                "front_right_camera": carla.Transform(carla.Location(x=86, y=66, z=3.6),
                                                      carla.Rotation(pitch=0, yaw=-178, roll=0)),
                "left_camera": carla.Transform(carla.Location(x=94, y=52, z=3.6),
                                               carla.Rotation(pitch=0, yaw=-0, roll=0)),
                "front_left_camera": carla.Transform(carla.Location(x=94, y=66, z=3.6),
                                                     carla.Rotation(pitch=0, yaw=-0, roll=0))
            }
        ),
        "road_intersection_2": IntersectionConfig(
            carla.Transform(carla.Location(x=336, y=197, z=0.98), carla.Rotation(pitch=0, yaw=90, roll=0)),
            {
                "back_camera": carla.Transform(carla.Location(x=336, y=190, z=3.6),
                                               carla.Rotation(pitch=0, yaw=-90, roll=0)),
                "front_camera": carla.Transform(carla.Location(x=336, y=204, z=3.6),
                                                carla.Rotation(pitch=0, yaw=90, roll=0)),
                "right_camera": carla.Transform(carla.Location(x=332, y=190, z=3.6),
                                                carla.Rotation(pitch=0, yaw=-180, roll=0)),
                "front_right_camera": carla.Transform(carla.Location(x=332, y=204, z=3.6),
                                                      carla.Rotation(pitch=0, yaw=-178, roll=0)),
                "left_camera": carla.Transform(carla.Location(x=340, y=190, z=3.6),
                                               carla.Rotation(pitch=0, yaw=-0, roll=0)),
                "front_left_camera": carla.Transform(carla.Location(x=340, y=204, z=3.6),
                                                     carla.Rotation(pitch=0, yaw=-0, roll=0))
            }
        ),
        "road_intersection_3": IntersectionConfig(
            carla.Transform(carla.Location(x=336, y=328, z=0.98), carla.Rotation(pitch=0, yaw=90, roll=0)),
            {
                "back_camera": carla.Transform(carla.Location(x=336, y=321, z=3.6),
                                               carla.Rotation(pitch=0, yaw=-90, roll=0)),
                "front_camera": carla.Transform(carla.Location(x=336, y=335, z=3.6),
                                                carla.Rotation(pitch=0, yaw=90, roll=0)),
                "right_camera": carla.Transform(carla.Location(x=332, y=321, z=3.6),
                                                carla.Rotation(pitch=0, yaw=-180, roll=0)),
                "front_right_camera": carla.Transform(carla.Location(x=332, y=335, z=3.6),
                                                      carla.Rotation(pitch=0, yaw=-178, roll=0)),
                "left_camera": carla.Transform(carla.Location(x=340, y=321, z=3.6),
                                               carla.Rotation(pitch=0, yaw=-0, roll=0)),
                "front_left_camera": carla.Transform(carla.Location(x=340, y=335, z=3.6),
                                                     carla.Rotation(pitch=0, yaw=-0, roll=0))
            }
        ),
        "road_intersection_4": IntersectionConfig(
            carla.Transform(carla.Location(x=336, y=59, z=0.98), carla.Rotation(pitch=0, yaw=90, roll=0)),
            {
                "back_camera": carla.Transform(carla.Location(x=336, y=52, z=3.6),
                                               carla.Rotation(pitch=0, yaw=-90, roll=0)),
                "front_camera": carla.Transform(carla.Location(x=336, y=66, z=3.6),
                                                carla.Rotation(pitch=0, yaw=90, roll=0)),
                "right_camera": carla.Transform(carla.Location(x=332, y=52, z=3.6),
                                                carla.Rotation(pitch=0, yaw=-180, roll=0)),
                "front_right_camera": carla.Transform(carla.Location(x=332, y=66, z=3.6),
                                                      carla.Rotation(pitch=0, yaw=-178, roll=0)),
                "left_camera": carla.Transform(carla.Location(x=340, y=52, z=3.6),
                                               carla.Rotation(pitch=0, yaw=-0, roll=0)),
                "front_left_camera": carla.Transform(carla.Location(x=340, y=66, z=3.6),
                                                     carla.Rotation(pitch=0, yaw=-0, roll=0))
            }
        ),
        "road_intersection_5": IntersectionConfig(
            carla.Transform(carla.Location(x=90, y=197, z=0.98), carla.Rotation(pitch=0, yaw=90, roll=0)),
            {
                "back_camera": carla.Transform(carla.Location(x=90, y=190, z=3.6),
                                               carla.Rotation(pitch=0, yaw=-90, roll=0)),
                "front_camera": carla.Transform(carla.Location(x=90, y=204, z=3.6),
                                                carla.Rotation(pitch=0, yaw=90, roll=0)),
                "right_camera": carla.Transform(carla.Location(x=86, y=190, z=3.6),
                                                carla.Rotation(pitch=0, yaw=-180, roll=0)),
                "front_right_camera": carla.Transform(carla.Location(x=86, y=204, z=3.6),
                                                      carla.Rotation(pitch=0, yaw=-178, roll=0)),
                "left_camera": carla.Transform(carla.Location(x=94, y=190, z=3.6),
                                               carla.Rotation(pitch=0, yaw=-0, roll=0)),
                "front_left_camera": carla.Transform(carla.Location(x=94, y=204, z=3.6),
                                                     carla.Rotation(pitch=0, yaw=-0, roll=0))
            }
        )
    },
    "Town10HD_Opt": {
        "road_intersection_1": IntersectionConfig(
            carla.Transform(carla.Location(x=-46, y=21, z=0.98), carla.Rotation(pitch=0, yaw=90, roll=0)),
            {
                "back_camera": carla.Transform(carla.Location(x=-46, y=14, z=3.6),
                                               carla.Rotation(pitch=0, yaw=-90, roll=0)),
                "front_camera": carla.Transform(carla.Location(x=-46, y=28, z=3.6),
                                                carla.Rotation(pitch=0, yaw=90, roll=0)),
                "right_camera": carla.Transform(carla.Location(x=-50, y=14, z=3.6),
                                                carla.Rotation(pitch=0, yaw=-178, roll=0)),
                "front_right_camera": carla.Transform(carla.Location(x=-50, y=28, z=3.6),
                                                      carla.Rotation(pitch=0, yaw=-178, roll=0)),
                "left_camera": carla.Transform(carla.Location(x=-42, y=14, z=3.6),
                                               carla.Rotation(pitch=0, yaw=-0, roll=0)),
                "front_left_camera": carla.Transform(carla.Location(x=-42, y=28, z=3.6),
                                                     carla.Rotation(pitch=0, yaw=-0, roll=0))
            }
        ),
        "road_intersection_2": IntersectionConfig(
            carla.Transform(carla.Location(x=104, y=21, z=0.98), carla.Rotation(pitch=0, yaw=90, roll=0)),
            {
                "back_camera": carla.Transform(carla.Location(x=104, y=14, z=3.6),
                                               carla.Rotation(pitch=0, yaw=-90, roll=0)),
                "front_camera": carla.Transform(carla.Location(x=104, y=28, z=3.6),
                                                carla.Rotation(pitch=0, yaw=90, roll=0)),
                "right_camera": carla.Transform(carla.Location(x=100, y=14, z=3.6),
                                                carla.Rotation(pitch=0, yaw=-178, roll=0)),
                "front_right_camera": carla.Transform(carla.Location(x=100, y=28, z=3.6),
                                                      carla.Rotation(pitch=0, yaw=-178, roll=0)),
                "left_camera": carla.Transform(carla.Location(x=108, y=14, z=3.6),
                                               carla.Rotation(pitch=0, yaw=-0, roll=0)),
                "front_left_camera": carla.Transform(carla.Location(x=108, y=28, z=3.6),
                                                     carla.Rotation(pitch=0, yaw=-0, roll=0))
            }
        ),
        "road_intersection_3": IntersectionConfig(
            carla.Transform(carla.Location(x=-106, y=21, z=0.98), carla.Rotation(pitch=0, yaw=90, roll=0)),
            {
                "back_camera": carla.Transform(carla.Location(x=-106, y=14, z=3.6),
                                               carla.Rotation(pitch=0, yaw=-90, roll=0)),
                "front_camera": carla.Transform(carla.Location(x=-106, y=28, z=3.6),
                                                carla.Rotation(pitch=0, yaw=90, roll=0)),
                "right_camera": carla.Transform(carla.Location(x=-110, y=14, z=3.6),
                                                carla.Rotation(pitch=0, yaw=-178, roll=0)),
                "front_right_camera": carla.Transform(carla.Location(x=-110, y=28, z=3.6),
                                                      carla.Rotation(pitch=0, yaw=-178, roll=0)),
                "left_camera": carla.Transform(carla.Location(x=-102, y=14, z=3.6),
                                               carla.Rotation(pitch=0, yaw=-0, roll=0)),
                "front_left_camera": carla.Transform(carla.Location(x=-102, y=28, z=3.6),
                                                     carla.Rotation(pitch=0, yaw=-0, roll=0))
            }
        ),
        "road_intersection_4": IntersectionConfig(
            carla.Transform(carla.Location(x=-46, y=-61, z=0.98), carla.Rotation(pitch=0, yaw=90, roll=0)),
            {
                "back_camera": carla.Transform(carla.Location(x=-46, y=-68, z=3.6),
                                               carla.Rotation(pitch=0, yaw=-90, roll=0)),
                "front_camera": carla.Transform(carla.Location(x=-46, y=-54, z=3.6),
                                                carla.Rotation(pitch=0, yaw=90, roll=0)),
                "right_camera": carla.Transform(carla.Location(x=-50, y=-68, z=3.6),
                                                carla.Rotation(pitch=0, yaw=-178, roll=0)),
                "front_right_camera": carla.Transform(carla.Location(x=-50, y=-54, z=3.6),
                                                      carla.Rotation(pitch=0, yaw=-178, roll=0)),
                "left_camera": carla.Transform(carla.Location(x=-42, y=-68, z=3.6),
                                               carla.Rotation(pitch=0, yaw=-0, roll=0)),
                "front_left_camera": carla.Transform(carla.Location(x=-42, y=-54, z=3.6),
                                                     carla.Rotation(pitch=0, yaw=-0, roll=0))
            }
        ),
        "road_intersection_5": IntersectionConfig(
            carla.Transform(carla.Location(x=-46, y=135, z=0.98), carla.Rotation(pitch=0, yaw=90, roll=0)),
            {
                "back_camera": carla.Transform(carla.Location(x=-46, y=128, z=3.6),
                                               carla.Rotation(pitch=0, yaw=-90, roll=0)),
                "front_camera": carla.Transform(carla.Location(x=-46, y=142, z=3.6),
                                                carla.Rotation(pitch=0, yaw=90, roll=0)),
                "right_camera": carla.Transform(carla.Location(x=-50, y=128, z=3.6),
                                                carla.Rotation(pitch=0, yaw=-178, roll=0)),
                "front_right_camera": carla.Transform(carla.Location(x=-50, y=142, z=3.6),
                                                      carla.Rotation(pitch=0, yaw=-178, roll=0)),
                "left_camera": carla.Transform(carla.Location(x=-42, y=128, z=3.6),
                                               carla.Rotation(pitch=0, yaw=-0, roll=0)),
                "front_left_camera": carla.Transform(carla.Location(x=-42, y=142, z=3.6),
                                                     carla.Rotation(pitch=0, yaw=-0, roll=0))
            }
        )
    }
}