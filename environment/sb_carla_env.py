import random
import time
import numpy as np
import math
import cv2
import gym
from gym import spaces
import carla


class CarEnv(gym.Env):
    def __init__(self, second_per_episode=25, show_preview=True, im_height=240, im_width=320, im_channel=3, steer_amt=1.0, fixed_delta_seconds=0.2):
        super(CarEnv, self).__init__()

        self.second_per_episode = second_per_episode
        self.show_preview = show_preview
        self.im_height = im_height
        self.im_width = im_width
        self.im_channel = im_channel
        self.steer_amt = steer_amt
        self.front_camera = None
        self.CAMERA_POS_X = 1.4
        self.CAMERA_POS_Z = 1.3

        self.action_space = spaces.MultiDiscrete([9, 4])
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(self.im_height, self.im_width, self.im_channel),
                                            dtype=np.float32)

        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(4.0)
        self.world = self.client.get_world()

        self.settings = self.world.get_settings()
        self.settings.no_rendering_mode = True
        self.settings.synchronous_mode = False
        self.settings.fixed_delta_seconds = fixed_delta_seconds
        self.world.apply_settings(self.settings)
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]

    def cleanup(self):
        for sensor in self.world.get_actors().filter("*sensor"):
            sensor.destroy()
        for actor in self.world.get_actors().filter("vehicle"):
            actor.destroy()
        cv2.destroyAllWindows()

    def step(self, action):
        self.step_counter += 1
        steer = action[0]
        throttle = action[1]

        if steer == 0:
            steer = -0.9
        elif steer == 1:
            steer = -0.25
        elif steer == 2:
            steer = -0.1
        elif steer == 3:
            steer = -0.05
        elif steer == 4:
            steer = 0.0
        elif steer == 5:
            steer = 0.05
        elif steer == 6:
            steer = 0.1
        elif steer == 7:
            steer = 0.25
        elif steer == 8:
            steer = 0.9
            # map throttle and apply steer and throttle
        if throttle == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=steer, brake=1.0))
        elif throttle == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=steer, brake=0.0))
        elif throttle == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.7, steer=steer, brake=0.0))
        else:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=steer, brake=0.0))

        if self.step_counter % 50 == 0:
            print('steer input from model:', steer, ', throttle: ', throttle)

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))

        distance_travelled = self.initial_location.distance(self.vehicle.get_location())

        cam = self.front_camera
        if self.SHOW_CAM:
            cv2.imshow('Sem Camera', cam)
            cv2.waitKey(1)

        # track steering lock duration to prevent "chasing its tail"
        lock_duration = 0
        if self.steering_lock == False:
            if steer < -0.6 or steer > 0.6:
                self.steering_lock = True
                self.steering_lock_start = time.time()
        else:
            if steer < -0.6 or steer > 0.6:
                lock_duration = time.time() - self.steering_lock_start

        reward = 0
        done = False

        if len(self.collision_hist) != 0:
            done = True
            reward = reward - 300
            self.cleanup()
        if lock_duration > 3:
            reward = reward - 150
            done = True
            self.cleanup()
        elif lock_duration > 1:
            reward = reward - 20

        if kmh < 10:
            reward = reward - 3
        elif kmh < 15:
            reward = reward - 1
        elif kmh > 40:
            reward = reward - 10
        else:
            reward = reward + 1

        if distance_travelled < 30:
            reward = reward - 1
        elif distance_travelled < 50:
            reward = reward + 1
        else:
            reward = reward + 2

        if self.episode_start + self.second_per_episode < time.time():
            done = True
            self.cleanup()
        return cam / 255.0, reward, done, {}

    def reset(self):
        self.collision_hist = []
        self.actor_list = []
        self.transform = random.choice(self.world.get_map().get_spawn_points())

        self.vehicle = None
        while self.vehicle is None:
            try:
                self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
            except:
                pass
        self.actor_list.append(self.vehicle)
        self.initial_location = self.vehicle.get_location()
        self.sem_cam = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        self.sem_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.sem_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.sem_cam.set_attribute("fov", f"90")

        camera_init_trans = carla.Transform(carla.Location(z=self.CAMERA_POS_Z, x=self.CAMERA_POS_X))
        self.sensor = self.world.spawn_actor(self.sem_cam, camera_init_trans, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(2)

        if self.SHOW_CAM:
            cv2.namedWindow('Sem Camera', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Sem Camera', self.front_camera)
            cv2.waitKey(1)
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, camera_init_trans, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.steering_lock = False
        self.steering_lock_start = None
        self.step_counter = 0
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.front_camera / 255.0

    def process_img(self, image):
        image.convert(carla.ColorConverter.CityScapesPalette)
        i = np.array(image.raw_data)
        i = i.reshape((self.im_height, self.im_width, 4))[:, :, :3]  # this is to ignore the 4th Alpha channel - up to 3
        self.front_camera = i

    def collision_data(self, event):
        self.collision_hist.append(event)
