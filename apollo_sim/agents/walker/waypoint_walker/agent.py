import copy
import math
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the non-interactive backend
import matplotlib.pyplot as plt

from threading import Thread
from collections import deque
from typing import Dict
from shapely.geometry import Point

from apollo_sim.actor import ActorClass, Waypoint
from apollo_sim.actor.control import WalkerControl
from apollo_sim.agents.walker.waypoint_walker.config import WaypointWalkerConfig, get_basic_config
from apollo_sim.tools import get_instance_logger
from apollo_sim.sim_env import SimEnv

class WaypointWalkerAgent(object):

    frequency = 25.0

    prefix = 'waypoint_walker'

    MIN_DISTANCE_PERCENTAGE = 0.9

    def __init__(
            self,
            actor: 'ActorClass',
            config: WaypointWalkerConfig,
            sim_env: 'SimEnv',
            output_folder: str,
            scenario_idx: str,
            parameters: Dict = get_basic_config(),
            # other configs
            debug: bool = False
    ):
        self.sim_env = sim_env
        self.actor = actor
        self.config = config
        self.output_folder = output_folder
        self.scenario_idx = scenario_idx
        self.debug = debug
        self.debug_folder = os.path.join(output_folder, f"debug/{self.prefix}")
        self.route = self.config.behavior

        # 1. inner parameters' configuration
        self._ignore_vehicle = parameters.get('ignore_vehicle', False)
        self._ignore_walker = parameters.get('ignore_walker', False)
        self._ignore_static_obstacle = parameters.get('ignore_static_obstacle', False)
        self._ignore_traffic_light = parameters.get('ignore_traffic_light', False)
        self._max_speed = parameters.get('max_speed', 12.0)
        self._max_speed_junction = parameters.get('max_speed_junction', 10.0)
        self._min_distance = parameters.get('min_distance', 1.0) * self.MIN_DISTANCE_PERCENTAGE
        self._max_acceleration = parameters.get('max_acceleration', 6.0)
        self._max_deceleration = parameters.get('max_deceleration', -6.0)
        self._max_steering = parameters.get('max_steering', 0.8)
        self._collision_threshold = parameters.get('collision_threshold', 5.0)
        self._finish_buffer = parameters.get('finish_buffer', 20.0)
        self._remove_after_finish = parameters.get('remove_after_finish', False)
        self._collision_distance_threshold = parameters.get('collision_distance_threshold', 5.0)

        # 2. auto config
        self._buffer_size = 5
        self._initial_waypoint = self.route[0]

        self._waypoints_queue = deque(maxlen=5000)
        self._waypoint_buffer = deque(maxlen=self._buffer_size)
        # set waypoints
        for i, waypoint in enumerate(self.route):
            if i == 0:
                continue
            self._waypoints_queue.append(waypoint)

        self._finish_time = 0.0

        if self.debug:
            if not os.path.exists(self.debug_folder):
                os.makedirs(self.debug_folder)

            log_file = os.path.join( self.debug_folder, f"{self.prefix}_{self.actor.id}.log")
            if os.path.exists(log_file):
                os.remove(log_file)
            self.logger = get_instance_logger(f"{self.prefix}_{self.actor.id}", log_file)
            self.logger.info(f"Logger initialized for {self.prefix}_{self.actor.id}")
            self.plan_x = []
            self.plan_y = []
            for i, waypoint in enumerate(self.route):
                self.plan_x.append(waypoint.location.x)
                self.plan_y.append(waypoint.location.y)

            self.actual_x = []
            self.actual_y = []

        self.step = 0
        self.draw_debug = False
        self.last_game_time = self.sim_env.game_time
        self.last_control = WalkerControl(0.0, 0.0)

        # async
        self.running = False
        self.thread_run = None

        # other flags
        self.scenario_idx = None

    def is_finished(self) -> bool:
        if self._finish_time > self._finish_buffer:
            return True
        else:
            return False

    def _async_run(self):
        while (not self.sim_env.termination) and self.running:
            # update state
            self.tick()
            time.sleep(1 / self.frequency)

    def start(self):
        self.running = True
        self.thread_run = Thread(target=self._async_run)
        self.thread_run.setDaemon(True)
        self.thread_run.start()

    def stop(self):
        self.running = False
        if self.thread_run is not None:
            self.thread_run.join()
            self.thread_run = None

    def tick(
            self,
    ):

        self.step += 1
        curr_location = Point([self.actor.location.x, self.actor.location.y])

        if self.debug:
            self.logger.info(f'=============Start {self.step}=============')
            self.logger.info(f"waypoint_queue length: {len(self._waypoints_queue)}")
            self.logger.info(f"waypoint_buffer length: {len(self._waypoint_buffer)}")
            self.logger.info(f"remain roue length: {len(self._waypoints_queue)}")

        # 1. if the queue is empty, stop the car
        if len(self._waypoints_queue) == 0 and len(self._waypoint_buffer) == 0:
            acceleration = (0 - self.actor.speed) / (1 / self.frequency)
            walker_control = WalkerControl(acceleration, self.last_control.heading)
            self.actor.apply_control(walker_control)
            self._finish_time += (self.sim_env.game_time - self.last_game_time)  # TODO: change to game time
            self.last_game_time = self.sim_env.game_time
            if self.debug:
                plt.figure()
                plt.plot(self.plan_x, self.plan_y, 'b-')
                plt.plot(self.actual_x, self.actual_y, 'r-')
                plt.savefig(f"{self.debug_folder}/{self.prefix}_{self.actor.id}_traj.png")
                plt.close()
            return

        # 2. buffering the waypoints
        if not self._waypoint_buffer:
            for _ in range(self._buffer_size):
                if self._waypoints_queue:
                    self._waypoint_buffer.append(
                        self._waypoints_queue.popleft())
                else:
                    break

        # 4. run with target_waypoint
        target_waypoint = copy.deepcopy(self._waypoint_buffer[0])
        target_speed = target_waypoint.speed

        # 4. calculate planning speed
        # 4.1 detect collision
        obstacles = self.sim_env.actors  # obtain observation from the traffic bridge
        hazard_detected = self._obstacle_detected(obstacles, 1 / self.sim_env.frequency)
        if hazard_detected:
            target_speed = 0.0

        target_waypoint.speed = target_speed
        # 4.2 detect traffic light
        # TODO: should be implemented

        # Run control
        next_acceleration, next_heading = self._run_control(
            target_waypoint,
            1 / self.frequency
        )  # current acceleration & steering

        walker_control = WalkerControl(acceleration=next_acceleration, heading=next_heading)


        if self.debug:
            self.logger.info(f"target speed: {target_waypoint.speed}")
            self.logger.info(f"target heading: {next_heading}")
            self.logger.info(f"target heading (waypoint): {target_waypoint.location.yaw}")
            self.logger.info(f"current heading: {self.actor.location.yaw}")
            self.logger.info(
                f"target_waypoint: {target_waypoint.lane.id} ({target_waypoint.location.x}, {target_waypoint.location.y})")
            self.logger.info(f"current_waypoint: ({self.actor.location.x}, {self.actor.location.y})")
            self.logger.info(
                f"distance: {((self.actor.location.x - target_waypoint.location.x) ** 2 + (self.actor.location.y - target_waypoint.location.y) ** 2) ** 0.5}")
            self.logger.info(f"hazard_detected: {hazard_detected}")
            self.logger.info(f"next_acceleration: {next_acceleration}")
            self.logger.info(f"next_heading: {next_heading}")
            self.logger.info(f"current_speed: {self.actor.speed}")
            self.logger.info('=============End=============')
            self.actual_x.append(self.actor.location.x)
            self.actual_y.append(self.actor.location.y)
            plt.figure()
            plt.plot(self.plan_x, self.plan_y, 'b-')
            plt.plot(self.actual_x, self.actual_y, 'r-')
            plt.savefig(f"{self.debug_folder}/{self.prefix}_{self.actor.id}_traj.png")
            plt.close()

        # 5. purge the queue of obsolete waypoints
        max_index = -1
        min_wp_distance = max(self.actor.speed * 2.0 * self.MIN_DISTANCE_PERCENTAGE, self._min_distance)
        for i, waypoint in enumerate(self._waypoint_buffer):
            waypoint_location = Point([waypoint.location.x, waypoint.location.y])
            if waypoint_location.distance(curr_location) < min_wp_distance:
                max_index = i
        if max_index >= 0:
            for i in range(max_index + 1):
                self._waypoint_buffer.popleft()

        self.actor.apply_control(walker_control)
        self.last_control = copy.deepcopy(walker_control)
        self.last_game_time = self.sim_env.game_time


    def _run_control(
            self,
            target_waypoint: Waypoint,
            delta_time: float
    ):
        curr_speed = self.actor.speed

        # calculate the acceleration
        next_acceleration = (target_waypoint.speed - curr_speed) / delta_time

        # calculate the steering.
        target_heading = math.atan2(
            target_waypoint.location.y - self.actor.location.y,
            target_waypoint.location.x - self.actor.location.x
        )
        next_heading = target_heading

        if self.debug:
            self.logger.info(f"target heading (calculate): {target_heading}")
        return next_acceleration, next_heading

    def _obstacle_detected(
            self,
            actor_dict: Dict[str, ActorClass],
            delta_t: float
    ):

        # TODO: Update to version 2
        # Total distance traveled
        # distance = curr_state.speed * time_to_stop - 0.5 * self._max_acceleration * time_to_stop ** 2
        distance = self.actor.speed * delta_t + 0.5 * self.actor.acceleration * delta_t ** 2
        distance = float(np.clip(distance, 0.0, None))
        brake_distance = self._collision_distance_threshold + distance * 1.2  # >= self._collision_vehicle_threshold

        buffer_polygon, _, _ = self.actor.get_polygon(buffer=brake_distance)

        # Step 1: get current points and future polygons
        curr_obs_polygon, _, _ = self.actor.get_polygon()
        curr_bbs = [curr_obs_polygon]
        curr_location = Point([self.actor.location.x, self.actor.location.y])
        for i, waypoint in enumerate(self._waypoint_buffer):
            waypoint_location = Point([waypoint.location.x, waypoint.location.y])
            if waypoint_location.distance(curr_location) > brake_distance:
                break
            tmp_state = copy.deepcopy(self.actor)
            tmp_state.x = waypoint.location.x
            tmp_state.y = waypoint.location.y
            tmp_state.heading = waypoint.location.heading
            tmp_state_polygon, _, _ = tmp_state.get_polygon()
            curr_bbs.append(tmp_state_polygon)

        for actor_id, actor in actor_dict.items():
            # self.logger.debug(f'vehicle id: {vehicle.id}, curr id: {curr_state.id}')
            if actor_id == self.actor.id:
                continue

            if actor.category.split('.')[0] == 'walker':
                continue

            if actor.category.split('.')[0] == 'static':
                continue

            if self._ignore_vehicle and actor.category.split('.')[0] == 'vehicle':
                continue


            obstacle_polygon, _, _ = actor.get_polygon()
            if buffer_polygon.intersects(obstacle_polygon):
                return True
            for curr_polygon in curr_bbs:
                if curr_polygon.intersects(obstacle_polygon):
                    return True

        return False