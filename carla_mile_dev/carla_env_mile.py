"""
An env for CIL method testing.

The state manager is state_manager_CIL.py

============================================================
This script is modified based on

This env is for both training and testing in various scenarios.

The scenario is created using the config dict.

The BEV image is added to the state manager.

"""
"""
This script is designed for the generalization experiments.
A new traffic flow manager will be used.
"""

import os
import gym
import numpy as np
import random
from datetime import datetime
from collections import deque
import h5py

import carla

# ====================   CARLA modules   ====================
from carla_mile_dev.BasicEnv import BasicEnv

# TODO check if keep this module
# # original
# from gym_carla.util_development.sensors import Sensors

# using the fixed version
from gym_carla.util_development.sensors import Sensors2 as Sensors

from gym_carla.util_development.kinetics import get_transform_matrix
from gym_carla.util_development.util_junction import get_junction_by_location

from gym_carla.modules.trafficflow.traffic_flow_manager_multi_task_2 import TrafficFlowManagerMultiTask2
from gym_carla.modules.traffic_lights.traffic_lights_manager4 import TrafficLightsManager
from gym_carla.modules.route_generator.junction_route_manager2 import JunctionRouteManager

# MILE modules
from carla_gym.core.obs_manager.obs_manager_handler import ObsManagerHandler
from carla_gym.core.task_actor.ego_vehicle.ego_vehicle_handler import EgoVehicleHandler
from carla_gym.core.task_actor.common.task_vehicle import TaskVehicle


DEBUG_TRAFFICFLOW_MANAGER = False

TIMESTAMP = "{0:%Y-%m-%d-Time%H-%M-%S}".format(datetime.now())


class CarlaEnvMile(gym.Env):
    # debug carla world reload
    # reload_freq = int(1_000)
    reload_freq = -1

    traffic_flow_manager_cls = TrafficFlowManagerMultiTask2
    traffic_lights_manager_cls = TrafficLightsManager
    route_manager_cls = JunctionRouteManager

    # todo add setter APIs
    # critical parameters for experiments
    simulator_timestep_length = 0.05
    # simulator_timestep_length = 0.02

    # frame numbers in each RL step
    # frame_skipping_factor = int(2)
    frame_skipping_factor = int(1)

    # max episode time in seconds
    max_episode_time = 60  # default episode length in seconds, og 60
    # max target speed of ego vehicle
    ego_max_speed = 15  # in m/s

    # ================ reward tuning ================
    # CAUTION: the key element must agree
    # todo add a check
    reward_dict = {
        'collision': -350.,
        'time_exceed': -100.,
        'success': 150.,
        # og
        'step': -0.3,
        # 'step': 0.,
    }

    # # these are deprecated
    # traffic_clear_period = int(100)
    # reload_world_period = int(100)

    # route option for ego vehicle
    # available route options
    route_options = [
        'left',
        'right',
        'straight',  # refers to straight_0

        # todo fix route option
        # straight_0
        # straight_1 is deprecated in this env
    ]

    # all available routes and spawn points
    route_info = {
        'left': {
            'ego_route': None,
            'spawn_point': None,
            'end_point': None,
            'route_length': None,
        },
        'right': {
            'ego_route': None,
            'spawn_point': None,
            'end_point': None,
            'route_length': None,
        },
        'straight': {
            'ego_route': None,
            'spawn_point': None,
            'end_point': None,
            'route_length': None,
        },
        'straight_0': {
            'ego_route': None,
            'spawn_point': None,
            'end_point': None,
            'route_length': None,
        },
        'straight_1': {
            'ego_route': None,
            'spawn_point': None,
            'end_point': None,
        },
    }

    # ------------------------------------------------------------
    # parameters for state representation
    # max historical trajectory length, for both ego and npc vehicles
    max_traj_length = int(20)
    # max frame waypoints
    graph_nodes_number = int(200)
    # waypoint distance
    waypoint_distance = 3.

    padding_length = int(50)  # padding the adjacent dict of state space
    # max NPC vehicle number for retrieving state
    state_vehicle_number = int(20)

    def __init__(
            self,
            config,
            # town='Town03',
            carla_port=2000,
            tm_port=int(8100),
            tm_seed=int(0),  # seed for autopilot controlled NPC vehicles

            train=True,  # training mode or evaluation mode
            collision_prob=0.99,  # collision prob for the test mode

            task_option='left',  # all available route options stored in the class attributes
            attention=False,

            initial_speed=1,  # if set an initial speed to ego vehicle
            state_noise=False,  # if adding noise on state vector

            sync_mode=True,
            no_render_mode=False,
            debug=False,
            verbose=False,

            collision_stop=False,
            use_proceed_reward=False,

            obs_configs=None,
            reward_configs=None,
            terminal_configs=None,
    ):
        """

        param collision_stop: if True, the episode will stop immediately after collision happens,
        else (False) the episode will continue running.
        """

        # todo add doc to denote the settings of the training mode
        self.train = train  # if using training mode
        self.collision_prob = collision_prob

        # task option and route option
        self.task_option = task_option
        if self.task_option == 'multi_task':
            self.multi_task = True
            self.route_option = 'left'  # current route in str, one of the route_options
        else:
            self.multi_task = False
            self.route_option = self.task_option

        self.task_code = None

        # params
        self.attention = attention  # if using attention mechanism

        self.initial_speed = initial_speed  # initial speed of ego vehicle when reset
        self.state_noise = state_noise  # noise on state vector

        self.sync_mode = sync_mode
        self.no_render_mode = no_render_mode
        self.debug = debug  # debug mode
        self.verbose = verbose  # visualization

        # # todo fix this
        # if self.train:
        #     frame_skipping_factor = int(2)
        # else:
        #     frame_skipping_factor = int(1)

        """
        use the frame skipping in training mode

        frame_skipping_factor is denoted as: f = N / n = t / T = F/ f
        in which:
             - N, T, F refers to (simulation) system params
             - n, t, f refers to RL module params
        """

        # RL timestep length
        self.rl_timestep_length = self.frame_skipping_factor * self.simulator_timestep_length
        # max episode timestep number for RL
        self.max_episode_timestep = int(self.max_episode_time / self.rl_timestep_length)

        self.config = config
        self.town = self.config['map']
        self.carla_port = carla_port
        self.tm_port = tm_port  # port number of the traffic manager
        self.tm_seed = tm_seed  # seed value for the traffic manager

        # todo add args to set
        #  - map for different scenarios
        #  - client timeout
        self.carla_env = BasicEnv(
            town=self.town,
            host='localhost',
            port=self.carla_port,
            client_timeout=20.0,
            timestep=self.simulator_timestep_length,
            tm_port=self.tm_port,
            sync_mode=self.sync_mode,
            no_render_mode=self.no_render_mode,
        )

        # get carla API
        self.carla_api = self.carla_env.get_env_api()
        self.client = self.carla_api['client']
        self.world = self.carla_api['world']
        self.map = self.carla_api['map']
        self.debug_helper = self.carla_api['debug_helper']  # carla.Debughelper is for visualization
        self.blueprint_library = self.carla_api['blueprint_library']
        self.spectator = self.carla_api['spectator']
        self.traffic_manager = self.carla_api['traffic_manager']

        # set the spectator on the top of junction
        # self.carla_env.set_spectator_overhead(junction_center, yaw=270, h=70)

        # set spectator and junction with config file
        location = self.config['spectator']['location']
        yaw = self.config['spectator']['yaw']
        self.junction_center = carla.Location(x=location[0], y=location[1], z=0.00)
        self.carla_env.set_spectator_overhead(self.junction_center, yaw=yaw, h=50.)

        self.junction = get_junction_by_location(
            carla_map=self.map,
            location=self.junction_center,
        )

        # todo merge waypoints buffer into local planner
        # ==================================================
        # ----------   waypoint buffer begin  ----------
        # a queue to store original route
        self._waypoints_queue = deque(maxlen=100000)  # maximum waypoints to store in current route
        # buffer waypoints from queue, get nearby waypoint
        self._buffer_size = 10
        self._waypoint_buffer = deque(maxlen=self._buffer_size)

        # todo near waypoints is waypoint popped out from buffer
        # self.near_waypoint_queue = deque(maxlen=50)
        # ----------   waypoint buffer end  ----------
        # ==================================================

        # ================   carla modules  ================

        # use the traffic flow manager to generate the ego route
        # # ================   route manager   ================
        # self.route_manager = self.route_manager_cls(
        #     carla_api=self.carla_api,
        #     junction=self.junction,
        #     route_distance=(3., 3.),
        # )
        #
        # # generate all routes for each route option
        # # route format: <list>.<tuple>.(transform, RoadOption)
        # for route in ['left', 'right', 'straight', 'straight_0']:
        #     ego_route, spawn_point, end_point, route_length = self.route_manager.get_route(route_option=route)
        #
        #     self.route_info[route]['ego_route'] = ego_route
        #     self.route_info[route]['spawn_point'] = spawn_point
        #     self.route_info[route]['end_point'] = end_point
        #     self.route_info[route]['route_length'] = route_length

        self.ego_route = []  # list of route waypoints
        self.spawn_point = None
        self.end_point = None
        self.route_length = None

        # attributes for ego routes management
        self.route_seq = []
        self.route_seq_index = None
        # set a route sequence for different task setting
        self.set_route_sequence()

        # ================   traffic flow manager   ================
        # todo use 2 types of traffic flow, CARLA Autopilot and AEB
        self.traffic_flow_manager = self.traffic_flow_manager_cls(
            carla_api=self.carla_api,
            junction=self.junction,
            scenario_config=self.config,

            tm_port=self.tm_port,
            tm_seed=self.tm_seed,

            # tls_red_phase_duration=20.,  # duration time of red phase of traffic lights
            phase_time=self.traffic_lights_manager_cls.phase_time,

            # debug=False,
            debug=DEBUG_TRAFFICFLOW_MANAGER,
            verbose=False,
        )

        # TODO add semantic tag to the ego route config file
        route = 'left'
        ego_route, spawn_point, end_point, route_length = self.traffic_flow_manager.get_ego_route()

        self.route_info[route]['ego_route'] = ego_route
        self.route_info[route]['spawn_point'] = spawn_point
        self.route_info[route]['end_point'] = end_point
        self.route_info[route]['route_length'] = route_length

        self.spawn_waypoint = spawn_point
        self.end_waypoint = end_point

        # update route info of ego vehicle
        self.update_route_info()

        # todo add params decay through training procedure
        #  - use some open-source codes?
        # collision detect decay
        param = {
            'initial_value': 0.75,
            'target_value': .95,
            'episode_number': int(3000),
            'scheduler': 'linear',
        }

        # todo fix the API of traffic flow params decay
        # self.collision_prob_decay = collision_prob_decay
        # if not self.collision_prob_decay:
        #     collision_prob = 1.  # todo add arg to set this value
        #     self.traffic_flow.set_collision_probability(collision_prob)
        #
        # self.tf_params_decay = tf_params_decay

        # ================   traffic_light_manager   ================
        # todo add API to set duration time of traffic lights
        self.traffic_light_manager = self.traffic_lights_manager_cls(
            carla_api=self.carla_api,
            junction=self.junction,

            scenario_config=self.config,

            # todo check this arg and training setting
            use_tls_control=True,
        )

        # ================   state manager   ================
        # # todo fix multi-task setting in lane graph RL methods
        # multi_task = True if self.route_option == 'multi_task' else False

        # additional reward for ego vehicle proceeding
        self.use_proceed_reward = use_proceed_reward

        # ================   about vehicles   ================
        self.ego_vehicle = None
        self.ego_id = None  # id is set by the carla, cannot be changed by user

        self.ego_location = None
        self.ego_transform = None
        # since we use frame skipping, the ego location of RL step need to be recorded
        self.ego_step_location = None

        # TODO check if deprecated
        # # original impl
        # self.ego_collision_sensor = None  # collision sensor for ego vehicle

        # management for all sensors
        self.ego_sensors = None

        self.collision_flag = False  # flag to identify a collision with ego vehicle

        self.npc_vehicles = []  # active NPC vehicles
        self.actor_list = []  # list contains all carla actors

        # ================   episodic info   ================
        # state of current timestep
        self.state = None  # state array of current timestep
        self.step_reward = None  # reward of current timestep
        self.episode_reward = 0  # accumulative reward of this episode
        self.action_array = None  # action of current timestep
        self.running_state = None  # running state of current timestep, [running, success, collision, time_exceed]

        self.frame_id = None
        self.elapsed_episode_number = int(0)  # a counter for episodes
        self.elapsed_timestep = int(0)  # elapsed timestep number of current episode
        self.elapsed_time = 0.  # elapsed time number of current episode

        self.episode_step_number = int(0)  # self.elapsed_timestep
        self.episode_time = 0.  # self.elapsed_time

        # help with episode time counting
        self.start_frame, self.end_frame = None, None
        self.start_elapsed_seconds, self.end_elapsed_seconds = None, None

        # reload world flag
        self.need_to_reload_world = False

        # episode ending flag
        self.collision_stop = collision_stop

        # ============================================================
        # MILE methods

        # for route generation
        self.entrance_point = self.config['ego']['entrance_point']
        self.exit_point = self.config['ego']['exit_point']

        self.entrance_location = carla.Location(x=self.entrance_point[0], y=self.entrance_point[1], z=0.)
        self.exit_location = carla.Location(x=self.exit_point[0], y=self.exit_point[1], z=0.)

        self.entrance_waypoint = self.map.get_waypoint(location=self.entrance_location)
        self.exit_waypoint = self.map.get_waypoint(location=self.exit_location)

        # handlers
        self._om_handler = ObsManagerHandler(obs_configs)
        self._ev_handler = EgoVehicleHandler(self.client, reward_configs, terminal_configs)

        self.task_vehicles = None

        # init the action and obs spaces
        self.init_spaces()

        # # TODO deactivate the traffic flow to debug the mile agent
        # self.traffic_flow_manager.use_traffic_flow(False)

        # todo print additional info, port number, and assign an ID for env
        print('A gym-carla env is initialized.')

    def init_spaces(self):
        """"""
        # observation spaces
        self.observation_space = self._om_handler.observation_space
        # define action spaces exposed to agent
        # throttle, steer, brake
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

    def set_task_option(self, task_option):
        """
        A setter method.

        Set a new task option for env, update related modules immediately.
        """
        # task option and route option
        self.task_option = task_option
        if self.task_option == 'multi_task':
            self.multi_task = True
            self.route_option = 'left'  # current route in str, one of the route_options
        else:
            self.multi_task = False
            self.route_option = self.task_option

        # update modules
        self.update_route_info()

        # todo add arg to determine if clear the traffic
        # reset traffic flow to initial state
        self.clear_traffic_flow()

        self.reset()

    def reset_episode_count(self):
        """
        Reset class attribute elapsed_episode_number to 0.

        :return:
        """
        self.elapsed_episode_number = int(0)

    def set_ego_route(self):
        """
        This method is called in reset.

        Set ego route for agent in reset method, before each episode starts.

        This method must be called before spawning ego vehicle.
        """
        # retrieve route option for current episode
        self.route_option = self.route_seq[self.route_seq_index]
        # update index for next route
        if self.route_seq_index == (len(self.route_seq) - 1):
            self.route_seq_index = 0
        else:
            self.route_seq_index += 1

        # update route info for next route
        self.update_route_info()

        # TODO check if need to remove
        # # update current route option to state manager
        # self.state_manager.set_ego_route(self.ego_route, self.spawn_point, self.end_point)

    def set_route_sequence(self):
        """
        todo add params to set seq length for multi-task training

        This method is supposed to be called in init.

        Generate a sequence of route options for ego vehicle.

        Experiments settings:
         - for multi-task training, we will deploy a sequence consists 10*3 route.
         - for single-task training, we will run the specified route repeatedly
        """
        # multi task condition
        if self.task_option == 'multi_task':
            # todo test different seq length
            #  add arg to set value
            # seq_len = int(10)
            # left_seq = ['left'] * seq_len
            # right_seq = ['right'] * seq_len
            # straight_seq = ['straight'] * seq_len

            seq_len_list = [int(5), int(3), int(2)]
            left_seq = ['left'] * seq_len_list[0]
            right_seq = ['right'] * seq_len_list[1]
            straight_seq = ['straight'] * seq_len_list[2]

            self.route_seq = left_seq + right_seq + straight_seq
        else:  # single task condition
            self.route_seq = [self.route_option]

        # reset route sequence index
        self.route_seq_index = int(0)

    def update_route_info(self):
        """
        Update ego route of current episode.
        """
        # retrieve route from class attribute route info
        self.ego_route = self.route_info[self.route_option]['ego_route']
        self.spawn_point = self.route_info[self.route_option]['spawn_point']
        self.end_point = self.route_info[self.route_option]['end_point']
        self.route_length = self.route_info[self.route_option]['route_length']

    @staticmethod
    def get_waypoint_list(routes):

        waypoints = []
        for wp, _ in routes:
            x = wp.location.x
            y = wp.location.y
            waypoints.append((x, y))

        return waypoints

    def spawn_ego_vehicle(self):
        """
        Spawn ego vehicle.
        """
        # ego vehicle blueprint
        bp = random.choice(self.world.get_blueprint_library().filter('vehicle.lincoln.mkz2017'))
        # set color
        if bp.has_attribute('color'):
            color = '255, 0, 0'  # use string to identify a RGB color
            bp.set_attribute('color', color)

        # attributes of a blueprint
        # print('-'*10, 'blueprint attributes', '-'*10)
        # print(bp)
        # for attr in bp:
        #     print('  - {}'.format(attr))

        # set role name
        # hero is the very specified name to activate physic mode
        bp.set_attribute('role_name', 'hero')

        # attributes of a blueprint is stored as dict
        # ego_attri = self.ego_vehicle.attributes

        # set sticky control
        """
        carla Doc:
        when “sticky_control” is “False”, 
        the control will be reset every frame to 
        its default (throttle=0, steer=0, brake=0) values.
        """
        bp.set_attribute('sticky_control', 'True')

        if self.ego_route:
            start_trans = self.spawn_point.transform
            # spawn transform need to be checked, z value must larger than 0
            start_trans.location.z += 0.1
            try:
                self.ego_vehicle = self.world.spawn_actor(bp, start_trans)

                # time.sleep(0.1)
                self.try_tick_carla()

                # update ego vehicle id
                self.ego_id = self.ego_vehicle.id

                # print('Ego vehicle is spawned.')
            except:
                raise Exception("Fail to spawn ego vehicle!")
        else:
            raise Exception("Ego route is not assigned!")

        # using the fixed API to create the sensors
        self.ego_sensors = Sensors(self.world, self.ego_vehicle)

        self.try_tick_carla()

        print('Ego vehicle is ready.')

    def get_min_distance(self):
        """
        Get a minimum distance for waypoint buffer
        """
        if self.ego_vehicle:

            # fixme use current ego speed
            # speed = get_speed(self.ego_vehicle)
            #
            # target_speed = 4.0  # m/s
            # ref_speed = max(speed, target_speed)

            # reference speed, in m/s
            ref_speed = 3

            # min distance threthold of waypoint reaching
            MIN_DISTANCE_PERCENTAGE = 0.75
            sampling_radius = ref_speed * 1.  # maximum distance vehicle move in 1 seconds
            min_distance = sampling_radius * MIN_DISTANCE_PERCENTAGE

            return min_distance
        else:
            raise

    def clear_traffic_flow(self):
        """
        Clear all NPC vehicles and their collision sensors through traffic flow manager.
        """
        if self.traffic_flow_manager:
            self.traffic_flow_manager.clean_up()
            print('Traffic flow is cleared.')

    def init_traffic_flow(self):
        """
        Reset traffic flow and check if traffic flow is ready for start a new episode.
        """
        print('Start to initialize the traffic flow...')

        # # clear traffic periodically
        # if (self.elapsed_episode_number + 1) % self.traffic_clear_period == 0:
        #     self.clear_traffic_flow()

        # todo fix reload carla world, problem is that all modules need to be reset.
        # if (self.elapsed_episode_number + 1) % self.reload_world_period == 0:
        #     pass

        # minimum npc vehicle number to start
        # # different settings for multi-task
        # min_npc_number = 5 if self.multi_task else 5

        # same npc number
        min_npc_number = 5
        # # TODO debug the mile method
        # min_npc_number = 0

        # =====  traffic lights state  =====
        # get target traffic lights phase according to route_option
        # phase = [0, 2] refers to x and y direction Green phase respectively
        if self.route_option in ['straight', 'left']:
            # target_phase = [2, 3]  # the phase index is same as traffic light module
            target_phase = [0, 1]
        else:  # right
            target_phase = [0, 1, 2, 3]  # [0, 1, 2, 3]
            # todo in training phase, make it more difficult
            # if self.training:
            #     target_phase = [0, 1, 2, 3]
            # else:
            #     target_phase = [0, 1]

        # todo add condition check for training and evaluation
        # if self.train_phase:
        #     conditions = [vehNum_cond, tls_cond, remain_time_cond]
        # else:
        #     # conditions = [vehNum_cond, tls_cond]  # huawei setting for evaluation
        #     conditions = [vehNum_cond, tls_cond, remain_time_cond]

        # ==============   init conditions   ==============
        # conditions for init traffic flow finishes
        # vehicle number condition
        self.update_vehicles()
        vehicle_numbers = len(self.npc_vehicles)
        vehNum_cond = vehicle_numbers >= min_npc_number

        # traffic light state condition
        # todo in this version, the traffic flow lights switch is deprecated
        # if self.use_tls_control:
        #     current_phase_index = self.tls_manager.get_phase_index()  # index, int
        #     # if current tls phase coordinates with target phase
        #     tls_cond = current_phase_index in target_phase
        # else:
        #     tls_cond = True

        # current_phase_index = self.traffic_light_manager.get_phase_index()  # index, int
        # # if current tls phase coordinates with target phase
        # tls_cond = current_phase_index in target_phase

        # use the fixed tls condition check
        tls_cond = self.traffic_light_manager.get_phase_condition()

        # TODO must coordinate to traffic light phase
        # remain time condition
        min_remain_time = 10.

        # debug remain_time methods
        # current_phase refers to the phased id 0, 1, 2, 3
        remain_time, current_phase = self.traffic_light_manager.get_remain_time()
        remain_time_cond = True if remain_time >= min_remain_time else False

        # ============================ conditions list ============================
        # must satisfy all of them to start an episode
        # # og, without remain time
        # conditions = [vehNum_cond, tls_cond]

        # new impl
        conditions = [
            vehNum_cond,
            tls_cond,
            remain_time_cond,
        ]

        while not all(conditions):
            # tick the simulator before check conditions
            self.tick_simulation()

            # ==============  vehicle number condition  ==============
            # self.update_vehicles()
            vehicle_numbers = len(self.npc_vehicles)
            vehNum_cond = vehicle_numbers >= min_npc_number

            # if self.debug:
            #     if vehicle_numbers >= min_npc_number:
            #         print('npc vehicle number: ', vehicle_numbers)

            # ==============  traffic light state condition  ==============
            # # get tls phase from tls module
            # current_phase_index = self.traffic_light_manager.get_phase_index()  # index, int
            # # if current tls phase coordinates with target phase
            # tls_cond = current_phase_index in target_phase

            # use the fixed tls condition check
            tls_cond = self.traffic_light_manager.get_phase_condition()

            # # debug tls cond
            # if not tls_cond:
            #     print('')

            # if self.debug:
            #     if tls_cond:
            #         print('')

            remain_time, current_phase = self.traffic_light_manager.get_remain_time()
            if remain_time >= min_remain_time:
                remain_time_cond = True
            else:
                remain_time_cond = False

            # todo add condition on the remain time of current tls phase
            # # remain time is considered in training phase
            # if tls_cond and vehNum_cond:
            #     remain_time = traci.trafficlight.getNextSwitch('99810003') - traci.simulation.getTime()
            #     remain_time_cond = remain_time >= 36.5
            #
            #     remain_time_cond = remain_time >= 36.9  # original
            #     remain_time_cond = remain_time <= 0.1  # debug waiting before junction
            #
            #     if using interval, interval module is required
            #     zoom = Interval(25., 37.)
            #     remain_time_cond = remain_time in zoom

            # ==============  append all conditions  ==============
            # # og impl
            # conditions = [vehNum_cond, tls_cond]

            # new impl with tls cond
            conditions = [
                vehNum_cond,
                tls_cond,
                remain_time_cond,
            ]

        print('The traffic flow is ready.')

    def parameters_decay(self):
        """
        todo fix and test this method
        todo use a config file to set the parameters and their range
        todo add args and APIs to tune params' range of traffic flow

        :return:
        """

        # # ==========   set collision probability   ==========
        # """
        # TrafficFlowManager5 and TrafficFlowManager5Fixed has such API.
        #
        # TrafficFlowManager5Fixed is a developing version.
        #
        # Instruction:
        # There are several mode of traffic flow settings:
        #
        #  - fixed traffic flow param
        #  - whether use stochastic process to generate traffic params
        #
        # """
        # if self.traffic_flow.__class__.__name__ in ['TrafficFlowManager5']:
        #     if self.collision_prob_decay:
        #         collision_prob_range = [0.75, 0.95]
        #         # todo add args to set decay range
        #         collision_decay_length = int(1000)  # episode number for collision prob increasing
        #         # default episode number is 2000
        #         collision_prob = collision_prob_range[0] + \
        #                          self.episode_number * (
        #                                      collision_prob_range[1] - collision_prob_range[0]) / collision_decay_length
        #         collision_prob = np.clip(collision_prob, collision_prob_range[0], collision_prob_range[1])
        #         self.traffic_flow.set_collision_probability(collision_prob)
        #
        #     # traffic flow params decay
        #     if self.tf_params_decay:
        #         # this setting is available when param noise is enabled
        #         if self.tf_randomize:
        #             # decay length, episode number, 5000 is current minimum training length
        #             # tf_param_decay_length = int(5000)
        #
        #             # for debug
        #             tf_param_decay_length = int(10)
        #
        #             # todo store params range through external data storage
        #             #  or retrieve traffic flow params form the class
        #             # for now the target params are set manually here
        #             final_speed_range = (25, 40)
        #             final_distance_range = (10, 25)
        #
        #             # retrieve original traffic flow params
        #             for tf in self.traffic_flow.active_tf_directions:
        #                 info_dict = self.traffic_flow.traffic_flow_info[tf]
        #                 # original range
        #                 speed_range = info_dict['target_speed_range']
        #                 distance_range = info_dict['distance_range']
        #
        #                 # new range
        #                 current_speed_range = (
        #                     self.linear_mapping(speed_range[0], final_speed_range[0], tf_param_decay_length,
        #                                         self.episode_number),
        #                     self.linear_mapping(speed_range[1], final_speed_range[1], tf_param_decay_length,
        #                                         self.episode_number),
        #                 )
        #                 current_distance_range = (
        #                     self.linear_mapping(distance_range[0], final_distance_range[0], tf_param_decay_length,
        #                                         self.episode_number),
        #                     self.linear_mapping(distance_range[1], final_distance_range[1], tf_param_decay_length,
        #                                         self.episode_number),
        #                 )
        #
        #                 self.traffic_flow.traffic_flow_info[tf]['target_speed_range'] = current_speed_range
        #                 self.traffic_flow.traffic_flow_info[tf]['distance_range'] = current_distance_range

        # use different traffic flow seed
        if self.elapsed_episode_number % 2 == 0:
            self.tm_seed += 1
            if self.tm_seed > 10:
                self.tm_seed = 0

            # this only works for the env4 and traffic flow multi-task2
            self.traffic_flow_manager.set_random_seed(self.tm_seed)

        # todo add setter methods to set the params
        if self.train:
            collision_prob_range = [0.75, 0.95]
            # collision_prob_range = [0.1, 0.15]
            # episode number for collision prob increasing
            collision_decay_length = int(2_000)

            # calculate the param by linear decay
            if self.elapsed_episode_number <= collision_decay_length:
                collision_prob = \
                    collision_prob_range[0] + \
                    (self.elapsed_episode_number - 1) * (
                                collision_prob_range[1] - collision_prob_range[0]) / collision_decay_length
                collision_prob = np.clip(collision_prob, collision_prob_range[0], collision_prob_range[1])

                self.traffic_flow_manager.set_collision_detection_rate(collision_prob)
            else:
                self.traffic_flow_manager.set_collision_detection_rate(collision_prob_range[-1])

            # todo add the traffic flow speed and gap distance/time decay

        else:  # test mode
            collision_prob = self.collision_prob
            self.traffic_flow_manager.set_collision_detection_rate(collision_prob)

    def reset(self):
        """
        Reset ego vehicle.
        :return:
        """

        if self.reload_freq > 0:
            if (self.elapsed_episode_number + 1) % self.reload_freq == 0:
                print('{} episodes are elapsed, the carla world will be reloaded.'.format(self.elapsed_episode_number))
                self.reload_carla_world()

        # check and destroy ego vehicle and its sensors
        if self.ego_vehicle:

            # mile
            self._om_handler.clean()

            self.destroy()
            # make destroy ego vehicle take effect
            self.tick_simulation()

        # clear buffered waypoints
        self._waypoints_queue.clear()
        self._waypoint_buffer.clear()

        # reset ego route
        self.set_ego_route()

        # prepare the traffic flow
        self.init_traffic_flow()

        # spawn ego only after traffic flow is ready
        self.spawn_ego_vehicle()

        # create the TaskVehicle for mile state manager
        self.task_vehicles = TaskVehicle(
            vehicle=self.ego_vehicle,
            target_transforms=[
                self.entrance_waypoint.transform,
                self.exit_waypoint.transform,
                self.end_waypoint.transform,
            ],
            spawn_transforms=self.spawn_waypoint.transform,
            endless=True,
        )

        # set initial speed
        if self.initial_speed:
            self.set_velocity(self.ego_vehicle, self.initial_speed)

        # todo move to spawn vehicle
        # buffer waypoints from ego route
        for elem in self.ego_route:
            self._waypoints_queue.append(elem)

        # # reset the state manager with ego vehicle and sensors
        # self.state_manager.reset(self.ego_vehicle, self.ego_sensors)

        task_vehicles = {'hero': self.task_vehicles}
        self._om_handler.reset(task_vehicles)

        self.try_tick_carla()

        snap_shot = self.world.get_snapshot()
        self.timestamp = {
            'step': 0,
            'frame': snap_shot.timestamp.frame,
            'relative_wall_time': 0.0,
            'wall_time': snap_shot.timestamp.platform_timestamp,
            'relative_simulation_time': 0.0,
            'simulation_time': snap_shot.timestamp.elapsed_seconds,
            'start_frame': snap_shot.timestamp.frame,
            'start_wall_time': snap_shot.timestamp.platform_timestamp,
            'start_simulation_time': snap_shot.timestamp.elapsed_seconds
        }

        # get observations
        obs_dict = self._om_handler.get_observation(self.timestamp)

        # TODO check this
        state = obs_dict

        # update the step location when the ego vehicle is re-spawned
        self.update_ego_info()
        self.ego_step_location = self.ego_location

        # ==========  reset episodic counters  ==========
        # accumulative reward of single episode
        self.episode_reward = 0

        # reset step number at the end of the reset method
        self.elapsed_timestep = 0
        self.elapsed_episode_number += 1  # update episode number

        # this API is still working for some modules
        self.episode_step_number = self.elapsed_timestep

        # update start time
        start_frame, start_elapsed_seconds = self.get_carla_time()
        self.start_frame, self.start_elapsed_seconds = start_frame, start_elapsed_seconds

        # todo add callback API for methods called at the beginning/ending of the episode/timestep
        # update params decay of modules
        self.parameters_decay()

        print('CARLA Env is reset.')

        return state

    def update_ego_info(self):
        """
        Update ego motion info.
        """
        # get transform and location
        self.ego_transform = self.ego_vehicle.get_transform()
        self.ego_location = self.ego_transform.location

    def buffer_waypoint(self):
        """
        Buffering waypoints for planner.

        This method should be called each timestep.

        _waypoint_buffer: buffer some wapoints for local trajectory planning
        _waypoints_queue: all waypoints of current route

        :return: 2 nearest waypoint from route list
        """

        # add waypoints into buffer
        least_buffer_num = 10
        if len(self._waypoint_buffer) <= least_buffer_num:
            for i in range(self._buffer_size - len(self._waypoint_buffer)):
                if self._waypoints_queue:
                    self._waypoint_buffer.append(
                        self._waypoints_queue.popleft())
                else:  # when there is not enough waypoint in the waypoint queue
                    break

        # purge the queue of obsolete waypoints
        max_index = -1
        for i, tuple in enumerate(self._waypoint_buffer):

            transform = tuple[0]

            # todo check if z coord value effect the dist calculation
            _dist = self.ego_location.distance(transform.location)
            _min_dist = self.get_min_distance()

            # if no.i waypoint is in the radius
            if _dist < _min_dist:
                max_index = i

        if max_index >= 0:
            for i in range(max_index + 1):  # (max_index+1) waypoints to pop out of buffer
                self._waypoint_buffer.popleft()

    def get_obs(self):
        """"""

        # update timestamp
        snap_shot = self.world.get_snapshot()

        self.timestamp['step'] = snap_shot.timestamp.frame-self.timestamp['start_frame']
        self.timestamp['frame'] = snap_shot.timestamp.frame
        self.timestamp['wall_time'] = snap_shot.timestamp.platform_timestamp
        self.timestamp['relative_wall_time'] = self.timestamp['wall_time'] - self.timestamp['start_wall_time']
        self.timestamp['simulation_time'] = snap_shot.timestamp.elapsed_seconds
        self.timestamp['relative_simulation_time'] = self.timestamp['simulation_time'] \
            - self.timestamp['start_simulation_time']

        # info_criteria = self.task_vehicles.tick(self.timestamp)

        # get observations
        obs_dict = self._om_handler.get_observation(self.timestamp)

        return obs_dict

    def tick_controller(self, action):
        """
        Tick controller of ego vehicle.

        Directly give the vehicle control instance.
        """

        # planning waypoints are deprecated
        self.buffer_waypoint()

        if self._waypoint_buffer:
            veh_control = action
        else:
            veh_control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)

        self.ego_vehicle.apply_control(veh_control)

    def step(self, action: tuple):
        """
        Run a step for RL training.

        This method will:

         - tick simulation multiple times according to frame_skipping_factor
         - tick carla world and all carla modules
         - apply control on ego vehicle
         - compute rewards and check if episode ends
         - update information

        :param action:
        :return:
        """

        # update ego location before this RL timestep
        self.update_ego_info()
        self.ego_step_location = self.ego_location

        # ================   Tick simulation   ================
        # todo ref carla co-sim method with time.time() to sync with real world time
        # execute multiple timestep according to frame_skipping_factor
        for i in range(self.frame_skipping_factor):

            # frame, elapsed_seconds = self.get_frame()
            # print('-----'*5)
            # print('before update_ego_info()')
            # print('frame: ', frame)
            # print('elapsed_seconds: ', elapsed_seconds)
            # print('-----'*5)

            # update ego transform
            self.update_ego_info()

            # frame, elapsed_seconds = self.get_frame()
            # print('-----'*5)
            # print('after update_ego_info()')
            # print('frame: ', frame)
            # print('elapsed_seconds: ', elapsed_seconds)
            # print('-----'*5)

            # tick ego vehicle controller
            self.tick_controller(action)

            # frame, elapsed_seconds = self.get_frame()
            # print('-----'*5)
            # print('after tick_controller(action)')
            # print('frame: ', frame)
            # print('elapsed_seconds: ', elapsed_seconds)
            # print('-----'*5)

            # ================ tick the all functional modules ================
            # # original lines
            # # other modules
            # self.tick_simulation()

            # ----------------  NOTICE: merge collision check into tick_simulation method  ----------------

            # spawn new NPC vehicles
            self.traffic_flow_manager.run_step_1()

            # frame, elapsed_seconds = self.get_frame()
            # print('-----'*5)
            # print('after traffic_flow_manager.run_step_1()')
            # print('frame: ', frame)
            # print('elapsed_seconds: ', elapsed_seconds)
            # print('-----'*5)

            # tick carla world
            self.try_tick_carla()

            # frame, elapsed_seconds = self.get_frame()
            # print('-----'*5)
            # print('carla tick')
            # print('after try_tick_carla()')
            # print('frame: ', frame)
            # print('elapsed_seconds: ', elapsed_seconds)
            # print('-----'*5)

            # =====  get state from state manager module  =====
            state = self.get_obs()

            # frame, elapsed_seconds = self.get_frame()
            # print('-----'*5)
            # print('after get_obs()')
            # print('frame: ', frame)
            # print('elapsed_seconds: ', elapsed_seconds)
            # print('-----'*5)

            # TODO rewrite this part
            # get reward and done flag of current step
            # episodic done check is within this method
            reward, done, info = self.compute_reward()
            aux = {'exp_state': info}

            # if done:
            #     print('')

            # frame, elapsed_seconds = self.get_frame()
            # print('-----'*5)
            # print('after compute_reward()')
            # print('frame: ', frame)
            # print('elapsed_seconds: ', elapsed_seconds)
            # print('-----'*5)

            # register vehicles to traffic manager
            # delete collision vehicles
            self.traffic_flow_manager.run_step_2()

            # frame, elapsed_seconds = self.get_frame()
            # print('-----'*5)
            # print('after traffic_flow_manager.run_step_2()')
            # print('frame: ', frame)
            # print('elapsed_seconds: ', elapsed_seconds)
            # print('-----'*5)

            # traffic light module
            self.traffic_light_manager.run_step()

            # frame, elapsed_seconds = self.get_frame()
            # print('-----'*5)
            # print('after traffic_light_manager.run_step()')
            # print('frame: ', frame)
            # print('elapsed_seconds: ', elapsed_seconds)
            # print('-----'*5)

            # update active npc vehicles
            self.update_vehicles()

            # frame, elapsed_seconds = self.get_frame()
            # print('frame: ', frame)
            # print('elapsed_seconds: ', elapsed_seconds)

            # end the frame skipping if violation happens
            if done:
                break

        # update the step number
        self.elapsed_timestep += 1
        self.episode_step_number = self.elapsed_timestep

        # update episodic reward
        self.episode_reward += reward

        if done:
            # result in str
            episode_result = info
            print(
                '\n',
                '=' * 10, 'env internal counts', '=' * 10, '\n',
                'episode result: {}'.format(episode_result), '\n',
                'episode number: {}'.format(self.elapsed_episode_number), '\n',
                'episode step number: {}'.format(self.elapsed_timestep), '\n',
                'episode reward: {:.2f}'.format(self.episode_reward), '\n',
                "Current time: {0:%Y-%m-%d-Time%H-%M-%S}".format(datetime.now()), '\n',
                '=' * 41, '\n',
            )

            # elapsed episode frame number
            end_frame, end_elapsed_seconds = self.get_carla_time()
            self.end_frame, self.end_elapsed_seconds = end_frame, end_elapsed_seconds
            episode_frame = self.end_frame - self.start_frame

            # duration time of this episode
            episode_elapsed_seconds = self.end_elapsed_seconds - self.start_elapsed_seconds
            self.episode_time = episode_elapsed_seconds

        return state, reward, done, aux

    def reload_carla_world(self):
        """
        Reload the entire carla world and init all modules.
        """

        # delete ego vehicle and collision sensor
        self.destroy()

        # must delete all actors manually before reload
        self.traffic_flow_manager.delete_vehicles()

        self.carla_env.reload_carla_world()

        # get carla API
        self.carla_api = self.carla_env.get_env_api()
        self.client = self.carla_api['client']
        self.world = self.carla_api['world']
        self.map = self.carla_api['map']
        self.debug_helper = self.carla_api['debug_helper']  # carla.Debughelper is for visualization
        self.blueprint_library = self.carla_api['blueprint_library']
        self.spectator = self.carla_api['spectator']
        self.traffic_manager = self.carla_api['traffic_manager']

        # set the spectator on the top of junction
        # self.carla_env.set_spectator_overhead(junction_center, yaw=270, h=70)

        # TODO merge to env additional params
        self.carla_env.set_spectator_overhead(self.junction_center, yaw=270, h=50.)

        # # todo merge waypoints buffer into local planner
        # # ==================================================
        # # ----------   waypoint buffer begin  ----------
        # # a queue to store original route
        # self._waypoints_queue = deque(maxlen=100000)  # maximum waypoints to store in current route
        # # buffer waypoints from queue, get nearby waypoint
        # self._buffer_size = 50
        # self._waypoint_buffer = deque(maxlen=self._buffer_size)
        #
        # # todo near waypoints is waypoint popped out from buffer
        # # self.near_waypoint_queue = deque(maxlen=50)
        # # ----------   waypoint buffer end  ----------
        # # ==================================================

        # ================   carla modules  ================
        #

        # # todo develop method to retrieve junction from general map
        # self.junction = get_junction_by_location(
        #     carla_map=self.map,
        #     location=junction_center,
        # )
        #
        # # ================   route manager   ================
        # self.route_manager = self.route_manager_cls(
        #     carla_api=self.carla_api,
        #     junction=self.junction,
        #     route_distance=(3., 3.),
        # )
        #
        # # generate all routes for each route option
        # # route format: <list>.<tuple>.(transform, RoadOption)
        # for route in ['left', 'right', 'straight', 'straight_0']:
        #     ego_route, spawn_point, end_point = self.route_manager.get_route(route_option=route)
        #
        #     self.route_info[route]['ego_route'] = ego_route
        #     self.route_info[route]['spawn_point'] = spawn_point
        #     self.route_info[route]['end_point'] = end_point
        #
        # self.ego_route = []  # list of route waypoints
        # self.spawn_point = None
        # self.end_point = None
        #
        # # attributes for ego routes management
        # self.route_seq = []
        # self.route_seq_index = None
        # # set a route sequence for different task setting
        # self.set_route_sequence()

        # ================   traffic flow manager   ================
        # todo use 2 types of traffic flow, CARLA Autopilot and AEB
        self.traffic_flow_manager = self.traffic_flow_manager_cls(
            carla_api=self.carla_api,
            junction=self.junction,
            scenario_config=self.config,

            tm_port=self.tm_port,
            tm_seed=self.tm_seed,

            # tls_red_phase_duration=20.,  # duration time of red phase of traffic lights
            phase_time=self.traffic_lights_manager_cls.phase_time,

            # debug=False,
            debug=DEBUG_TRAFFICFLOW_MANAGER,
            verbose=False,
        )

        # todo improve method for init task and route

        # update route info of ego vehicle
        self.update_route_info()

        # todo add params decay through training procedure
        #  - use some open-source codes?
        # collision detect decay
        param = {
            'initial_value': 0.5,
            'target_value': 1.,
            'episode_number': int(2000),
            'scheduler': 'linear',
        }

        # todo fix the API of traffic flow params decay
        # self.collision_prob_decay = collision_prob_decay
        # if not self.collision_prob_decay:
        #     collision_prob = 1.  # todo add arg to set this value
        #     self.traffic_flow.set_collision_probability(collision_prob)
        #
        # self.tf_params_decay = tf_params_decay

        # ================   traffic_light_manager   ================
        # todo add API to set duration time of traffic lights
        self.traffic_light_manager = self.traffic_lights_manager_cls(
            carla_api=self.carla_api,
            junction=self.junction,
            # todo check this arg and training setting

            scenario_config=self.config,

            use_tls_control=True,
            verbose=False,
        )

        # ================   state manager   ================
        # # todo fix multi-task setting in lane graph RL methods
        # multi_task = True if self.route_option == 'multi_task' else False

        # # counters need be reset
        # self.reset_episode_count()

        # set flag
        self.need_to_reload_world = False

    def try_tick_carla(self):
        """
        todo fix this method as a static method for all carla modules

        Tick the carla world with try-exception method.

        In fixing fail to tick the world in carla
        """

        max_try = int(20)
        tick_success = False
        tick_counter = int(0)
        while not tick_success:
            # # debug condition
            # if self.elapsed_episode_number == 2 and self.elapsed_timestep == 5:

            # critical condition
            if tick_counter >= max_try:
                print('The CARLA world is stuck, will be reloaded immediately...')

                # reload the entire carla world
                self.reload_carla_world()  # default is not init the world settings
                print('CARLA world is reloaded successfully.')

                self.reset()

                break

                # # raise the error
                # raise RuntimeError('Fail to tick carla for ', max_try, ' times...')
            try:
                # if this step success, a frame id will return
                frame_id = self.world.tick(10.)
                if frame_id:
                    self.frame_id = frame_id
                    tick_success = True
            except:  # for whatever the error is...
                print('*-' * 20)
                print('Fail to tick the world for once...')
                print('Last frame id is ', self.frame_id)
                print('*-' * 20)
                tick_counter += 1

        if tick_counter > 0:
            print('carla client is successfully ticked after ', tick_counter, 'times')

            print('CARLA world will be reloaded after this episode.')
            self.need_to_reload_world = True

    # # ===========================================================================
    # # original impl
    # def try_tick_carla(self):
    #     """
    #     todo fix this method as a static method for all carla modules
    #
    #     Tick the carla world with try exception method.
    #
    #     In fixing fail to tick the world in carla
    #     """
    #
    #     max_try = int(100)
    #     tick_success = False
    #     tick_counter = int(0)
    #     while not tick_success:
    #         if tick_counter >= max_try:
    #             raise RuntimeError('Fail to tick carla for ', max_try, ' times...')
    #         try:
    #             # if this step success, a frame id will return
    #             frame_id = self.world.tick(20.)
    #             if frame_id:
    #                 self.frame_id = frame_id
    #                 tick_success = True
    #         except:  # for whatever the error is..
    #             print('*-' * 20)
    #             print('Fail to tick the world for once...')
    #             print('Last frame id is ', self.frame_id)
    #             print('*-' * 20)
    #             tick_counter += 1
    #
    #     if tick_counter > 0:
    #         print('carla client is successfully ticked after ', tick_counter, 'times')

    def get_frame(self):
        """
        Get frame id from world.

        :return: frame_id
        """
        snapshot = self.world.get_snapshot()
        frame = snapshot.timestamp.frame
        elapsed_seconds = snapshot.timestamp.elapsed_seconds

        # print('frame: ', frame)
        # print('elapsed_seconds: ', elapsed_seconds)

        return frame, elapsed_seconds

    def tick_simulation(self):
        """
        todo we don't use this method self.step() method, since the error in NPC collision check.
        todo manage modules with a manager, and tick all registered modules using loop

        Tick the simulation.

        The following modules will be ticked.

        :return:
        """

        # ===================  co-simulation code  ===================
        # for _ in range(int(self.control_dt/self.server_dt)):
        #     # world tick
        #     start = time.time()
        #     self.world.tick()
        #     end = time.time()
        #     elapsed = end - start
        #     if elapsed < self.server_dt:
        #         time.sleep(self.server_dt - elapsed)

        # ===================   render local map in pygame   ===================
        #
        # vehicle_poly_dict = self.localmap.get_actor_polygons(filter='vehicle.*')
        # self.all_polygons.append(vehicle_poly_dict)
        # while len(self.all_polygons) > 2: # because two types(vehicle & walker) of polygons are needed
        #     self.all_polygons.pop(0)
        # # pygame render
        # self.localmap.display_localmap(self.all_polygons)

        # frame, elapsed_seconds = self.get_frame()
        # print('-----'*5)
        # print('-----')
        # print('before traffic_flow_manager.run_step_1()')
        # print('frame: ', frame)
        # print('elapsed_seconds: ', elapsed_seconds)

        # # fix traffic flow manager run_step_1
        # try:
        #     self.traffic_flow_manager.run_step_1()
        # except:
        #     raise RuntimeError('run_step_1')

        self.traffic_flow_manager.run_step_1()

        # frame, elapsed_seconds = self.get_frame()
        # print('-----')
        # print('after traffic_flow_manager.run_step_1()')
        # print('frame: ', frame)
        # print('elapsed_seconds: ', elapsed_seconds)

        # tick carla server
        self.try_tick_carla()

        # frame, elapsed_seconds = self.get_frame()
        # print('-----')
        # print('carla tick')
        # print('before traffic_flow_manager.run_step_2()')
        # print('frame: ', frame)
        # print('elapsed_seconds: ', elapsed_seconds)

        # try:
        #     self.traffic_flow_manager.run_step_2()
        # except:
        #     raise RuntimeError('run_step_2')

        self.traffic_flow_manager.run_step_2()

        # frame, elapsed_seconds = self.get_frame()
        # print('-----')
        # print('after traffic_flow_manager.run_step_2()')
        # print('frame: ', frame)
        # print('elapsed_seconds: ', elapsed_seconds)
        # print('-----'*5)

        # todo register all carla modules, add a standard api
        # ================   tick carla modules   ================

        # # original line
        # # tick the traffic flow module
        # self.traffic_flow_manager.run_step()

        # frame, elapsed_seconds = self.get_frame()
        # print('frame: ', frame)
        # print('elapsed_seconds: ', elapsed_seconds)

        # traffic light module
        self.traffic_light_manager.run_step()

        # frame, elapsed_seconds = self.get_frame()
        # print('frame: ', frame)
        # print('elapsed_seconds: ', elapsed_seconds)

        # update active npc vehicles
        self.update_vehicles()

        # frame, elapsed_seconds = self.get_frame()
        # print('frame: ', frame)
        # print('elapsed_seconds: ', elapsed_seconds)

    def update_vehicles(self):
        """
        Modified from BasicEnv.

        Only update NPC vehicles.
        """
        self.npc_vehicles = []

        # update vehicle list
        actor_list = self.world.get_actors()
        vehicle_list = actor_list.filter('vehicle.*')  # as a ActorList instance, iterable

        if vehicle_list:
            for veh in vehicle_list:
                attr = veh.attributes  # dict
                # filter ego vehicle by role name
                if attr['role_name'] == 'ego' or attr['role_name'] == 'hero':
                    continue
                else:
                    self.npc_vehicles.append(veh)

    def set_reward_dict(self, reward_dict):
        """
        Set the reward dict through an external API.

        original reference:

        reward_dict = {
            'collision': -350.,
            'time_exceed': -100.,
            'success': 150.,
            'step': -0.3,
        }
        """
        self.reward_dict = reward_dict
        print('Reward setting is reset, current reward dict is: \n{}'.format(self.reward_dict))

    def compute_reward(self):
        """
        Compute reward value of current time step.
        Notice that the time step is specified for RL process.

        :return:
        """
        # todo check reward calculation in multi-task mode
        reward = 0.

        # # original training reward settings
        # reward_dict = {
        #     'collision': -350.,
        #     'time_exceed': -100.,
        #     'success': 150.,
        #     'step': -0.3,
        # }

        # use the class attribute to set the reward dict
        reward_dict = self.reward_dict

        # todo fix aux info for stable baselines
        # collision = False
        # time_exceed = False
        aux_info = 'running'

        done = False

        # update collision condition from collision sensor attribute collision flag

        # # original
        # self.collision_flag = self.ego_collision_sensor.collision_flag

        # using the fixed API
        self.collision_flag = self.ego_sensors.collision_flag

        # check if collision happens by collision sensor of ego vehicle
        collision = self.collision_flag
        # check time exceed
        time_exceed = self.elapsed_timestep > self.max_episode_timestep
        # check if success
        # check if ego reach goal
        # check distance between ego vehicle and end_point
        dist_threshold = 5.0
        ego_loc = self.ego_vehicle.get_location()
        end_loc = self.end_point.transform.location
        dist_ego2end = ego_loc.distance(end_loc)
        # success indicator
        success = True if dist_ego2end <= dist_threshold else False

        episode_result = None
        # compute reward value based on above
        if collision:
            reward += reward_dict['collision']
            if self.collision_stop:
                done = True
            else:
                # # original line
                # self.ego_collision_sensor.reset_sensors()

                # using the fixed API
                self.ego_sensors.reset_sensors()

                done = False
            aux_info = 'collision'
            # print('Failure: collision!')
            episode_result = 'Failure. Collision!'
        elif time_exceed:
            reward += reward_dict['time_exceed']
            done = True
            aux_info = 'time_exceed'
            # print('Failure: Time exceed!')
            episode_result = 'Failure. Time exceed!'
        elif success:
            done = True
            aux_info = 'success'
            reward += reward_dict['success']
            # print('Success: Ego vehicle reached goal.')
            episode_result = 'Success!'
        else:  # still running
            done = False
            aux_info = 'running'
            # calculate step reward according to elapsed time step
            if self.elapsed_timestep >= 0.5 * self.max_episode_timestep:
                reward += 2 * reward_dict['step']
            else:
                reward += reward_dict['step']

            # if using the step proceeding reward
            if self.use_proceed_reward:
                proceed_dist = self.ego_location.distance(self.ego_step_location)
                proceed_reward = proceed_dist / self.route_length * reward_dict['success']

                reward += proceed_reward

        return reward, done, aux_info

    def destroy(self):
        """
        Destroy ego vehicle and its collision sensors

        In this method we use the client command to delete sensors
        """
        delete_list = []
        # delete ego vehicle actor
        if self.ego_vehicle:
            delete_list.append(self.ego_vehicle)

        # # retrieve collision sensor from the Sensor API
        # # original impl
        # if self.ego_collision_sensor:
        #     for sensor in self.ego_collision_sensor.sensor_list:
        #         delete_list.append(sensor)

        # fixed API
        if self.ego_sensors:
            for sensor in self.ego_sensors.sensor_list:
                delete_list.append(sensor)

        # todo original version of delete_actors() is from carla_module
        self.traffic_flow_manager.delete_actors(delete_list)

        self.ego_vehicle = None

        # # original impl
        # self.ego_collision_sensor = None

        # fixed version
        self.ego_sensors = None

        # print('Ego vehicle and its sensors are destroyed.')

    def get_carla_time(self):
        """
        Get carla simulation time.
        :return:
        """

        # reset carla
        snapshot = self.world.get_snapshot()
        frame = snapshot.timestamp.frame
        elapsed_seconds = snapshot.timestamp.elapsed_seconds

        return frame, elapsed_seconds

    @staticmethod
    def linear_mapping(original_value: float, target_value: float, total_step: int, current_step: int):
        """
        One-dimensional linear mapping.

        Shrinking of traffic flow params

        :return: current_range
        """
        delta_y = target_value - original_value
        k = delta_y / total_step
        b = original_value

        current_value = k * current_step + b

        return current_value

    @staticmethod
    def set_velocity(vehicle, target_speed: float):
        """
        Set a vehicle to the target velocity.

        params: target_speed: in m/s
        """

        transform = vehicle.get_transform()
        # transform matrix Actor2World
        trans_matrix = get_transform_matrix(transform)

        # target velocity in world coordinate system
        target_vel = np.array([[target_speed], [0.], [0.]])
        target_vel_world = np.dot(trans_matrix, target_vel)
        target_vel_world = np.squeeze(target_vel_world)

        # carla.Vector3D
        target_velocity = carla.Vector3D(
            x=target_vel_world[0],
            y=target_vel_world[1],
            z=target_vel_world[2],
        )
        #
        vehicle.set_target_velocity(target_velocity)

        # tick twice to reach target speed
        # for i in range(2):
        #     self.try_tick_carla()

    def get_carla_world(self):

        return self.world

    def get_ego_vehicle(self):

        return self.ego_vehicle

    def get_episodic_data(self):
        """
        Call this method at the end of an episode, before the reset method is called.

        :return: total timestep and time duration of last episode
        """

        return self.elapsed_timestep, self.elapsed_time

