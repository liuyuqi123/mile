"""
Run env loop with trained mile model.
"""

import os
import sys
import glob

# 1660ti laptop
sys.path.append('/home/liuyuqi/PycharmProjects/gym-carla/')
sys.path.append('/home/liuyuqi/PycharmProjects/mile/')
# 3080ti PC
sys.path.append('/home/lyq/PycharmProjects/gym-carla/')
sys.path.append('/home/lyq/PycharmProjects/mile/')

# ================   Append CARLA Path   ================
from carla_config import version_config

carla_version = version_config['carla_version']
root_path = version_config['root_path']

# ==================================================
# -------------  import carla module  -------------
# ==================================================
carla_root = os.path.join(root_path, 'CARLA_' + carla_version)
carla_path = os.path.join(carla_root, 'PythonAPI')
sys.path.append(carla_path)
sys.path.append(os.path.join(carla_root, 'PythonAPI/carla'))
sys.path.append(os.path.join(carla_root, 'PythonAPI/carla/agents'))

try:
    sys.path.append(glob.glob(carla_path + '/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import numpy as np
import argparse
import logging
import pickle
from datetime import datetime
import subprocess
import hydra
import time
from omegaconf import DictConfig, OmegaConf
import logging
import pygame

log = logging.getLogger(__name__)

from carla_mile_dev.carla_env_mile import CarlaEnvMile

from utils.server_utils import kill_carla
from stable_baselines3.common.vec_env.base_vec_env import tile_images


# def get_root_path(project_path: str):
#     # 获取文件目录
#     curPath = os.path.abspath(os.path.dirname(__file__))
#     # 获取项目根路径，内容为当前项目的名字
#     rootPath = curPath[:curPath.find(project_path)+len(project_path)]
#
#     return  rootPath


def init_carla_env(cfg, **additional_kwargs):
    """
    todo debug using args to init a env instance

    Init the carla env instance with env cls.

    :return: env instance
    """

    # ============================================================
    # scenario config

    from gym_carla.modules.trafficflow.traffic_flow_config import config_default as config
    # from gym_carla.modules.trafficflow.traffic_flow_config import config as config
    # from gym_carla.modules.trafficflow.traffic_flow_config import config3 as config
    # from gym_carla.modules.trafficflow.traffic_flow_config import config6 as config

    # # deprecated
    # from gym_carla.modules.trafficflow.traffic_flow_config import config2 as config
    # from gym_carla.modules.trafficflow.traffic_flow_config import config4 as config
    # from gym_carla.modules.trafficflow.traffic_flow_config import config5 as config

    # T-junctions are not available
    # from gym_carla.modules.trafficflow.traffic_flow_config import config7 as config
    # from gym_carla.modules.trafficflow.traffic_flow_config import config7b as config

    # from gym_carla.modules.trafficflow.traffic_flow_config import config8 as config
    # from gym_carla.modules.trafficflow.traffic_flow_config import config9 as config

    # ============================================================
    # from gym_carla.envs.carla_env_lanegraph4 import CarlaEnvLaneGraph
    from carla_mile_dev.carla_env_mile import CarlaEnvMile

    env = CarlaEnvMile(

        config=config,
        carla_port=cfg['carla_port'],
        tm_port=cfg['tm_port'],
        tm_seed=cfg['tm_seed'],  # seed for autopilot controlled NPC vehicles

        task_option="left",  # all available route options stored in the class attributes

        train=False,  # training mode or evaluation mode
        collision_prob=0.99,

        # TODO check if remove this setting
        attention=False,

        initial_speed=None,  # if set an initial speed to ego vehicle
        state_noise=False,  # if adding noise on state vector

        no_render_mode=False,
        debug=False,
        verbose=False,

        collision_stop=True,
        use_proceed_reward=False,

        # mile API
        obs_configs=additional_kwargs['obs_configs'],
        reward_configs=additional_kwargs['reward_configs'],
        terminal_configs=additional_kwargs['terminal_configs'],
    )

    return env


@hydra.main(config_path='../config', config_name='evaluate_dev')
def test_loop(cfg: DictConfig):
    """
    Run the complete test loop.
    """

    # # open the carla port
    # kill_carla(cfg['carla_port'])
    # cmd = f'bash {cfg["carla_sh_path"]} ' \
    #       f'-fps=20 -quality-level=Epic -carla-rpc-port={cfg["port"]}'
    # log.info(cmd)
    # server_process = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
    # time.sleep(3.)

    print('--' * 10, 'Begin Testing', '--' * 10)

    # init the MILE agent
    from carla_mile_dev.mile_agent_dev import MileAgent

    cfg_agent = cfg.agent['mile']
    cfg_agent['ckpt'] = '/home/lyq/PycharmProjects/mile/mile.ckpt'
    OmegaConf.save(config=cfg_agent, f='config_agent.yaml')
    mile_agent = MileAgent('config_agent.yaml')

    # configs for the env init
    _obs_configs = mile_agent.obs_configs
    obs_configs = {
        'hero': _obs_configs,
    }

    ev_cfg = cfg.actors['hero']
    reward_configs = OmegaConf.to_container(ev_cfg.reward)
    terminal_configs = OmegaConf.to_container(ev_cfg.terminal)

    print('MILE agent initialized.')

    # init a carla env
    env = init_carla_env(
        cfg=cfg,

        # TODO init modules with configs
        obs_configs=obs_configs,
        reward_configs=reward_configs,
        terminal_configs=terminal_configs,
    )

    # # reset internal episode number counter
    # env.reset_episode_count()

    # # ============================================================
    # # render images
    # surface = pygame.display.set_mode(
    #     (1472, 430),
    #     pygame.HWSURFACE | pygame.DOUBLEBUF)

    # traffic manager random seed
    random_seed = 0

    success_count = 0
    episode_total_time = 0.
    for episode in range(1, cfg.total_episodes + 1):

        episode_reward, done = 0, False
        state = env.reset()
        timestamp = env.timestamp

        while True:

            # TODO fix the mile agent API
            action = mile_agent.run_step(
                input_data=state['hero'],
                timestamp=timestamp,
            )

            next_state, reward, done, info = env.step(action)
            timestamp = env.timestamp

            # render_imgs = []
            # reward_debug, terminal_debug = None, None
            #
            # render_imgs.append(mile_agent.render(reward_debug=reward_debug, terminal_debug=terminal_debug))
            #
            # # visualize the stored image
            # _render_image = tile_images(render_imgs)
            #
            # image_surface = pygame.surfarray.make_surface(_render_image.transpose(1, 0, 2))
            # surface.blit(image_surface, (0, 0))
            # pygame.display.flip()

            episode_reward += reward
            if done:

                # reset the tm seed
                if episode % 2 == 0:
                    random_seed += 1
                    if random_seed > 10:
                        random_seed = 0

                    # this only works for the env4 and traffic flow multi-task2
                    env.traffic_flow_manager.set_random_seed(random_seed)

                # save the memory
                # if episode >= noised_episodes:
                #     if episode % rl_config.memory_save_frequency == 0:
                #         memory.save_memory(os.path.join(rl_config.memory_save_path, 'memory.pkl'), memory)

                if info['exp_state'] == 'success':
                    success_count += 1

                # print results in terminal
                episode_duration_time = env.episode_step_number*env.simulator_timestep_length
                episode_total_time += episode_duration_time

                print(
                    '\n',
                    '---' * 20, '\n',
                    'Episode No.{} is finished'.format(episode), '\n',
                    'Episode result is: {}'.format(info['exp_state']), '\n',
                    'Episodic total reward is: {}'.format(episode_reward), '\n',
                    'Episode total steps: {}, total duraion time: {:.2f}'.format(
                        env.episode_step_number,
                        episode_duration_time
                    ), '\n',
                    # "Average reward is {:.2f}".format(np.mean(recent_rewards)), '\n',
                    # "Success rate: {:.1%}".format(avarage_success_rate), '\n',
                    '---' * 20, '\n',
                )

                # end single episode
                break

            else:
                state = next_state

    success_rate = success_count / cfg.total_episodes
    average_duration_time = episode_total_time / cfg.total_episodes

    print('=====' * 20)
    print('Test results')
    print('The success rate is :{}'.format(success_rate))
    print('Success count: {}, total episodes: {}'.format(success_count, cfg.total_episodes))
    print('Average_ time: {}'.format(average_duration_time))
    print('=====' * 20)


def main():

    try:
        test_loop()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
