import gym
import numpy as np
from util import gray_scale, resize, resize_gray
import PIL
from PIL import Image
from config import *
from atari_wrappers import *


class Env():
    def __init__(self, agent, env_id,seed,env_name):
        from nes_py.wrappers import JoypadSpace
        import gym_super_mario_bros
        from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
        env = gym_super_mario_bros.make(env_name)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        # env =  NoopResetEnv(env , noop_max=env_id)
        env = WarpFrame(env, width=IMG_W, height=IMG_H, grayscale=True)
        env = MaxAndSkipEnv(env,skip=4)
        env = FrameStack(env, k=k)  # return (IMG_H , IMG_W ,k)
        self.env = env

        self.agent = agent
        self.env_id = env_id
        self.seed = seed

    def run(self):
        np.random.seed(self.seed)
        self.env.seed(self.seed)

        # use to count episode
        count = 1
        state = self.env.reset()
        one_episode_reward = 0
        # use to count step in one epoch
        step = 0
        done = True
        while True:
            step += 1
            a = self.agent.choice_action(state, done)

            state_, r, done, info = self.env.step(a)

            r /= 60
            one_episode_reward += r

            if done:
                state_ = self.env.reset()

            # This can limit max step
            # if step >= 60000:
            #    done = True

            self.agent.observe(state, a, r, state_, done,info['flag_get'])

            state = state_

            if done:
                print(str(self.env_id) + ":" + str(count) + "      :       " + str(round(one_episode_reward, 3)),
                      info['x_pos'], info['flag_get'])
                count += 1
                one_episode_reward = 0
                state = state_
                step = 0
