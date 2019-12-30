import gym
import numpy as np
from util import gray_scale, resize, resize_gray
import PIL
from PIL import Image
from config import *
from atari_wrappers import *


from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# env =  NoopResetEnv(env , noop_max=env_id)
env = WarpFrame(env, width=IMG_W, height=IMG_H, grayscale=True)
env = MaxAndSkipEnv(env,skip=4)
a = env.reset()
print(1)
