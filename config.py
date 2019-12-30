process_num = 16                             # number of actors
IMG_H = 84                                   # height of state after processing
IMG_W = 84                                   # width of state after processing
batch_size = 64                              # Rollout length , horizon , how much step in one agent in one learn time
recode_span = 50                             # tensorflow record span
save_span = 500*3                           # after (save_span//epoch) update parameter , save parameter , for example if save_span=16000 , epoch=3 , after about 16000/3 will save model parameter
beta = 0.01                                  # Entropy coeff
clip_epsilon = 0.2                           # clip ε , I dont use annealed
lr = 0.001                                 # learning rate
max_learning_times = int(1e7/process_num/batch_size)          # max learning time  9765
gamma = 0.99                                 # discount reward
learning_batch = process_num*batch_size//4   # learn batch
epochs = 3                                   # learning epochs time
VFcoeff = 1                                  # same as PPO paper

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
a_num = env.action_space.n
del env

use_RNN = False
hidden_unit_num = 128
if use_RNN:
    k = 1                                    # can be other number
else:
    k = 4                                    # frame stack number
