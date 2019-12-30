import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import numba
import tensorflow as tf
import time
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten
from atari_wrappers import *
from monitor import Monitor
from config import *
import gym
from model import CNNModel
from multiprocessing import Process


class ACAgent():
    def __init__(self, dir):
        super(ACAgent, self).__init__()
        self.Model = CNNModel(None,test=True)
        self.Model.load_weights(dir)

    def choice_action(self, state):
        data = self.Model(np.array(state)[np.newaxis, :].astype(np.float32))
        prob_weights = data[0].numpy()
        v = data[1].numpy()

        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())

        return action,v,prob_weights





class Env():
    def __init__(self, agent , video_dir):
        from nes_py.wrappers import JoypadSpace
        import gym_super_mario_bros
        from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
        env = gym_super_mario_bros.make(env_name)
        if record:
            env = Monitor(env, video_dir, force=True)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        # env =  NoopResetEnv(env , noop_max=env_id)
        env = WarpFrame(env, width=IMG_W, height=IMG_H, grayscale=True)
        env = MaxAndSkipEnv(env, skip=4)
        env = FrameStack(env, k=k)  # return (IMG_H , IMG_W ,k)
        self.env = env

        self.agent = agent

    def preprocess(self, state):
        return state

    @staticmethod
    @numba.jit(nopython=True)
    def calc_realv_and_adv_GAE(v, r, done):
        length = r.shape[0]
        num = r.shape[1]

        adv = np.zeros((length + 1, num), dtype=np.float32)

        for t in range(length - 1, -1, -1):
            delta = r[t, :] + v[t + 1, :] * gamma * (1 - done[t, :]) - v[t, :]
            adv[t, :] = delta + gamma * 0.95 * adv[t + 1, :] * (1 - done[t, :])

        adv = adv[:-1, :]

        realv = adv + v[:-1, :]

        return realv, adv

    def run(self):

        # use to show game

        np.random.seed(int(time.time()))
        self.env.seed(int(time.time()))
        state = self.env.reset()
        state = self.preprocess(state)
        one_episode_reward = 0
        step = 0
        while 1:
            step += 1
            a,v,prob_weights = self.agent.choice_action(state)
            # self.env.render()
            # time.sleep(0.03)
            state_, r, done, info = self.env.step(a)

            one_episode_reward += r

            state_ = self.preprocess(state_)

            state = state_

            if done:
                return one_episode_reward


    def run2(self):

        # use to show att

        process_num = 1
        batch_size = 10000
        total_obs = np.zeros((batch_size, process_num, IMG_H, IMG_W, k), dtype=np.float32)
        total_v = np.zeros((batch_size + 1, process_num), dtype=np.float32)
        total_as = np.zeros((batch_size, process_num), dtype=np.int32)
        total_rs = np.zeros((batch_size, process_num), dtype=np.float32)
        total_is_done = np.zeros((batch_size, process_num), dtype=np.float32)
        total_old_ap = np.zeros((batch_size, process_num, a_num), dtype=np.float32)

        temp_obs = np.zeros((IMG_H, IMG_W, k), dtype=np.float32)

        np.random.seed(int(time.time()))
        self.env.seed(int(time.time()))
        state = self.env.reset()
        temp_obs[:,:,:] = self.preprocess(state)

        i = 0


        while 1:
            if i >= batch_size:
                break

            a,v,prob_weights = self.agent.choice_action(temp_obs)
            # self.env.render()
            # time.sleep(0.03)

            total_obs[i, :, :, :, :] = temp_obs[np.newaxis, :]

            state_, r, done, info = self.env.step(a)

            total_v[i,:] = v
            total_as[i,:] = a
            r /= 60
            total_rs[i,:] = r
            total_is_done[i,:] = done
            total_old_ap[i,:] = prob_weights

            temp_obs[:,:,:] = state_

            i += 1

            if done:
                break

        total_obs = total_obs[:i,:,:,:,:]
        total_v = total_v[:i+1,:]
        total_as = total_as[:i,:]
        total_rs = total_rs[:i,:]
        total_is_done = total_is_done[:i,:]
        total_old_ap = total_old_ap[:i,:]

        total_realv, total_adv = self.calc_realv_and_adv_GAE(total_v, total_rs, total_is_done)


        random_i = np.random.randint(0,i)

        self.cnn_learn(
            total_obs[random_i,:,:,:,:],
            tf.one_hot(total_as[random_i,:], depth=a_num).numpy(),
            total_old_ap[random_i,:],
            total_adv[random_i,:],
            total_realv[random_i,:]
        )




    def cnn_learn(self, total_obs, total_as, total_old_ap, total_adv, total_real_v):


        att, loss  = self.agent.Model.total_grad2(total_obs,
                                            total_as,
                                            total_adv,
                                            total_real_v,
                                            total_old_ap)

        print(1)


# config
repeat_times = 10
index = 29295
weight_dir = '/media/zx/新加卷/data/1scw'
video_root_dir = '/media/zx/8ACAF3CECAF3B493/Linux/data/video'
ban_level = ['4-4','7-4','8-4',]
record = False



####################################################################################################

# 测试成绩和录像 get test average score and record video


#
for i in range(2,9):
    for j in range(4,5):
        level = str(i) + '-' + str(j)
        if level in ban_level:
            continue
        env_name = 'SuperMarioBros-{}-v0'.format(level)
        restore_weight_dir = weight_dir+'/'+env_name+'/'+str(index)

        res_l = []
        for t in range(1,repeat_times+1):
            video_dir = video_root_dir + '/' + level + '-' + str(t)
            if record and not os.path.exists(video_dir):
                os.makedirs(video_dir)
            env = Env(ACAgent(restore_weight_dir),video_dir=video_dir)
            res = env.run()
            res_l.append(res)

        print(i,j,'average:',sum(res_l)/repeat_times,'best:',res_l.index(max(res_l))+1)


####################################################################################################



# def f(i,j):
#     level = str(i) + '-' + str(j)
#     env_name = 'SuperMarioBros-{}-v0'.format(level)
#     restore_weight_dir = weight_dir + '/' + env_name + '/' + str(index)
#
#     res_l = []
#     for t in range(1, repeat_times + 1):
#         video_dir = video_root_dir + '/' + level + '-' + str(t)
#         if record and not os.path.exists(video_dir):
#             os.makedirs(video_dir)
#         env = Env(ACAgent(restore_weight_dir), video_dir=video_dir)
#         res = env.run()
#         res_l.append(res)
#
#     print(i, j, 'average:', sum(res_l) / repeat_times, 'best:', res_l.index(max(res_l)) + 1)
#
#
# envs_p = []
# for i in range(1, 5):
# # for i in range(5, 9):
#     for j in range(1, 5):
#         level = str(i) + '-' + str(j)
#         if level in ban_level:
#             continue
#         envs_p.append(Process(target=f, args=(i,j)))
#
#
# for i in envs_p:
#     i.start()



####################################################################################################

# use Grad-CAM to visualize attention
# level = '{}-{}'.format(8,1)
# env_name = 'SuperMarioBros-{}-v0'.format(level)
#
# restore_weight_dir = weight_dir+'/'+env_name+'/'+str(index)
#
# res_l = []
#
# env = Env(ACAgent(restore_weight_dir),video_dir=None)
# res = env.run2()
# res_l.append(res)

####################################################################################################


