from multiprocessing import Process
from communication import Communication
from config import *
from brain import ACBrain
from agent import Agent
from environment import Env
import subprocess
from util import get_seed
import sys
import time
#   tensorboard --logdir logs/scalars
#python3 run.py 'SuperMarioBros-4-1-v0' ; python3 run.py 'SuperMarioBros-5-1-v0' ; python3 run.py 'SuperMarioBros-6-1-v0' ; python3 run.py 'SuperMarioBros-7-1-v0'

def main():
    # num = sys.argv[1]
    num = '81'
    print(num)
    env_name = 'SuperMarioBros-{}-{}-v0'.format(num[0], num[1])
    print(env_name)
    communication = Communication(child_num=process_num)

    brain = ACBrain(talker=communication.master,env_name=env_name)

    envs_p = []

    seed = get_seed()
    for i in range(process_num):
        agent = Agent(talker=communication.children[i])
        env_temp = Env(agent, i, seed=seed+i,env_name=env_name)
        envs_p.append(Process(target=env_temp.run, args=()))

    for i in envs_p:
        i.start()

    tfb_p = subprocess.Popen(['tensorboard', '--logdir', "./logs/scalars"])

    brain.run()

    for p in envs_p:
        p.terminate()
    tfb_p.kill()


if __name__ == '__main__':
    main()
