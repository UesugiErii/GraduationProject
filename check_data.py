import tensorflow as tf
from multiprocessing import Process
from tensorboard.backend.event_processing import event_accumulator
import os

ban_level = ['4-4','7-4','8-4',]

root_dir = '/media/zx/8ACAF3CECAF3B493/Linux/data'

def f(work_dir):
    for i in range(1,9):
        for j in range(1,4):
            level = str(i) + '-' + str(j)
            if level in ban_level:
                continue
            level_full_name = 'SuperMarioBros-{}-v0'.format(level)
            event_path = work_dir + '/' + level_full_name + '/metrics'
            all_file = os.listdir(event_path)
            event_path = event_path + '/' + all_file[0]

            ea=event_accumulator.EventAccumulator(
            event_path,
            size_guidance={event_accumulator.TENSORS: 0,event_accumulator.SCALARS: 0,}  # 0 means load all data
            )                       # see other use dir(event_accumulator)


            ea.Reload()  # loads events from file
            ea.Tags()

            ['one_episode_reward', 'one_episode_flag_get', 'total loss', 'Lclip', 'H', 'c loss']


            if len(ea.Tags()['tensors']) != 6:
                print('-'*20)
                print('lack tags')
                print(work_dir)
                print(i,j)
            elif len(ea.Tensors('H')) != 586:
                print('-' * 20)
                print('lack H')
                print(work_dir)
                print(i, j)
                print(len(ea.Tensors('H')))
            elif len(ea.Tensors('total loss')) != 586:
                print('-' * 20)
                print('lack total loss')
                print(work_dir)
                print(i, j)
                print(len(ea.Tensors('total loss')))
            elif len(ea.Tensors('Lclip')) != 586:
                print('-' * 20)
                print('lack Lclip')
                print(work_dir)
                print(i, j)
                print(len(ea.Tensors('Lclip')))
            elif len(ea.Tensors('c loss')) != 586:
                print('-' * 20)
                print('lack c loss')
                print(work_dir)
                print(i, j)
                print(len(ea.Tensors('c loss')))
            elif len(ea.Tensors('one_episode_flag_get')) != len(ea.Tensors('one_episode_reward')):
                print('-' * 20)
                print('flag != reward')
                print(work_dir)
                print(i, j)
                print(len(ea.Tensors('one_episode_flag_get')))
                print(len(ea.Tensors('one_episode_reward')))



    print(work_dir,'done')

p_l = []

# origin    single_sc
#   OK         OK

for t in range(1,4):
    work_dir = root_dir + '/' + str(t) + ''
    p_l.append(Process(target=f, args=(work_dir,)))
for t in range(1,4):
    work_dir = root_dir + '/' + str(t) + 'sc'
    p_l.append(Process(target=f, args=(work_dir,)))
for t in range(1,4):
    work_dir = root_dir + '/' + str(t) + 'sc-full'
    p_l.append(Process(target=f, args=(work_dir,)))


for i in p_l:
    i.start()


