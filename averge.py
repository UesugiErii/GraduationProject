import tensorflow as tf
import numpy as np
from multiprocessing import Process
from tensorboard.backend.event_processing import event_accumulator
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from collections import defaultdict
ban_level = ['1-3','4-3','4-4','5-3','7-4','8-4',]
root_dir = '/media/zx/8ACAF3CECAF3B493/Linux/data'


# tf.summary.scalar(name, data=data, step=step)



def f(work_dir,sc):



    # for i in range(1,9):
    #     for j in range(1,5):
    for level in ['8-1','8-2','8-3',]:
            # level = str(i) + '-' + str(j)
            if level in ban_level:
                continue
            level_full_name = 'SuperMarioBros-{}-v0'.format(level)

            logdir = "./logs/scalars/averge" + '/' + level_full_name
            file_writer = tf.summary.create_file_writer(logdir + "/metrics")
            file_writer.set_as_default()

            d = defaultdict(list)

            for t in range(1,4):
                if not sc:
                    event_path = work_dir + '/' + str(t) + '/' + level_full_name + '/metrics'
                else:
                    event_path = work_dir + '/' + str(t) + 'sc' + '/' + level_full_name + '/metrics'
                all_file = os.listdir(event_path)
                event_path = event_path + '/' + all_file[0]

                ea=event_accumulator.EventAccumulator(
                event_path,
                size_guidance={event_accumulator.TENSORS: 0,event_accumulator.SCALARS: 0,}  # 0 means load all data
                )                       # see other use dir(event_accumulator)


                ea.Reload()  # loads events from file
                for v in ea.Tags()['tensors']:  # <class 'list'>: ['one_episode_reward', 'one_episode_flag_get', 'total loss', 'Lclip', 'H', 'c loss']
                    length = len(ea.Tensors(v))
                    temp = []
                    for k in range(length):
                        tensor_content = ea.Tensors(v)[k].tensor_proto.tensor_content
                        dtype = ea.Tensors(v)[k].tensor_proto.dtype
                        data = float(tf.io.decode_raw(tensor_content, dtype))
                        temp.append(data)
                        # print(k)
                    d[v].append(temp)
                    print(level,t,v,'done')


            for key in d:
                if key == 'one_episode_reward':
                    k1 = 'one_episode_reward'
                    k2 = 'one_episode_flag_get'
                    data1 = d[k1]
                    data2 = d[k2]
                    min_len = min(len(data1[0]),len(data1[1]),len(data1[2]))
                    data1 = [d[k1][0][:min_len],d[k1][1][:min_len],d[k1][2][:min_len]]
                    data2 = [d[k2][0][:min_len],d[k2][1][:min_len],d[k2][2][:min_len]]
                    data1 = np.array(data1).astype(np.float32)
                    data2 = np.array(data2).astype(np.float32)
                    data1 = np.mean(data1,axis=0)
                    data2 = np.mean(data2,axis=0)

                    for step in range(min_len):
                        tf.summary.scalar(k1, data=data1[step], step=step)
                        tf.summary.scalar(k2, data=data2[step], step=step)
                else:
                    continue





# t = ea.Tensors('H')[0].tensor_proto.tensor_content
# d = ea.Tensors('H')[0].tensor_proto.dtype
# print(float(tf.io.decode_raw(t, d)))


# f(root_dir,sc=False)

# f(root_dir,sc=True)
# print('done')



def g(work_dir,l):



    # for i in range(1,9):
        # for j in range(1,5):
    # for level in ['1-3','4-3','5-3',]:
    for level in ['2-4',]:
            # level = str(i) + '-' + str(j)
            # if level in ban_level:
            #     continue
            level_full_name = 'SuperMarioBros-{}-v0'.format(level)

            logdir = "./logs/scalars/averge"  + l + '/' + level_full_name
            file_writer = tf.summary.create_file_writer(logdir + "/metrics")
            file_writer.set_as_default()

            d = defaultdict(list)

            for t in range(1,4):
                event_path = work_dir + '/' + str(t) + l + '/' + level_full_name + '/metrics'
                all_file = os.listdir(event_path)
                event_path = event_path + '/' + all_file[0]

                ea=event_accumulator.EventAccumulator(
                event_path,
                size_guidance={event_accumulator.TENSORS: 0,event_accumulator.SCALARS: 0,}  # 0 means load all data
                )                       # see other use dir(event_accumulator)


                ea.Reload()  # loads events from file
                for v in ea.Tags()['tensors']:  # <class 'list'>: ['one_episode_reward', 'one_episode_flag_get', 'total loss', 'Lclip', 'H', 'c loss']
                    length = len(ea.Tensors(v))
                    temp = []
                    for k in range(length):
                        tensor_content = ea.Tensors(v)[k].tensor_proto.tensor_content
                        dtype = ea.Tensors(v)[k].tensor_proto.dtype
                        data = float(tf.io.decode_raw(tensor_content, dtype))
                        temp.append(data)
                        # print(k)
                    d[v].append(temp)
                    print(level,t,v,'done')


            for key in d:
                if key == 'one_episode_reward':
                    k1 = 'one_episode_reward'
                    k2 = 'one_episode_flag_get'
                    data1 = d[k1]
                    data2 = d[k2]
                    min_len = min(len(data1[0]),len(data1[1]),len(data1[2]))
                    data1 = [d[k1][0][:min_len],d[k1][1][:min_len],d[k1][2][:min_len]]
                    data2 = [d[k2][0][:min_len],d[k2][1][:min_len],d[k2][2][:min_len]]
                    data1 = np.array(data1).astype(np.float32)
                    data2 = np.array(data2).astype(np.float32)
                    data1 = np.mean(data1,axis=0)
                    data2 = np.mean(data2,axis=0)

                    for step in range(min_len):
                        tf.summary.scalar(k1, data=data1[step], step=step)
                        tf.summary.scalar(k2, data=data2[step], step=step)
                else:
                    continue

# g(root_dir,l='')
# g(root_dir,l='sc')
# g(root_dir,l='sc-full')

p_l = []
# p_l.append(Process(target=g, args=(root_dir,'')))
# p_l.append(Process(target=g, args=(root_dir,'sc')))
p_l.append(Process(target=g, args=(root_dir,'sc-full')))

for i in p_l:
    i.start()

print('done')