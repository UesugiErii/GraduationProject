import os
import sys

print(sys.argv)

for i in range(1,len(sys.argv)):
    env_num = str(sys.argv[i])
    os.system('python3 run.py ' + env_num)