import matplotlib
from matplotlib import pyplot as plt
import numpy as np 

bank = 'nica' 

portvalue = np.load('portfolio array\\'+bank+'_a2c 1 .npy')
portvalue1 = np.load('portfolio array\\'+bank+'_dqn 1 .npy')
portvalue2 = np.load('portfolio array\\'+bank+'_ppo 1 .npy')

x=[]
for i in range(0,len(portvalue)):
  x.append(i)
x1=[]
for i in range(0,len(portvalue1)):
  x1.append(i)
x2=[]
for i in range(0,len(portvalue2)):
  x2.append(i)


plt.figure(figsize=(10,6))
plt.cla()
plt.plot(x, portvalue, color='r', label = 'A2C')

plt.plot(x1, portvalue1, color='g', label = 'DQN')
plt.plot(x2, portvalue2, color='b', label = 'PPO')

plt.xlabel("Days")
plt.ylabel("Portfolio value")
plt.title("Portfolio values for NICA using DRL")

plt.legend()
plt.show()

def plt2(env):
  f2 = plt.figure(2)
  x=[]
  for i in range(0,len(env.portfolio)):
      x.append(i)

  plt.plot(x, env.portfolio, color='b')

  plt.xlabel("Days")
  plt.ylabel("Portfolio value")
  plt.title("Portfolio values for "+str+ " using DRL")

  #print(len(env.portfolio))
  plt.show(block=False) 