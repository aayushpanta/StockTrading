'''pip install stable_baselines3[extra] gym-anytrading
   pip install finta
'''
from flask import flash
# Gym stuff
import gym
#import gym_anytrading
from environment import StocksEnv
#from gym_anytrading.envs import StocksEnv

# Stable baselines - rl stuff
from stable_baselines3.common.vec_env import DummyVecEnv
#from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines3 import A2C, DQN, PPO

# Processing libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from finta import TA

# algo  = input("Choose the algorithm : ")

def preprocess(df):
    df['Date'] = pd.to_datetime(df['Date'])     #change datatype of date
    df.set_index('Date', inplace=True)          #set date as index
    df.drop(['Symbol'], axis=1, inplace=True)   #remove symbol from dataframe
    df['Volume'] = df['Volume'].apply(lambda x: float(x))   #change datatype of volume to float
    df.drop('Percent Change', axis=1, inplace=True) #remove percentage change from dataframe

    df['SMA'] = TA.SMA(df, 12)  #add custom fields to dataframe
    df['RSI'] = TA.RSI(df)
    df['OBV'] = TA.OBV(df)
    df.fillna(0, inplace=True)  #replace N/A by zeros
    return df

def add_signals(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'Low'].to_numpy()[start:end]
    signal_features = env.df.loc[:, ['Low', 'Volume','SMA', 'RSI', 'OBV']].to_numpy()[start:end]
    return prices, signal_features

class MyCustomEnv(StocksEnv):
    _process_data = add_signals

def training(modelname, algo, df, start_date, end_date ):
    #print(df)
    df = pd.read_csv('static/dataset/'+df)
    df = preprocess(df)
    df.loc[start_date:end_date]
    #end = len(df)
    env = MyCustomEnv(df=df, window_size=12, frame_bound=(12,len(df)))

    env_maker = lambda: env
    env3 = DummyVecEnv([env_maker])

    if algo == 'PPO':
        model = PPO('MlpPolicy', env3, verbose=1, tensorboard_log="/tensorboard/")
        model.learn(total_timesteps=10000) 

    if algo == 'A2C':
        model = A2C('MlpPolicy', env3, verbose=1, tensorboard_log="/tensorboard/") 
        model.learn(total_timesteps=10000)

    if algo == 'DQN':
        model = DQN('MlpPolicy', env3, verbose=1, tensorboard_log="/tensorboard/") 
        model.learn(total_timesteps=400000)

    
    # fname = 'static\saved models'+ modelname
    print("Model saved")#static\saved models  static\saved models\1st_PPO.zip
    model.save('static\saved models\\'+ modelname)
    del model

def prediction(modelname, bankname, start_date, end_date):
    name = bankname.split('.')[0]   
    df = pd.read_csv('static/dataset/'+bankname)
    df = preprocess(df)
    df = df.loc[start_date:end_date]
    #end_frame= len(df.loc[start_date:end_date])
    env4 = MyCustomEnv(df=df, window_size=12, frame_bound=(12,len(df)))
    algo = modelname.split('_')[-1]   #PPO.zip
    print("from prediction, algorithm is --", algo)
    algo = algo.split('.')[0]         #PPO
    print(algo)
    #obs= env4.reset()
    fname =  'static\saved models\\'+modelname
    #print(fname)
    #flash(fname, "Primary")

    if algo == 'PPO':
        model = PPO.load(fname) \

    if algo == 'A2C':
        model = A2C.load(fname) 

    if algo == 'DQN':
        model = DQN.load(fname)

    print("algorithm used is ", algo)
    obs= env4.reset() 
    while True:  
        obs = obs[np.newaxis, ...]
        action, _states = model.predict(obs)
        #print(action)
        obs, rewards, done, info = env4.step(action)
        #print(rewards, end =",")
        if done:
            print()
            print("info", info)
            #graph(env4, name)
            break
    print(len(df))
    del model
    return env4, name

def graph(env, str, model):
    #plt.figure(figsize=(12,6))
    #plt.cla()
    plt.title(str)
    env.render_all()
    np.save('static\portfolio\\'+str+'_'+model, env.portfolio)

def graph2(env, str):
    x=[]
    for i in range(0,len(env.portfolio)):
        x.append(i)

    plt.plot(x, env.portfolio, color='b')

    plt.xlabel("Days")
    plt.ylabel("Portfolio value")
    plt.title("Portfolio values for "+str+ " using DRL")
