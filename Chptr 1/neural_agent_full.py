#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gym
import envs
from tqdm import tqdm


# In[2]:


class Brain(keras.Model):
    def __init__(self, action_dim=5,input_shape=(1, 8 * 8)):
        """Initialize the Agent's Brain model
        Args:
        action_dim (int): Number of actions
        """
        super(Brain, self).__init__()
        self.dense1 = layers.Dense(32, input_shape=input_shape, activation="relu")
        self.logits = layers.Dense(action_dim)
        
    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        if len(x.shape) >= 2 and x.shape[0] != 1:
            x = tf.reshape(x, (1, -1))
        return self.logits(self.dense1(x))
    
    def process(self, observations):
        # Process batch observations using `call(inputs)`
        # behind-the-scenes
        action_logits = self.predict_on_batch(observations)
        return action_logits


# In[3]:


class Agent(object):
    def __init__(self, action_dim=5, input_shape=(1, 8 * 8)):
        """Agent with a neural-network brain powered policy
        Args:
            brain (keras.Model): Neural Network based model
        """
        self.brain = Brain(action_dim, input_shape)
        self.policy = self.policy_mlp

    def policy_mlp(self, observations):
        observations = observations.reshape(1, -1)
        # action_logits = self.brain(observations)
        action_logits = self.brain.process(observations)
        action = tf.random.categorical(tf.math.log(action_logits), num_samples=1)
        return tf.squeeze(action, axis=1)

    def get_action(self, observations):
        return self.policy(observations)

    def learn(self, samples):
        raise NotImplementedError


# In[4]:


#evaluate the agent in a given environment for one episode
def evaluate(agent, env, render=True):
    obs, episode_reward, done, step_num = env.reset(),0.0, False, 0
    while not done:
        action = agent.get_action(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        step_num += 1
        if render:
            env.render()
    return step_num, episode_reward, done, info


# In[7]:


#main function
if __name__ == "__main__":
    env = gym.make("Gridworld-v0")
    agent = Agent(env.action_space.n,env.observation_space.shape)
    for episode in tqdm(range(10)):
        steps, episode_reward, done, info = evaluate(agent, env)
        print(f"EpReward:{episode_reward:.2f} steps:{steps} done:{done} info:{info}")
    env.close()


# In[ ]:




