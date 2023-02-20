from argparse import Action
import tensorflow as tf
import tensorflow.python.keras as keras
from keras import optimizers
from tensorflow.python.keras.optimizer_v2.adam import Adam
import numpy as np
from keras.layers import Dense
from keras.layers import Flatten
from math import *

import csv

class DuelingDQN(keras.Model):
    def __init__(self, n_actions, fc1Dims, fc2Dims):
        super(DuelingDQN, self).__init__()
        self.flatten = Flatten()
        self.dense1 = Dense(fc1Dims, activation='relu')
        self.dense2 = Dense(fc2Dims, activation='relu')
        self.V = Dense(1, activation=None)
        self.A = Dense(n_actions, activation=None)
    

    def call(self, state):
        x = self.flatten(state)
        x = self.dense1(x)
        x = self.dense2(x)
        V = self.V(x)
        A = self.A(x)
        
        Q = (V + (A - tf.reduce_mean(A, axis=1, keepdims=True)))
        return Q

    def advantage(self, state):
        x = self.flatten(state)
        x = self.dense1(x)
        x = self.dense2(x)
        A = self.A(x)

        return A


class ReplayBuffer():
    def __init__(self, max_size, input_shape, per_on, rwd_components):
        self.per_on = per_on #Prioritized Experience Replay 사용 여부
        print('Use Prioritized Sampling:', self.per_on)

        self.mem_size = max_size
        self.mem_cntr = 0 #인덱스 지정을 위한 카운터
        self.mem_N = 0 #저장된 데이터 개수
        self.rwd_components = rwd_components

        #NumPy 행렬을 이용한 메모리 구현
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros((self.mem_size, self.rwd_components), dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool8)
        self.tderror_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.blackbox_memory = np.zeros(self.mem_size, dtype=np.float32)


    def store_transition(self, state, action, reward, new_state, done, tderror):
        self.mem_cntr += 1
        
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        for i in range(self.rwd_components):
            self.reward_memory[index, i] = reward[i]
        self.terminal_memory[index] = done
        self.tderror_memory[index] = tderror
        self.blackbox_memory[index] = 1e-7

        self.mem_N = max(index, self.mem_N)
    
    def set_blackbox(self, cnt):
        for i in range(cnt):
            idx = (self.mem_cntr - i) % self.mem_size
            self.blackbox_memory[idx] = 1.0
    
    def update_tderror(self, index, tderror):
        self.tderror_memory[index] = tderror

    def sample_buffer(self, batch_size, alpha, exp_no):

        sample_scores = self.tderror_memory[:self.mem_N]
        sample_scores = np.power(sample_scores, alpha) + 0.01
        # https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.htm
        if self.per_on: #Prioritized (Stochatic) Sampling
            # (1) Prioritization Based on TD-Error

            sample_prob = sample_scores / np.sum(sample_scores)
            
            batch = np.random.choice(self.mem_N, batch_size, replace=False, p=sample_prob)

            #debug
            #print(sample_scores[batch])

        else: #Random Sampling
            batch = np.random.choice(self.mem_N, batch_size, replace=False)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return batch, states, actions, rewards, new_states, dones

class Agent(): #신경망 학습을 관장하는 클래스
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size, input_dims, per_on,
        eps_dec = 1e-4, eps_end = 0.01, mem_size = 500000, fc1_dims=128,
        fc2_dims=128, replace = 100):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_end = eps_end
        self.replace = replace
        self.batch_size = batch_size

        self.rwd_components = 3
        self.learn_step_counter = 0
        self.memory = ReplayBuffer(mem_size, input_dims, per_on, self.rwd_components)
        self.episode_frame_cnt = 0 # BlackBox Prioritization을 위한 카운터

        # Double DQN
        self.q_evals = []
        self.q_nexts = []

        for i in range(self.rwd_components):
            self.q_evals.append(DuelingDQN(n_actions, fc1_dims, fc2_dims))
            self.q_nexts.append(DuelingDQN(n_actions, fc1_dims, fc2_dims))
            self.q_evals[i].compile(optimizer=Adam(learning_rate=lr), loss = "mse")
            self.q_nexts[i].compile(optimizer=Adam(learning_rate=lr), loss = "mse")


        self.init_q_next = True


    def store_transition(self, state, action, reward, new_state, done, pred):
        #TD-Error 계산
        state_ = np.array([new_state])
        q_next = 0.0
        for i in range(self.rwd_components):
            q_next += self.q_nexts[i](state_)

        target = reward.sum() + np.max(self.gamma * q_next * (1-int(done)))
        tderror = abs(target - pred)
        #print('TD-Error:', tderror)

        self.memory.store_transition(state, action, reward, new_state, done, tderror)
        self.episode_frame_cnt += 1
    
    def choose_action(self, observation):
        state = np.array([observation])
        
        actions = np.zeros((1, len(self.action_space)))
        pred_C = []
        for i in range(self.rwd_components):
            pred_c = self.q_evals[i](state).numpy()
            actions += pred_c
            pred_C.append(pred_c)
        
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            action = actions.argmax(axis=1)[0]
        
        pred = np.max(actions, axis=1)[0]
        return action, pred, pred_C

    def choose_actions(self, observation):
        states = observation
        
        actions = np.zeros((self.batch_size, len(self.action_space)))
        for i in range(self.rwd_components):
            actions += self.q_evals[i](states).numpy()
        
        max_actions = actions.argmax(axis=1)
        #print(max_actions)
        
        return max_actions, np.max(actions, axis=1)

    def set_next_weight(self):
        for i in range(self.rwd_components):
            self.q_nexts[i].set_weights(self.q_evals[i].get_weights())
        print("q_next weight set!")
    
    def learn(self, exp_no):
        if self.memory.mem_N < self.batch_size:
            return 0.0
        
        batch, states, actions, rewards, states_, dones = \
            self.memory.sample_buffer(self.batch_size, self.epsilon, exp_no)
            #alpha ~ 1.0 ~ 0.0

        #if self.learn_step_counter % self.replace == 0:
        #    for i in range(self.rwd_components):
        #        self.q_nexts[i].set_weights(self.q_evals[i].get_weights())
        #    print("q_next weight set!")
        
        
        loss = np.zeros(self.rwd_components)

        q_t = np.zeros((len(dones), len(self.action_space))) # 전체 Component를 합산한 target Q 값
        for i in range(self.rwd_components):
            q_pred = self.q_evals[i](states)
            next_actions = np.argmax(self.q_evals[i](states_), axis=1)

            q_next = self.q_nexts[i](states_)
            q_target = q_pred.numpy()

            #Component별 target Q value 계산
            for idx, terminal in enumerate(dones):
                q_target[idx, actions[idx]] = rewards[idx, i] + \
                    self.gamma*( q_next[idx, next_actions[idx]] )*(1-int(dones[idx]))
                    #self.gamma*( np.max(q_next[idx]) )*(1-int(dones[idx]))
            
            q_t += q_target

            #Conponent별 Q function 학습
            loss[i] = self.q_evals[i].train_on_batch(states, q_target)
        
        #데이터 학습에 사용 후 저장된 TD-Error 값 업데이트 (모든 Reward Compoenent 고려)
        max_actions, pred = self.choose_actions(states)
        for idx in range(len(dones)):
            tderror = abs(np.max(q_t[idx]) - pred[idx])
            self.memory.update_tderror(batch[idx], tderror)

        #print(loss)
        # epsilon 조정은 훈련 코드에서 수동으로 하는 걸로 조정함. (20220920)
        #if self.epsilon > self.eps_end:
        #    self.epsilon -= self.eps_dec
        #else:
        #    self.epsilon = self.eps_end
        self.learn_step_counter += 1

        return loss

    def save_model(self, path):
        for i in range(self.rwd_components):
            self.q_evals[i].save_weights(path + f'_r{i}')
        print("saved weights to " + path)
    
    def load_model(self, path):
        
        for i in range(self.rwd_components):
            self.q_evals[i].load_weights(path + f'_r{i}').expect_partial()
            self.q_nexts[i].load_weights(path + f'_r{i}').expect_partial()
        print("loaded weights from " + path)
