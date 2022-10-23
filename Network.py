from argparse import Action
import tensorflow as tf
import tensorflow.python.keras as keras
from keras import optimizers
from tensorflow.python.keras.optimizer_v2.adam import Adam
import numpy as np
from keras.layers import Dense
from keras.layers import Multiply
from math import *

class ActionWeightLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
    super(ActionWeightLayer, self).__init__()
    self.num_outputs = num_outputs
    self.trainable = False #훈련을 해도 가중치가 바뀌지 않음

  def build(self, input_shape):
    self.kernel = self.add_weight("kernel",
                                  shape=[int(input_shape[-1]),
                                         self.num_outputs])
  def set_action_weights(self, weights):
    self.kernal = weights

  def call(self, inputs):
    return tf.math.multiply(inputs, self.kernal)

class DuelingDQN(keras.Model):
    def __init__(self, n_actions, fc1Dims, fc2Dims, fc3Dims):
        super(DuelingDQN, self).__init__()
        self.dense1 = Dense(fc1Dims, activation='relu')
        self.dense2 = Dense(fc2Dims, activation='relu')
        self.dense3 = Dense(fc3Dims, activation='relu')
        self.V = Dense(1, activation=None)
        #self.A = Dense(n_actions, activation='softplus')
        self.A = Dense(n_actions, activation=None)
        self.Aw = ActionWeightLayer(n_actions)
        #self.Aw.set_action_weights([0.0701, 0.0611, 0.0065, 0.1748, 0.1370])
        self.Aw.set_action_weights([1.0, 1.0, 1.0, 1.0, 1.0]) # -> Action Weight가 없는 것과 같은 효과
    

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        x = self.dense3(x)
        V = self.V(x)
        A = self.A(x)
        A = self.Aw(A)
        
        Q = (V + (A - tf.reduce_mean(A, axis=1, keepdims=True)))
        return Q

    def advantage(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        x = self.dense3(x)
        A = self.A(x)
        A = self.Aw(A)

        return A


class ReplayBuffer():
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0 #인덱스 지정을 위한 카운터

        #NumPy 행렬을 이용한 메모리 구현
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool8)


    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        # https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
        # PER 구현시 참고할것

        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, new_states, dones

class Agent(): #신경망 학습을 관장하는 클래스
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size, input_dims,
        eps_dec = 1e-4, eps_end = 0.01, mem_size = 100000, fc1_dims=32,
        fc2_dims=32, fc3_dims=16, replace = 100):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_end = eps_end
        self.replace = replace
        self.batch_size = batch_size

        self.learn_step_counter = 0
        self.memory = ReplayBuffer(mem_size, input_dims)

        # Double DQN
        self.q_eval = DuelingDQN(n_actions, fc1_dims, fc2_dims, fc3_dims)
        self.q_next = DuelingDQN(n_actions, fc1_dims, fc2_dims, fc3_dims)

        self.action_weights = [1.0, 1.0, 1.0, 1.0, 1.0] #Action weight 초깃값 (적용하지 않은 것과 같은 상태)

        self.q_eval.compile(optimizer=Adam(learning_rate=lr), loss = "mse")
        self.q_next.compile(optimizer=Adam(learning_rate=lr), loss = "mse")

        self.init_q_next = True
        #self.q_eval.compile(optimizer=Adam(learning_rate=lr), loss = "mean_squared_error")
        #self.q_next.compile(optimizer=Adam(learning_rate=lr), loss = "mean_squared_error")

    def set_action_weights(self, weights):
        self.action_weights = weights
        self.q_eval.Aw.set_action_weights(weights)
        self.q_next.Aw.set_action_weights(weights)


    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
    
    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            weight = np.power(self.action_weights, self.epsilon)
            weight = weight / np.sum(weight)
            action = np.random.choice(self.action_space, p=weight)
        else:
            state = np.array([observation])
            actions = self.q_eval.advantage(state)
            action = tf.math.argmax(actions, axis=1).numpy()[0]

        return action
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        states, actions, rewards, states_, dones = \
            self.memory.sample_buffer(self.batch_size)

        if self.learn_step_counter % self.replace == 0:
            self.q_next.set_weights(self.q_eval.get_weights())
            #print("q_next weight set!")

        q_pred = self.q_eval(states)
        q_next = self.q_next(states_)
        q_target = q_pred.numpy()

        #print(self.q_eval(np.array([[0,0,0,0,0,0]])))
        max_actions = tf.math.argmax(self.q_eval(states_), axis=1)

        for idx, terminal in enumerate(dones):
            q_target[idx, actions[idx]] = rewards[idx] + \
                self.gamma*q_next[idx, max_actions[idx]]*(1-int(dones[idx]))

        loss = self.q_eval.train_on_batch(states, q_target)

        #print(loss)
        # epsilon 조정은 훈련 코드에서 수동으로 하는 걸로 조정함. (20220920)
        #if self.epsilon > self.eps_end:
        #    self.epsilon -= self.eps_dec
        #else:
        #    self.epsilon = self.eps_end
        self.learn_step_counter += 1

    def save_model(self, path):
        self.q_eval.save_weights(path)
        print("saved weights to " + path)
    
    def load_model(self, path):
        self.q_eval(np.array([[0,0,0,0,0,0]]))
        self.q_next(np.array([[0,0,0,0,0,0]]))

        self.q_eval.load_weights(path)
        self.q_next.load_weights(path)
        print("loaded weights from " + path)



#ActionWeightLayer 테스트

#Aw = ActionWeightLayer(5)
#Aw.setweight([0.017, 0.728, 0.0026, 0.119, 0.134])
#_ = np.array([1.0,1.0,1.0,1.0,1.0])
#__ = Aw(_)
#print(_)
#print(__)