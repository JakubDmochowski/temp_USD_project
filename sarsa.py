import numpy as np 
import gym
import json
import matplotlib.pyplot as plot
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
loaded_episode = 40 # available_models: 40

# set load_snapshot to False to create model from scratch
load_snapshot = "./model_snapshots/model_sarsa2_ep{}".format(loaded_episode)

env = gym.make('SpaceInvaders-ram-v0') 
if(load_snapshot):
	model = keras.models.load_model(load_snapshot)
else:
	inputs = layers.Input(shape=(env.observation_space.shape[0] + 1,))

	hidden_layer_1 = layers.Dense(64, activation="relu")(inputs)
	normalized = layers.LayerNormalization()(hidden_layer_1)
	value = layers.Dense(1)(normalized)
	model = keras.Model(inputs=inputs, outputs=value)

optimizer = keras.optimizers.Adam(learning_rate=0.01)
def to_state_index(state):
	result = ''
	for s in state:
		result += '{:02x}'.format(s)
	return hash(result) 

#Function to choose the next action 
def choose_action(state): 
	action=0

	if np.random.uniform(0, 1) < epsilon: 
		action = env.action_space.sample() 
	else: 
		best = 0
		for i in range(0,env.action_space.n):
			t = state.copy()
			t= np.append(t,i)
			tmp = model(np.array([t]))
			if tmp > best:
				action = i
		
	return action 

#Function to learn the Q-value 
def update(state, next_state, reward, action, action2): 
	x = 1 + 1
	#Q[i1][action] = Q[i1][action] + beta * (reward + gamma * Q[i2][action2] - Q[i1][action]) 

beta = 1
epsilon = .6
total_episodes = 100
#beta = 0.85
gamma = 0.95


#Initializing the Q-matrix 
Q = dict()


reward=0
rewards_per_episode = []
# Starting the SARSA learning 
for episode in range(total_episodes): 
	previous_state = env.reset()
	previous_action = choose_action(previous_state)
	total_reward = 0
	while True: 
		
		env.render() 
	
		next_state, reward, done, info = env.step(previous_action) 
		next_state = list(map(float, next_state))
		total_reward += reward
		
		if np.random.uniform(0, 1) < epsilon: 
			action2 = env.action_space.sample() 
		else: 
			best = 0
			best_action = 0
			for i in range(0,env.action_space.n):
				t = next_state.copy()
				t= np.append(t,i)
				tmp = model(np.array([t]))
				if tmp > best:
					best = tmp
					best_action = i
				action2 = best_action
		

		
		t = previous_state.copy()
		t = np.append(t, previous_action)
		r = next_state.copy()
		r = np.append(r, action2)
		r = model(np.array([r]))
		with tf.GradientTape() as tape:
			expected = model(np.array([t]))
			gradient = tape.gradient(tf.constant((-expected +reward)), model.trainable_variables)
			#gradient = gradient * beta
			optimizer.apply_gradients(zip(gradient, model.trainable_variables))
		previous_state = next_state 
		previous_action = action2 
		if done: 
			break
	if(not episode % 5):
		model.save('./model_snapshots/model_sarsa2_ep{}'.format(episode))

	epsilon -= 0.08
	epsilon = max(0.05, epsilon)
	rewards_per_episode.append(total_reward)
z = np.polyfit(range(0, len(rewards_per_episode)), rewards_per_episode, 1)
p = np.poly1d(z)
plot.plot(rewards_per_episode)
plot.plot(p(range(0, len(rewards_per_episode))))
plot.show()



#Visualizing the Q-matrix 
print(str(Q)) 
