import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.compat.v1.enable_eager_execution()

algorithm_name = "actor_critic_lambda_ram_joined"
version = "1"
save_models = True
loaded_episode = 1265 # available_models: 130, 345, 630, 900, 1265

# set load_snapshot to False to create model from scratch
load_snapshot = "./model_snapshots/model_{}_{}_ep{}".format(algorithm_name, version, loaded_episode)
episode_count = loaded_episode if load_snapshot else 0
num_episodes = 100000

seed = 9
gamma = 0.99 #discount factor
_lambda = 0.99
max_steps_per_episode = 10000
env = gym.make('SpaceInvaders-ram-v0')

env.seed(seed)
eps = np.finfo(np.float32).eps.item()

num_inputs = 128
hidden_layer_1_units = 64
hidden_layer_2_units = 64
num_actions = 6
num_critic = 1

inputs = layers.Input(shape=(num_inputs,))


model = None
actions_constraint=tf.keras.constraints.MinMaxNorm(min_value=0.0,max_value=1.0,rate=1.0,axis=0)

if(load_snapshot):
    model = keras.models.load_model(load_snapshot)
else:
    hidden_layer_1 = layers.Dense(hidden_layer_1_units, activation="relu")(inputs)
    hidden_layer_2 = layers.Dense(hidden_layer_2_units, activation="relu")(hidden_layer_1)
    normalize = layers.LayerNormalization(epsilon=eps)(hidden_layer_2)
    actions = layers.Dense(num_actions, activation="softmax", kernel_constraint=actions_constraint)(normalize)
    critic = layers.Dense(num_critic)(normalize)
    model = keras.Model(inputs=inputs, outputs=[actions, critic])

optimizer = keras.optimizers.Adam(learning_rate=0.0001)

def map_to_grayscale(state):
    Y = range(len(state))
    X = range(len(state[0]))
    C = range(len(state[0][0]))
    mapped = np.array([[0 for col in X] for row in Y])
    for y in Y:
        for x in X:
            val = 0
            for c in C:
                val += state[y][x][c]
            mapped[y][x] = val / 3
    return mapped

_lamb_gamm_const = tf.constant(_lambda * gamma)
running_reward = 0

while episode_count in range(num_episodes):
    state = env.reset()
    state = tf.convert_to_tensor(state)
    state = tf.expand_dims(state, 0)
    episode_reward = 0
    done = False
    zy = None
    reward_history = []
    critic_value_history = []
    frame_count = 0
    while not done:
        frame_count += 1
        with tf.GradientTape() as tape:
            env.render()

            action_probs, critic_value = model(state)
            critic_value_history.append(critic_value)
        
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))

            print("critic_value: {}".format(critic_value.numpy()[0,0]))
            state, reward, done, _ = env.step(action)
            reward_history.append(reward)
            episode_reward += reward
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)
            
            time_diff = reward - critic_value.numpy()[0,0]
            if(not done):
                fut_action_probs, fut_critic_val = model(state)
                time_diff += fut_critic_val.numpy()[0,0] * gamma
            print("action probabilities: {}".format(action_probs[0].numpy()))

            log_prob = tf.math.log(action_probs[0, action])
            log_action_probs = list(map(lambda x: tf.math.log(x+eps), action_probs[0]))
            args = [log_action_probs, critic_value[0]]
            if zy is None:
                zy = tape.gradient(args, model.trainable_variables)
            else:
                tmp_zy = list(map(lambda m: tf.scalar_mul(_lamb_gamm_const, m) if m is not None else m, zy))
                tmp2_zy = tape.gradient(args, model.trainable_variables)
                zy = []
                for i in range(len(tmp_zy)):
                    zy.append(tmp_zy[i] + tmp2_zy[i])
            td_zy = []
            for i in range(len(zy)):
                td_zy.append(tf.scalar_mul(time_diff, zy[i]))
            
            optimizer.apply_gradients(zip(td_zy, model.trainable_variables))
            
    running_reward = 0.95 * running_reward + 0.05 * episode_reward
    episode_count += 1
    print("episode {}: running_reward:{}, rewarded with {}".format(episode_count, running_reward, episode_reward))
    if(not episode_count % 5 and save_models):
        model.save('./model_snapshots/model_{}_{}_ep{}'.format(algorithm_name, version, episode_count))
