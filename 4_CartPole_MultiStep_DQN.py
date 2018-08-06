# Cartpole
# State  -> x, x_dot, theta, theta_dot
# Action -> force (+1, -1)

# Import modules
import tensorflow as tf
import random
import numpy as np
import copy
import matplotlib.pyplot as plt
import datetime
import time
import gym

env = gym.make('CartPole-v0')
game_name = 'CartPole'
algorithm = 'DQN'

################### Multi-step ###################
# Parameter for Multi-step
n_step = 1
episode_step = 1
state_list = []
reward_list = []
##################################################

# Parameter setting
Num_action = 2
Gamma = 0.99
Learning_rate = 0.00025
Epsilon = 1
Final_epsilon = 0.01

Num_replay_memory = 10000
Num_start_training = 5000
Num_training = 15000
Num_testing  = 10000
Num_update = 150
Num_batch = 32
Num_episode_plot = 20

first_fc  = [4, 512]
second_fc = [512, 128]
third_fc  = [128, Num_action]

# Initialize weights and bias
def weight_variable(shape):
    return tf.Variable(xavier_initializer(shape))

def bias_variable(shape):
	return tf.Variable(xavier_initializer(shape))

# Xavier Weights initializer
def xavier_initializer(shape):
	dim_sum = np.sum(shape)
	if len(shape) == 1:
		dim_sum += 1
	bound = np.sqrt(2.0 / dim_sum)
	return tf.random_uniform(shape, minval=-bound, maxval=bound)

# Assigning network variables to target network variables
def assign_network_to_target():
	update_wfc1 = tf.assign(w_fc1_target, w_fc1)
	update_wfc2 = tf.assign(w_fc2_target, w_fc2)
	update_wfc3 = tf.assign(w_fc3_target, w_fc3)
	update_bfc1 = tf.assign(b_fc1_target, b_fc1)
	update_bfc2 = tf.assign(b_fc2_target, b_fc2)
	update_bfc3 = tf.assign(b_fc3_target, b_fc3)

	sess.run(update_wfc1)
	sess.run(update_wfc2)
	sess.run(update_wfc3)
	sess.run(update_bfc1)
	sess.run(update_bfc2)
	sess.run(update_bfc3)

# Input
x = tf.placeholder(tf.float32, shape = [None, 4])

# Densely connect layer variables
w_fc1 = weight_variable(first_fc)
b_fc1 = bias_variable([first_fc[1]])

w_fc2 = weight_variable(second_fc)
b_fc2 = bias_variable([second_fc[1]])

w_fc3 = weight_variable(third_fc)
b_fc3 = bias_variable([third_fc[1]])

h_fc1 = tf.nn.relu(tf.matmul(x, w_fc1)+b_fc1)
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2)+b_fc2)

output = tf.matmul(h_fc2, w_fc3) + b_fc3


# Densely connect layer variables target
w_fc1_target = weight_variable(first_fc)
b_fc1_target = bias_variable([first_fc[1]])

w_fc2_target = weight_variable(second_fc)
b_fc2_target = bias_variable([second_fc[1]])

w_fc3_target = weight_variable(third_fc)
b_fc3_target = bias_variable([third_fc[1]])

h_fc1_target = tf.nn.relu(tf.matmul(x, w_fc1_target)+b_fc1_target)
h_fc2_target = tf.nn.relu(tf.matmul(h_fc1_target, w_fc2_target)+b_fc2_target)

output_target = tf.matmul(h_fc2_target, w_fc3_target) + b_fc3_target

# Loss function and Train
action_target = tf.placeholder(tf.float32, shape = [None, Num_action])
y_prediction = tf.placeholder(tf.float32, shape = [None])

y_target = tf.reduce_sum(tf.multiply(output, action_target), reduction_indices = 1)
Loss = tf.reduce_mean(tf.square(y_prediction - y_target))
train_step = tf.train.AdamOptimizer(Learning_rate).minimize(Loss)

# Initialize variables
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.InteractiveSession(config=config)
init = tf.global_variables_initializer()
sess.run(init)

# Initial parameters
Replay_memory = []
step = 1
score = 0
plot_y_loss = []
plot_y_maxQ = []
loss_list = []
maxQ_list = []
episode = 0

data_time = str(datetime.date.today()) + '_' + str(datetime.datetime.now().hour) + '_' + str(datetime.datetime.now().minute)

observation = env.reset()
action = env.action_space.sample()
observation, reward, terminal, info = env.step(action)

# Figure and figure data setting
# plt.figure(1)
plot_x = []
plot_y = []

f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

# Making replay memory
while True:
    # Rendering
    env.render()

    if step <= Num_start_training:
        state = 'Observing'
        action = np.zeros([Num_action])
        action[random.randint(0, Num_action - 1)] = 1.0
        action_step = np.argmax(action)

        observation_next, reward, terminal, info = env.step(action_step)
        reward -= 5 * abs(observation_next[0])

        Replay_memory.append([observation, action, reward, observation_next, terminal])

        if step % 10 == 0:
            print('step: ' + str(step) + ' / '  + 'state: ' + state)

        episode_step = 0

    elif step <= Num_start_training + Num_training:
        # Training
        state = 'Training'

        if len(Replay_memory) > Num_replay_memory:
            del Replay_memory[0]

        # if random value(0 - 1) is smaller than Epsilon, action is random. Otherwise, action is the one which has the largest Q value
        if random.random() < Epsilon:
            action = np.zeros([Num_action])
            action[random.randint(0, Num_action - 1)] = 1.0
            action_step = np.argmax(action)

        else:
            observation_feed = np.reshape(observation, (1,4))
            Q_value = output.eval(feed_dict={x: observation_feed})[0]
            action = np.zeros([Num_action])
            action[np.argmax(Q_value)] = 1
            action_step = np.argmax(action)

        observation_next, reward, terminal, info = env.step(action_step)
        reward -= 5 * abs(observation_next[0])

        # Save experience to the Replay memory
        ###################################### Multi-step ######################################
        state_list.append(observation)
        reward_list.append(reward)

        if episode_step > n_step:
            del state_list[0]
            del reward_list[0]

        if terminal:
            for i in range(n_step):
                Replay_memory.append([state_list[i], action, sum(reward_list[i: n_step]), observation_next, terminal])
        else:
            if episode_step >= n_step:
                Replay_memory.append([state_list[0], action, sum(reward_list), observation_next, terminal])
        ########################################################################################

        minibatch =  random.sample(Replay_memory, Num_batch)

        # Save the each batch data
        observation_batch      = [batch[0] for batch in minibatch]
        action_batch           = [batch[1] for batch in minibatch]
        reward_batch           = [batch[2] for batch in minibatch]
        observation_next_batch = [batch[3] for batch in minibatch]
        terminal_batch 	       = [batch[4] for batch in minibatch]

        y_batch = []

        # Update target network according to the Num_update value
        if step % Num_update == 0:
            assign_network_to_target()

        # Get y_prediction
        Q_batch = output_target.eval(feed_dict = {x: observation_next_batch})
        for i in range(len(minibatch)):
            if terminal_batch[i] == True:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + Gamma * np.max(Q_batch[i]))

        train_step.run(feed_dict = {action_target: action_batch, y_prediction: y_batch, x: observation_batch})
        loss = Loss.eval(feed_dict = {action_target: action_batch, y_prediction: y_batch, x: observation_batch})

        loss_list.append(loss)
        maxQ_list.append(np.max(Q_batch))

        # Reduce epsilon at training mode
        if Epsilon > Final_epsilon:
            Epsilon -= 1.0/Num_training

    elif step < Num_start_training + Num_training + Num_testing:
        # Testing
        state = 'Testing'
        observation_feed = np.reshape(observation, (1,4))
        Q_value = output.eval(feed_dict={x: observation_feed})[0]

        action = np.zeros([Num_action])
        action[np.argmax(Q_value)] = 1
        action_step = np.argmax(action)

        observation_next, reward, terminal, info = env.step(action_step)

        Epsilon = 0

    else:
        # Test is finished
        print('Test is finished!!')
        plt.savefig('./Plot/' + data_time + '_' + algorithm + '_' + game_name + '.png')
        break

    # Update parameters at every iteration
    step += 1
    episode_step += 1
    score += reward

    observation = observation_next

    # Plot average score
    if len(plot_x) % Num_episode_plot == 0 and len(plot_x) != 0 and state != 'Observing':
        ax1.plot(np.average(plot_x), np.average(plot_y_loss), '*')
        ax1.set_title('Mean Loss')
        ax1.set_ylabel('Mean Loss')
        ax1.hold(True)

        ax2.plot(np.average(plot_x), np.average(plot_y),'*')
        ax2.set_title('Mean score')
        ax2.set_ylabel('Mean score')
        ax2.hold(True)

        ax3.plot(np.average(plot_x), np.average(plot_y_maxQ),'*')
        ax3.set_title('Mean Max Q')
        ax3.set_ylabel('Mean Max Q')
        ax3.set_xlabel('Episode')
        ax3.hold(True)

        plt.draw()
        plt.pause(0.000001)

        plot_x = []
        plot_y = []
        plot_y_loss = []
        plot_y_maxQ = []


    # Terminal
    if terminal == True:
        print('step: ' + str(step) + ' / '  + 'state: ' + state  + ' / '  + 'epsilon: ' + str(Epsilon) + ' / '  + 'score: ' + str(score))

        if state != 'Observing':
            # data for plotting
            plot_x.append(episode)
            plot_y.append(score)
            plot_y_loss.append(np.mean(loss_list))
            plot_y_maxQ.append(np.mean(maxQ_list))

        score = 0
        loss_list = []
        maxQ_list = []
        episode += 1
        episode_step = 1

        observation = env.reset()
