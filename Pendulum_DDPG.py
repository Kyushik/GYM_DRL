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
import gym
import time

env = gym.make('Pendulum-v0')
game_name = 'Pendulum'
algorithm = 'DDPG'

# Parameter setting
Num_action = 1
Gamma = 0.99
Learning_rate_actor  = 0.0001
Learning_rate_critic = 0.0005

Num_training = 100000
Num_testing  = 10000

Num_batch = 32
Num_replay_memory = 50000
Num_start_train_episode = 50

Num_episode_plot = 30

first_fc_actor  = [3, 1024]
first_fc_critic  = [4, 1024]
second_fc = [1024, 512]
third_fc_actor  = [512, Num_action]
third_fc_critic = [512, 1]

# Variables for Ornstein-Uhlenbeck Process
mu_UL = 0
x_UL = 0
theta_UL = 0.3
sigma_UL = 0.5

tau = 0.01

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

# Input
x = tf.placeholder(tf.float32, shape = [None, 3])
is_train_actor = tf.placeholder(tf.bool)

# Actor Network
with tf.variable_scope('network_actor'):
    w_fc1_actor = weight_variable(first_fc_actor)
    b_fc1_actor = bias_variable([first_fc_actor[1]])

    w_fc2_actor = weight_variable(second_fc)
    b_fc2_actor = bias_variable([second_fc[1]])

    w_fc3_actor = weight_variable(third_fc_actor)
    b_fc3_actor = bias_variable([third_fc_actor[1]])

h_fc1_actor = tf.nn.relu(tf.matmul(x, w_fc1_actor)+b_fc1_actor)
h_fc2_actor = tf.nn.relu(tf.matmul(h_fc1_actor, w_fc2_actor)+b_fc2_actor)

output_actor = tf.nn.tanh(tf.matmul(h_fc2_actor, w_fc3_actor) + b_fc3_actor)

with tf.variable_scope('target_actor'):
    w_fc1_actor_target = weight_variable(first_fc_actor)
    b_fc1_actor_target = bias_variable([first_fc_actor[1]])

    w_fc2_actor_target = weight_variable(second_fc)
    b_fc2_actor_target = bias_variable([second_fc[1]])

    w_fc3_actor_target = weight_variable(third_fc_actor)
    b_fc3_actor_target = bias_variable([third_fc_actor[1]])

h_fc1_actor_target = tf.nn.relu(tf.matmul(x, w_fc1_actor_target)+b_fc1_actor_target)
h_fc2_actor_target = tf.nn.relu(tf.matmul(h_fc1_actor_target, w_fc2_actor_target)+b_fc2_actor_target)

output_actor_target = tf.nn.tanh(tf.matmul(h_fc2_actor_target, w_fc3_actor_target) + b_fc3_actor_target)

action_in = tf.cond(is_train_actor, lambda: output_actor, lambda: tf.placeholder(tf.float32, shape = [None, 1]))
input = tf.concat([x, action_in], axis = 1)

with tf.variable_scope('network_critic'):
    w_fc1_critic = weight_variable(first_fc_critic)
    b_fc1_critic = bias_variable([first_fc_critic[1]])

    w_fc2_critic = weight_variable(second_fc)
    b_fc2_critic = bias_variable([second_fc[1]])

    w_fc3_critic = weight_variable(third_fc_critic)
    b_fc3_critic = bias_variable([third_fc_critic[1]])

# Critic Network
h_fc1_critic = tf.nn.relu(tf.matmul(input, w_fc1_critic)+b_fc1_critic)
h_fc2_critic = tf.nn.relu(tf.matmul(h_fc1_critic, w_fc2_critic)+b_fc2_critic)

output_critic = tf.matmul(h_fc2_critic, w_fc3_critic) + b_fc3_critic

with tf.variable_scope('target_critic'):
    w_fc1_critic_target = weight_variable(first_fc_critic)
    b_fc1_critic_target = bias_variable([first_fc_critic[1]])

    w_fc2_critic_target = weight_variable(second_fc)
    b_fc2_critic_target = bias_variable([second_fc[1]])

    w_fc3_critic_target = weight_variable(third_fc_critic)
    b_fc3_critic_target = bias_variable([third_fc_critic[1]])

h_fc1_critic_target = tf.nn.relu(tf.matmul(input, w_fc1_critic_target)+b_fc1_critic_target)
h_fc2_critic_target = tf.nn.relu(tf.matmul(h_fc1_critic_target, w_fc2_critic_target)+b_fc2_critic_target)

output_critic_target = tf.matmul(h_fc2_critic_target, w_fc3_critic_target) + b_fc3_critic_target

# Set Loss
target_critic = tf.placeholder(tf.float32, shape = [None])

Loss_actor = -tf.reduce_mean(output_critic)
Loss_critic = tf.reduce_mean(tf.square(target_critic - output_critic))

# Get trainable variables
trainable_variables = tf.trainable_variables()

# network variables
trainable_variables_actor = [var for var in trainable_variables if var.name.startswith('network_actor')]
trainable_variables_critic = [var for var in trainable_variables if var.name.startswith('network_critic')]

opt_actor  = tf.train.AdamOptimizer(learning_rate = Learning_rate_actor)
opt_critic = tf.train.AdamOptimizer(learning_rate = Learning_rate_critic)

train_critic = opt_critic.minimize(Loss_critic, var_list = trainable_variables_critic)
train_actor  = opt_actor.minimize(Loss_actor, var_list = trainable_variables_actor)

# Initialize variables
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.InteractiveSession(config=config)
init = tf.global_variables_initializer()
sess.run(init)

# Initial parameters
step = 1
score = 0
episode = 0

data_time = str(datetime.date.today()) + '_' + str(datetime.datetime.now().hour) + '_' + str(datetime.datetime.now().minute)

state = env.reset()
action = env.action_space.sample()
state, reward, terminal, info = env.step(action)

replay_memory = []

# Figure and figure data setting
plot_x = []
plot_y = []
plot_loss = []
plot_Q = []

step_old = 10000

# Get trainable variables
trainable_variables = tf.trainable_variables()
# network variables
trainable_vars_network  = [var for var in trainable_variables if var.name.startswith('network')]

# target variables
trainable_vars_target  = [var for var in trainable_variables if var.name.startswith('target')]

# Making replay memory
while True:
    # Rendering
    env.render()

    if step <= Num_training:
        # Training
        progress = 'Training'

        state_feed = np.reshape(state, (1,3))

        # Do Orstein-Uhlenbeck Process
        action_actor = 2 * output_actor.eval(feed_dict={x: state_feed})
        noise_UL = theta_UL * (mu_UL - action_actor) + sigma_UL * np.random.randn(Num_action)
        critic_test = output_critic.eval(feed_dict={x: state_feed, action_in: action_actor, is_train_actor: False})

        action = action_actor + noise_UL
        np.clip(action, -2.0, 2.0)
        action = np.reshape(action, (Num_action))

        state_next, reward, terminal, info = env.step(action)
        state_next_feed = np.reshape(state_next, (1,3))

        # Experience replay
        if len(replay_memory) >= Num_replay_memory:
        	del replay_memory[0]

        replay_memory.append([state, action, reward, state_next, terminal])

        if episode > Num_start_train_episode:
            # Select minibatch
            minibatch =  random.sample(replay_memory, Num_batch)

            # current_time = time.time()
            # print('action_actor: ' + str(action_actor))
            # print('noise_UL: ' + str(noise_UL))

            # Save the each batch data
            state_batch      = [batch[0] for batch in minibatch]
            action_batch     = [batch[1] for batch in minibatch]
            reward_batch     = [batch[2] for batch in minibatch]
            next_state_batch = [batch[3] for batch in minibatch]
            terminal_batch   = [batch[4] for batch in minibatch]

            output_actor_batch = output_actor.eval(feed_dict = {x: state_batch})
            next_output_actor_batch = output_actor_target.eval(feed_dict = {x: next_state_batch})
            Q_batch = output_critic_target.eval(feed_dict = {x: next_state_batch, action_in: next_output_actor_batch, is_train_actor: False})

            target_batch = []

            # print('time1: ' + str(time.time() - current_time))
            # current_time = time.time()

            for i in range(len(minibatch)):
            	if terminal == True:
            		target_batch.append(reward_batch[i])
            	else:
            		target_batch.append(reward_batch[i] + Gamma * np.max(Q_batch[i]))

            _, loss_critic = sess.run([train_critic, Loss_critic], feed_dict = {target_critic: target_batch, x: state_batch, action_in: output_actor_batch, is_train_actor: False})
            train_actor.run(feed_dict = {x: state_batch, is_train_actor: True})

            # print('time2: ' + str(time.time() - current_time))
            # current_time = time.time()

            # Update Target Network

            plot_loss.append(loss_critic)
            plot_Q.append(np.mean(Q_batch))

            if step % (Num_training / 100) == 0:
                for i in range(len(trainable_vars_network)):
                    sess.run(tf.assign(trainable_vars_target[i], trainable_vars_network[i]))

                # sess.run(tf.assign(trainable_vars_target[i], tau * trainable_vars_network[i] + (1-tau) * trainable_vars_target[i]))
                #
                # sess.run(tf.assign(trainable_vars_tar_critic[i],
                #                         tau * trainable_vars_net_critic[i] + (1-tau) * trainable_vars_tar_critic[i]))

            # print('time3: ' + str(time.time() - current_time))

    elif step < Num_training + Num_testing:
        # Testing
        progress = 'Testing'

        action_actor = 2 * output_actor.eval(feed_dict={x: state_feed})
        action = action_actor

        np.clip(action, -2.0, 2.0)

        state_next, reward, terminal, info = env.step(action)
    else:
    	# Training is finished
    	print('Training is finished!!')
    	plt.savefig('./Plot/' + data_time + '_' + algorithm + '_' + game_name + '.png')
    	break

    # Update parameters at every iteration
    step += 1
    score += reward

    state = state_next

    # Plot average score
    if len(plot_x) % Num_episode_plot == 0 and len(plot_x) != 0 and progress != 'Observing':
        plt.figure(1)
    	plt.xlabel('Episode')
    	plt.ylabel('Score')
    	plt.title('Inverted Pendulum ' + algorithm)
    	plt.grid(True)

    	plt.plot(np.average(plot_x), np.average(plot_y), hold = True, marker = '*', ms = 5)

    	plt.draw()
    	plt.pause(0.000001)

        plt.figure(2)
    	plt.xlabel('Episode')
    	plt.ylabel('Loss')
    	plt.title('Inverted Pendulum ' + algorithm)
    	plt.grid(True)
        plt.plot(np.average(plot_x), np.average(plot_loss), hold = True, marker = 'o', ms = 5)

    	plt.draw()
    	plt.pause(0.000001)

        plt.figure(3)
    	plt.xlabel('Episode')
    	plt.ylabel('Q-value')
    	plt.title('Inverted Pendulum ' + algorithm)
    	plt.grid(True)
        plt.plot(np.average(plot_x), np.average(plot_Q), hold = True, marker = 'd', ms = 5)

    	plt.draw()
    	plt.pause(0.000001)

    	plot_x = []
    	plot_y = []
        plot_loss = []
        plot_Q = []

    # Terminal
    if terminal == True:
    	print('step: ' + str(step) + ' / '  + 'episode: ' + str(episode) + ' / '  + 'state: ' + progress  + ' / '  + 'score: ' + str(score))

    	if progress != 'Observing':
    		# data for plotting
    		plot_x.append(episode)
    		plot_y.append(score)

    	score = 0
    	episode += 1

    	state = env.reset()
