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
algorithm = 'Categorical_DQN'

# Parameter setting
Num_action = 2
Gamma = 0.99
Learning_rate = 0.00025
Epsilon = 1
Final_epsilon = 0.1

Num_replay_memory = 10000
Num_start_training = 5000
Num_training = 30000
Num_testing  = 10000
Num_update = 300
Num_batch = 32
Num_episode_plot = 20

# Categorical Parameters
Num_atom = 51
V_min = -10
V_max = 10
delta_z = (V_max - V_min) / (Num_atom - 1)

# Network Parameters
first_fc  = [4, 512]
second_fc = [512, 128]
third_fc  = [128, Num_action * Num_atom]

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

# Set z
z = tf.reshape ( tf.range(V_min, V_max + delta_z, delta_z), [1, Num_atom])

# Densely connect layer variables
w_fc1 = weight_variable(first_fc)
b_fc1 = bias_variable([first_fc[1]])

w_fc2 = weight_variable(second_fc)
b_fc2 = bias_variable([second_fc[1]])

w_fc3 = weight_variable(third_fc)
b_fc3 = bias_variable([third_fc[1]])

h_fc1 = tf.nn.relu(tf.matmul(x, w_fc1)+b_fc1)
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2)+b_fc2)

# Get Q value for each action
logits = tf.matmul(h_fc2, w_fc3) + b_fc3
logits_reshape = tf.reshape(logits, [-1, Num_action, Num_atom])
p_action = tf.nn.softmax(logits_reshape)
z_action = tf.tile(z, [tf.shape(logits_reshape)[0] * tf.shape(logits_reshape)[1], 1])
z_action = tf.reshape(z_action, [-1, Num_action, Num_atom])
Q_action = tf.reduce_sum(tf.multiply(z_action, p_action), axis = 2)

# Densely connect layer variables target
w_fc1_target = weight_variable(first_fc)
b_fc1_target = bias_variable([first_fc[1]])

w_fc2_target = weight_variable(second_fc)
b_fc2_target = bias_variable([second_fc[1]])

w_fc3_target = weight_variable(third_fc)
b_fc3_target = bias_variable([third_fc[1]])

h_fc1_target = tf.nn.relu(tf.matmul(x, w_fc1_target)+b_fc1_target)
h_fc2_target = tf.nn.relu(tf.matmul(h_fc1_target, w_fc2_target)+b_fc2_target)

logits_target = tf.matmul(h_fc2_target, w_fc3_target) + b_fc3_target
logits_reshape_target = tf.reshape(logits_target, [-1, Num_action, Num_atom])
p_action_target = tf.nn.softmax(logits_reshape_target)
z_action_target = tf.tile(z, [tf.shape(logits_reshape_target)[0] * tf.shape(logits_reshape_target)[1], 1])
z_action_target = tf.reshape(z_action_target, [-1, Num_action, Num_atom])
Q_action_target = tf.reduce_sum(tf.multiply(z_action_target, p_action_target), axis = 2)

# Loss function and Train
m_loss = tf.placeholder(tf.float32, shape = [Num_batch, Num_atom])
action_binary_loss = tf.placeholder(tf.float32, shape = [None, Num_action * Num_atom])
logit_valid_loss = tf.multiply(logits, action_binary_loss)

diagonal = tf.ones([Num_atom])
diag = tf.diag(diagonal)
diag = tf.tile(diag, [Num_action, 1])

logit_final_loss = tf.matmul(logit_valid_loss, diag)
p_loss = tf.nn.softmax(logit_final_loss)

p_loss_log = tf.log(p_loss)

Loss = - tf.reduce_mean(tf.reduce_sum(tf.multiply(m_loss, tf.log(p_loss)), axis = 1))

train_step = tf.train.AdamOptimizer(Learning_rate, epsilon = 1e-2 / Num_batch).minimize(Loss)
# train_step = tf.train.AdamOptimizer(Learning_rate).minimize(Loss)

# Initialize variables
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3

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

    elif step <= Num_start_training + Num_training:
    	# Training
        state = 'Training'

        # if random value(0 - 1) is smaller than Epsilon, action is random. Otherwise, action is the one which has the largest Q value
        if random.random() < Epsilon:
        	action = np.zeros([Num_action])
        	action[random.randint(0, Num_action - 1)] = 1.0
        	action_step = np.argmax(action)

        else:
        	Q_value = Q_action.eval(feed_dict = {x: [observation]})
        	action = np.zeros([Num_action])
        	action[np.argmax(Q_value)] = 1
        	action_step = np.argmax(action)

        observation_next, reward, terminal, info = env.step(action_step)
        reward -= 5 * abs(observation_next[0])

        # Select Minibatch
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

        # Training
        Q_batch = Q_action.eval(feed_dict = {x: observation_next_batch})
        p_batch = p_action_target.eval(feed_dict = {x: observation_next_batch})
        z_batch = z.eval()

        m_batch = np.zeros([Num_batch, Num_atom])
        for i in range(len(minibatch)):
            action_max = np.argmax(Q_batch[i, :])
            if terminal_batch[i]:
                for j in range(Num_atom):
                    Tz = reward_batch[i]

                    # Bounding Tz
                    if Tz >= V_max:
                        Tz = V_max
                    elif Tz <= V_min:
                        Tz = V_min

                    b = (Tz - V_min) / delta_z
                    l = np.int32(np.floor(b))
                    u = np.int32(np.ceil(b))

                    m_batch[i, l] += (u - b)
                    m_batch[i, u] += (b - l)
            else:
                for j in range(Num_atom):
                    Tz = reward_batch[i] + Gamma * z_batch[0,j]

                    # Bounding Tz
                    if Tz >= V_max:
                        Tz = V_max
                    elif Tz <= V_min:
                        Tz = V_min

                    b = (Tz - V_min) / delta_z
                    l = np.int32(np.floor(b))
                    u = np.int32(np.ceil(b))

                    m_batch[i, l] += p_batch[i, action_max, j] * (u - b)
                    m_batch[i, u] += p_batch[i, action_max, j] * (b - l)

        action_binary = np.zeros([Num_batch, Num_action * Num_atom])

        for i in range(len(action_batch)):
        	action_batch_max = np.argmax(action_batch[i])
        	action_binary[i, Num_atom * action_batch_max : Num_atom * (action_batch_max + 1)] = 1

        _, loss, p_log, p_test = sess.run([train_step, Loss, p_loss_log, p_loss],
                                  feed_dict = {x:observation_batch, m_loss: m_batch, action_binary_loss: action_binary})

        if np.any(np.isnan(p_log)):
            if np.any(p_test < 0):
                print('jojojojo')
            print('heyhey')
            break

        loss_list.append(loss)
        maxQ_list.append(np.max(Q_batch))

        # Reduce epsilon at training mode
        if Epsilon > Final_epsilon:
        	Epsilon -= 1.0/Num_training

    elif step < Num_start_training + Num_training + Num_testing:
    	# Testing
    	state = 'Testing'

    	Q_value = Q_action.eval(feed_dict = {x: [observation]})
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
    score += reward

    # Save experience to the Replay memory
    Replay_memory.append([observation, action, reward, observation_next, terminal])

    if len(Replay_memory) > Num_replay_memory:
    	del Replay_memory[0]

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

        observation = env.reset()
