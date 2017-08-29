# Cartpole 
# State  -> x, x_dot, theta, theta_dot
# Action -> force (+1, -1)

# Actor Critic Version 2
# - Use Experience Replay

# Import modules 
import tensorflow as tf 
import random
import numpy as np 
import copy 
import matplotlib.pyplot as plt 
import datetime 
import gym

env = gym.make('CartPole-v0')
game_name = 'CartPole'
algorithm = 'Actor_Critic_v2'

# Parameter setting 
Num_action = 2
Gamma = 0.99
Learning_rate_actor  = 0.0001 
Learning_rate_critic = 0.0005

Num_replay_memory = 10000
Num_start_training = 5000
Num_training = 20000
Num_testing  = 10000 

Num_update = 200
Num_batch = 32
Num_episode_plot = 30

first_fc  = [4, 256]
second_fc = [256, 128]
third_fc_actor  = [128, Num_action]
third_fc_critic = [128, 1]

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
x = tf.placeholder(tf.float32, shape = [None, 4])

# Actor Network 
w_fc1_actor = weight_variable(first_fc)
b_fc1_actor = bias_variable([first_fc[1]])

w_fc2_actor = weight_variable(second_fc)
b_fc2_actor = bias_variable([second_fc[1]])

w_fc3_actor = weight_variable(third_fc_actor)
b_fc3_actor = bias_variable([third_fc_actor[1]])

h_fc1_actor = tf.nn.relu(tf.matmul(x, w_fc1_actor)+b_fc1_actor)
h_fc2_actor = tf.nn.relu(tf.matmul(h_fc1_actor, w_fc2_actor)+b_fc2_actor)

output_actor = tf.nn.softmax(tf.matmul(h_fc2_actor, w_fc3_actor) + b_fc3_actor)

# Critic Network
w_fc1_critic = weight_variable(first_fc)
b_fc1_critic = bias_variable([first_fc[1]])

w_fc2_critic = weight_variable(second_fc)
b_fc2_critic = bias_variable([second_fc[1]])

w_fc3_critic = weight_variable(third_fc_critic)
b_fc3_critic = bias_variable([third_fc_critic[1]])

h_fc1_critic = tf.nn.relu(tf.matmul(x, w_fc1_critic)+b_fc1_critic)
h_fc2_critic = tf.nn.relu(tf.matmul(h_fc1_critic, w_fc2_critic)+b_fc2_critic)

output_critic = tf.matmul(h_fc2_critic, w_fc3_critic) + b_fc3_critic

# Loss function and Train (Actor)
action_actor = tf.placeholder(tf.float32, shape = [None, Num_action])
advantage_actor = tf.placeholder(tf.float32, shape = [None, 1])

action_prob = tf.reduce_sum(tf.multiply(action_actor, output_actor))
cross_entropy = tf.multiply(tf.log(action_prob), advantage_actor) 
Loss_actor = - tf.reduce_sum(cross_entropy)

train_actor = tf.train.AdamOptimizer(Learning_rate_actor).minimize(Loss_actor)

# Loss function and Train (Critic)
target_critic = tf.placeholder(tf.float32, shape = [None, 1])

Loss_critic = tf.reduce_mean(tf.square(target_critic - output_critic))
train_critic = tf.train.AdamOptimizer(Learning_rate_critic).minimize(Loss_critic)

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
state_next, reward, terminal, info = env.step(action)

# Figure and figure data setting
plt.figure(1)
plot_x = []
plot_y = []

Replay_memory = []

while True:
	# Rendering
	env.render()
	
	# Making replay memory	
	if step <= Num_start_training:
		progress = 'Observing'

		state_feed = np.reshape(state, (1,4))

		action_step = random.randint(0, Num_action - 1)
		action = np.zeros([Num_action])
		action[action_step] = 1

		state_next, reward, terminal, info = env.step(action_step)
		
		Replay_memory.append([state, action, reward, state_next, terminal])

	elif step <= Num_start_training + Num_training:
		# Training 
		progress = 'Training'

		if len(Replay_memory) > Num_replay_memory:
			del Replay_memory[0]

		state_feed = np.reshape(state, (1,4))

		Policy = output_actor.eval(feed_dict={x: state_feed}).flatten()
		action_step = np.random.choice(Num_action, 1, p = Policy)[0]
		action = np.zeros([Num_action])
		action[action_step] = 1

		state_next, reward, terminal, info = env.step(action_step)
		state_next_feed = np.reshape(state_next, (1,4))

		if terminal == True and score < 190:
			reward -= 10
    			
		# reward -= 5 * abs(state_next[0])

		Replay_memory.append([state, action, reward, state_next, terminal])
	
		minibatch =  random.sample(Replay_memory, Num_batch)

		# Save the each batch data 
		state_batch      = [batch[0] for batch in minibatch]
		action_batch     = [batch[1] for batch in minibatch]
		reward_batch     = [batch[2] for batch in minibatch]
		state_next_batch = [batch[3] for batch in minibatch]
		terminal_batch 	 = [batch[4] for batch in minibatch]

		# Get values
		value_batch = output_critic.eval(feed_dict = {x: state_batch})
		value_next_batch = output_critic.eval(feed_dict = {x: state_next_batch})

		# Training
		advantage = []
		target = []

		for i in range(Num_batch):
			if terminal_batch[i] == True:
				advantage.append(reward - value_batch[i])
				target.append([reward])
			else:
				advantage.append((reward_batch[i] + Gamma * value_next_batch[i]) - value_batch[i])
				target.append(reward_batch[i] + Gamma * value_next_batch[i])
		
		train_actor.run(feed_dict = {action_actor: action_batch, advantage_actor: advantage, x: state_batch})
		train_critic.run(feed_dict = {target_critic: target, x: state_batch})

	elif step < Num_start_training + Num_training + Num_testing:
    		# Testing
		progress = 'Testing'

		state_feed = np.reshape(state, (1, 4))
		Policy = output_actor.eval(feed_dict={x: state_feed})[0]
		action_step = np.random.choice(Num_action, 1, p = Policy)[0]
		action = np.zeros([1, Num_action])
		action[0, action_step] = 1
		
		state_next, reward, terminal, info = env.step(action_step)

		if terminal == True and score < 190:
			reward -= 10

		# reward -= 5 * abs(state_next[0,0])

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
		plt.xlabel('Episode')
		plt.ylabel('Score')
		plt.title('Cartpole' + algorithm)
		plt.grid(True)

		plt.plot(np.average(plot_x), np.average(plot_y), hold = True, marker = '*', ms = 5)
		plt.draw()
		plt.pause(0.000001)

		plot_x = []
		plot_y = [] 

	# Terminal
	if terminal == True:
		print('step: ' + str(step) + ' / '  + 'episode: ' + str(episode) + ' / '  + 'state: ' + progress  + ' / '  + 'score: ' + str(score)) 

		if progress != 'Observing':
			# data for plotting
			plot_x.append(episode)
			plot_y.append(score)
			episode += 1
		
		score = 0
		
		state = env.reset()

