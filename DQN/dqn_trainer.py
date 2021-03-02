"""Module containing the classes required for the training of a DQN to solve an environment."""

import math
import random
from collections import deque

import numpy as np
import tensorflow as tf


class Network:
    """Represents a NN used to predict Q-values in the DQN algorithm."""

    def __init__(self, input_shape, output_dim, hidden_architecture=None):
        """Set up the desired network architecture in tensorflow and initialise.

        The output layer is always fully-connected with a linear activation fn.
        The form of other layers can be specified through the hidden architecture argument.
        'conv2d' layers are relevant when the state is an image. In this case, the network
        should resemble something similar to input -> conv2d -> conv2d -> relu -> output .
        All conv2d layers must exist before any fully-connected layers, as the latter flatten
        the feature vector.

        Args:
            input_dim (int): Number of input layer units. This should be the dimension of
                             the state vector in DQN.
            output_dim (int): Number of output layer units. In DQN, this is the number of
                              actions in the discrete action space.
            hidden_architecture (tuple of lists in the form [type, params]):
                type (str): Type of layer to be used ('linear', 'relu' or 'conv2d').
                params (int or tuple):
                    Necessary params to define the layer. For fully connected layers
                    ('linear' or 'relu'), this is just the number of units in the layer (int).
                    For 'conv2d', this should be a tuple. The elements of this tuple will be
                    passed directly as arguments to tf.contrib.layers.conv2d through
                    tf.contrib.layers.conv2d(x, *params).

        Returns:
            None

        Example:
            Network(64, 4, (["relu", 32],["linear", 16])) creates a network which expects
            an input vector with 64 features and will output a vector with 4 entries for a given
            input vector. In addition to an input layer with 64 units and an output layer with
            4 units, there are 2 hidden layers with 32 and 16 units respectively. The activation
            function linking the input layer and first hidden layer is ReLU, that linking the
            two hidden layers is linear, and that linking the final hidden layer and the output
            layer is linear.

        """
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.hidden_architecture = hidden_architecture

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.state = tf.placeholder(tf.float32, shape=[None, *self.input_shape], name="state")
            self.output = self._feedforward(self.state)

            with tf.variable_scope("training"):
                self.action = tf.placeholder(tf.int32, shape=[None], name="action")
                example_num = tf.range(tf.size(self.action), dtype=tf.int32)
                indices = tf.stack([example_num, self.action], axis=1)
                self.q_sa = tf.gather_nd(self.output, indices)
                self.target = tf.placeholder(tf.float32, shape=[None], name="target")
                loss = tf.losses.mean_squared_error(self.target, self.q_sa)
                learning_rate = 0.00025
                self.trainer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

            with tf.variable_scope("saving"):
                self.saver = tf.train.Saver(max_to_keep=100)

            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

    def train_on_batch(self, targets, batch):
        """Train the network on a batch of experiences.

        An experience is a tuple in the form (s,a,r,s_,done). batch should be a
        numpy array containing N experiences stacked along axis 0.

        Args:
            targets (numpy array): N entries, with targets[i] as the target value
                                   for experience i.
            batch (numpy array): N entries along axis 0, each of which is an
                                 experience tuple (s,a,r,s_,done).

        Returns:
            None

        """
        states = [exp[0] for exp in batch]
        actions = [exp[1] for exp in batch]
        self.sess.run(self.trainer, feed_dict={self.state: states,
                                               self.action: actions,
                                               self.target: targets})

    def assign_weights(self, assign_net):
        """Set the values of this network's weights to the value of assign_net's weights.

        Args:
            assign_net (Network): The values of assign_net's weights will be assigned to
                                  the self network's weights.

        Returns:
            None

        """
        with assign_net.graph.as_default():
            assign_net_weights = tf.trainable_variables()
        with self.graph.as_default():
            self_weights = tf.trainable_variables()
            ops = []
            for i, var in enumerate(assign_net_weights):
                ops.append(tf.assign(self_weights[i], assign_net.sess.run(var)))
            self.sess.run(ops)

    def save(self, model_files_path, num_steps):
        """Save the tensorflow model to files.

        Args:
            model_files_path (str): Destination for model files.
            num_steps (int): Number of training steps elapsed.

        Returns:
            None

        Example:
            save("./q_model", 1000) will save files in the form q_model-1000.* within
            the current working directory.

        """
        print("Saving model as {}-{} within the current directory.".format(model_files_path, num_steps))
        self.saver.save(self.sess, model_files_path, global_step=num_steps)

    def predict(self, s):
        """Return estimated Q(s,a) for each action a from single state s.

        s should be a single state. The function will then return a 1-D numpy array
        with self.output_dim entries. These are such that q_values[i] is
        Q(s,i).

        Args:
            s (numpy array): A single state.

        Returns:
            q_values (numpy array): Contains a Q-value for each action from state s.

        """
        q_values = self.sess.run(self.output, feed_dict={self.state: [s]})[0]
        return q_values

    def predict_batch(self, states):
        """Return estimated Q(s,a) for each s within states and valid a.

        Args:
            states (numpy array): N state vectors stacked along axis 0.

        Returns:
            q_values (numpy array): N predictions stacked along axis 0. Prediction
                                    i is Q(states[i],a) for each valid a.

        """
        q_values = self.sess.run(self.output, feed_dict={self.state: states})
        return q_values

    def _feedforward(self, x):
        """Implement the network's achitecture and return the output tensor based on input x."""
        # If hidden architecture not specified, use a linear network with no hidden layer.
        if not self.hidden_architecture:
            if len(x.shape) > 2:
                x = tf.contrib.layers.flatten(x)
            return tf.contrib.layers.fully_connected(x, self.output_dim, activation_fn=None,
                                                     scope="output")

        y = x
        for i, (type, params) in enumerate(self.hidden_architecture):
            if type == "linear":
                if len(y.shape) > 2:
                    y = tf.contrib.layers.flatten(y)
                y = tf.contrib.layers.fully_connected(y, params, activation_fn=None,
                                                      scope="linear{}".format(i+1))
            elif type == "relu":
                if len(y.shape) > 2:
                    y = tf.contrib.layers.flatten(y)
                y = tf.contrib.layers.fully_connected(y, params, activation_fn=tf.nn.relu,
                                                      scope="relu{}".format(i+1))
            elif type == "conv2d":
                y = tf.contrib.layers.conv2d(y, *params, scope="conv{}".format(i+1))
            else:
                raise ValueError("Invalid layer type, '{}', passed to dqn_trainer.Network(). "
                                 "Supported layer types are 'linear', 'relu' and 'conv2d'.".format(type))

        # Final layer, always linear.
        if len(y.shape) > 2:
            y = tf.contrib.layers.flatten(y)
        y = tf.contrib.layers.fully_connected(y, self.output_dim, activation_fn=None,
                                              scope="output")
        return y


class ReplayMemory:
    """A class to implement non-prioritised experience replay memory.

    Entries are in the form (state, action, reward, next_state, done) and
    are stored within a deque with most recent entries at the end.

    """

    def __init__(self, capacity):
        """Initialise the replay memory.

        Args:
            capacity (int): Maximum number of experiences that can be stored in the replay memory.

        Returns:
            None

        """
        self.capacity = capacity # Maximum number of entries stored
        self.samples = deque()

    def populate(self, env, preprocessor=None, n=None):
        """Populate replay memory with n experiences generated by executing a random policy in environment env.

        Args:
            env (OpenAI Gym environment): Created using e.g. gym.make("CartPole-v0"). The environment should
                                          have a discrete action space.
            preprocessor (Preprocessor): A Preprocessor object whose process() function should be applied to
                                         a state before storing in replay memory.
            n (int): Number of experiences to populate the replay memory with. Should satisfy 0 <= n <= self.capacity.
                     self.capacity experiences are added if n is None.

        Returns:
            None

        """
        if n is None:
            n = self.capacity
        if n > self.capacity:
            raise ValueError("Replay memory cannot be populated with "
                             "a number of samples exceeding its capacity.\n"
                             "Attemped to populate a ReplayMemory object with "
                             "capacity {} with {} experiences".format(self.capacity, n))

        s = env.reset()
        if preprocessor is not None:
            s = preprocessor.process(s)
        for i in range(n):
            a = np.random.randint(env.action_space.n)
            s_, r, done, info = env.step(a)
            if preprocessor is not None:
                s_ = preprocessor.process(s_)
            self.add((s,a,r,s_,done))
            if done:
                s = env.reset()
                if preprocessor is not None:
                    s = preprocessor.process(s)
            else:
                s = s_
            if (i+1) % 10000 == 0:
                print("Added experience {}/{}".format(i+1, n))



    def add(self, experience):
        """Add an experience to the replay memory.

        Args:
            experience (tuple): In the form (state,action,reward,next_state,done).

        Returns:
            None

        """
        self.samples.append(experience)

        if len(self.samples) > self.capacity:
            self.samples.popleft()

    def sample(self, n):
        """Return min(n, num_samples) experiences from the replay memory.

        The probability that a given experience is sampled is proportional to the
        priority associated with that experience.

        Args:
            n (int): The number of experiences to sample.

        Returns:
            sample (numpy array): Sampled (s,a,r,s_,done) experiences.

        """
        if n >= len(self.samples):
            return self.samples
        tmp = random.sample(list(self.samples), n)
        samples = np.array(tmp, dtype=object)
        return samples


class SumTree:
    """Sum tree data structure to enable an efficient prioritised experience replay implementation."""

    def __init__(self, num_leaves):
        """Initialise the tree structure.

        Args:
            num_leaves (int): Number of lowest-level nodes in the tree. This should be the number of
                              priorities to be stored in total, i.e. the capacity of the associated
                              ReplayMemory object.

        Returns:
            None

        """
        # NOTE: Layers are counted from 0, so a depth of 4 implies 5 layers in total.
        self.num_leaves = num_leaves
        self.depth = math.ceil(math.log(num_leaves, 2))
        self.num_nodes = 2**(self.depth) - 1 + num_leaves
        self.tree = np.zeros(self.num_nodes)

    def add(self, leaf_idx, p):
        """Assign the value stored in the leaf node corresponding to leaf_idx to p.

        Each leaf in the tree is labelled by an index from 0 to self.num_leaves - 1.
        leaf_idx specifies which leaf should be used. Priorities are typically added starting
        with leaf_idx=0 and incrementing leaf_idx by 1 with each call. After add() has been
        called with leaf_idx=self.num_leaves-1, one should then again begin from 0 to overwrite the
        previous priorities, starting with the oldest examples.

        Args:
            leaf_idx (int): The leaf index of the leaf node whose value should be assigned to p.
                            Should satisfy 0 <= leaf_idx < self.num_leaves.
            p (float): The value to store within the appropriate leaf node.

        Returns:
            None

        """
        tree_idx = self._get_tree_idx(leaf_idx)
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        self._propagate(tree_idx, change)

    def get(self, s):
        """Return the smallest leaf index, i, for which sum(leaf nodes up to and including i) > s.

        The motivation behind this method is that if it is called repeatedly with uniformly random numbers
        between 0 and sum(all leaf node values), then the indices returned will follow a probability
        distribution where the probability of index i being returned is
        (the value in leaf node i) / sum(all leaf node values).

        Args:
            s (float): As in line 1 of docstring. Should satisfy 0 <= s < sum(all leaf node values).

        Returns:
            i (int): As in line 1 of docstring.

        """
        idx = self._retrieve(0, s)
        return self._get_leaf_idx(idx)

    def _propagate(self, tree_idx, change):
        """Propagate the change in an entry at tree_idx up the tree to maintain the sum tree property."""
        parent = (tree_idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """Return the smallest tree index, i, of the leaf node for which sum(leaf nodes up to and including i) > s."""
        left = 2 * idx + 1
        right = left + 1

        if left >= self.tree.size:
            return idx

        if s < self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def _get_leaf_idx(self, tree_idx):
        """Return leaf index given tree index."""
        return tree_idx - (2**(self.depth)-1)

    def _get_tree_idx(self, leaf_idx):
        """Return tree index given leaf index."""
        return leaf_idx + (2**(self.depth)-1)


class PrioritisedReplayMemory:
    """Implements the replay memory element of prioritised experience replay.

    Class variables:
        alpha (float), eps(float): Priority is calculated with (error + eps)^(alpha).
                                   error here refers to the TD error of a given step.

    """

    alpha = 0.6
    eps = 0.01

    def __init__(self, capacity):
        """Initialise the replay memory.

        Args:
            capacity (int): Maximum number of experiences that can be stored in the replay memory.

        Returns:
            None

        """
        self.capacity = capacity
        self.experiences = np.empty(capacity, dtype=object)
        self.tree = SumTree(capacity)
        self.insertion_idx = 0 # Index of the experience that will be replaced during next add() operation

    def populate(self, env, preprocessor=None, n=None):
        """Populate replay memory with n experiences generated by executing a random policy in environment env.

        Args:
            env (OpenAI Gym environment): Created using e.g. gym.make("CartPole-v0"). The environment should
                                          have a discrete action space.
            preprocessor (Preprocessor): A Preprocessor object whose process() function should be applied to
                                         a state before storing in replay memory.
            n (int): Number of experiences to populate the replay memory with. Should satisfy 0 <= n <= self.capacity.
                     self.capacity experiences are added if n is None.

        Returns:
            None

        """
        if n is None:
            n = self.capacity
        if n > self.capacity:
            raise ValueError("Replay memory cannot be populated with "
                             "a number of samples exceeding its capacity.\n"
                             "Attemped to populate a ReplayMemory object with "
                             "capacity {} with {} experiences".format(self.capacity, n))

        s = env.reset()
        if preprocessor is not None:
            s = preprocessor.process(s)
        for i in range(n):
            self.insertion_idx = i
            a = np.random.randint(env.action_space.n)
            s_, r, done, info = env.step(a)
            if preprocessor is not None:
                s_ = preprocessor.process(s_)
            # Set priority equal to (|reward| + eps)^alpha initially
            p = (abs(r) + self.eps)**self.alpha
            self.tree.add(self.insertion_idx, p)
            self.experiences[self.insertion_idx] = (s,a,r,s_,done)
            if done:
                s = env.reset()
                if preprocessor is not None:
                    s = preprocessor.process(s)
            else:
                s = s_
            if (i+1) % 10000 == 0:
                print("Added experience {}/{}".format(i+1, n))

    def add(self, experience):
        """Add an experience to the replay memory.

        Args:
            experience (tuple): In the form (state,action,reward,next_state,done,priority).

        Returns:
            None

        """
        p = experience[-1]
        self.tree.add(self.insertion_idx, p)
        self.experiences[self.insertion_idx] = experience[:-1]
        self.insertion_idx += 1
        if self.insertion_idx >= self.capacity:
            self.insertion_idx = 0

    def sample(self, n):
        """Sample n (s,a,r,s_,done) experiences from the replay memory.

        The probability that a given experience is sampled is proportional to the
        priority associated with that experience.

        Args:
            n (int): The number of experiences to sample.

        Returns:
            sample (numpy array): n (s,a,r,s_,done) experiences stacked along axis 0.

        """
        return self.experiences[self._get_sample_indices(n)]

    def _get_sample_indices(self, n):
        """Return n indices. These indices index elements within the replay buffer.

        The frequency with which each index is sampled is proportional to the priority
        associated with that index.
        """
        sum_p = self.tree.tree[0]
        indices = np.zeros(n, dtype=int)
        for i in range(n):
            s = np.random.uniform(low=0, high=sum_p)
            indices[i] = self.tree.get(s)
        return indices


class DQNAgent:
    def __init__(self, env, hidden_architecture=(["relu",64],), preprocessor=None):
        """Initialise a DQNAgent to act in environment env.

        Args:
            env (OpenAI Gym environment): Created using e.g. gym.make("CartPole-v0"). The environment should
                                          have a discrete action space.
            hidden_architecture (tuple of lists in the form [type, params]):
                This specifies the architecture of the hidden layers by passing this directly to
                the Network class' __init__ fn. See the Network class __init__ docstring for more info.
                    type (str): Type of layer to be used ('linear', 'relu' or 'conv2d').
                    params (int or tuple):
                        Necessary params to define the layer. For fully connected layers
                        ('linear' or 'relu'), this is just the number of units in the layer (int).
                        For 'conv2d', this should be a tuple. The elements of this tuple will be
                        passed directly as arguments to tf.contrib.layers.conv2d through
                        tf.contrib.layers.conv2d(x, *params).
            preprocessor (Preprocessor): A Preprocessor object whose process() function should be applied to
                                         a state before storing in replay memory or passing to the neural network.

        Returns:
            None

        """
        self.env = env
        self.hidden_architecture = hidden_architecture
        self.preprocessor = preprocessor

        if self.preprocessor is not None:
            self.state_shape = self.preprocessor.process(self.env.reset()).shape
        else:
            self.state_shape = self.env.reset().shape
        self.num_actions = self.env.action_space.n

        self.live_network = Network(input_shape=self.state_shape, output_dim=self.num_actions,
                                    hidden_architecture=self.hidden_architecture)


    def learn(self, model_files_path, tensorboard_logdir, num_training_episodes,
              prioritised_experience_replay=False, double_dqn=True,
              save_frequency=10000, memory_capacity=100000, batch_size=64, max_epsilon=1,
              min_epsilon=0.1, num_annealing_steps=100000, gamma=0.99, update_target_freq=1000):
        """Train the agent to perform in the environment self.env, log training through tensorboard and save tf files.

        This method uses DDQN and Prioritised Experience Replay.

        Args:
            model_files_path (str): Destination for model files, e.g. passing "./q_model" would result
                                    in files of the form q_model-n.* in the current directory.
            tensorboard_logdir (str): Directory to write tensorboard files to. Created if it doesn't exist.
            num_training_episodes (int): Training ends after this number of episodes.
            prioritised_experience_replay (bool): Whether to use prioritised experience replay.
            double_dqn (bool): Whether to use double DQN instead of vanilla DQN.
            save_frequency (int): The model is saved every save_frequency training steps and at the end of training.
            memory_capacity (int): Capacity of the DQNAgent's ReplayMemory.
            batch_size (int): Batch size used during training.
            max_epsilon (float): The maximum and initial value of epsilon used for epsilon-greedy action selection.
            min_epsilon (float): The minimum and final value of epsilon used for epsilon-greedy action selection.
            num_annealing_steps (int): epsilon is annealed linearly from max_epsilon to min_epsilon over
                                       num_annealing_steps training steps.
            gamma (float): Discount factor used in the Bellman backup loss calculation.
            update_target_freq (int): The weights of the target network are updated every update_target_freq
                                      training steps.

        Returns:
            None

        """
        # Assign attributes
        self.model_files_path = model_files_path
        self.tensorboard_logdir = tensorboard_logdir
        self.num_training_episodes = num_training_episodes
        self.prioritised_experience_replay = prioritised_experience_replay
        self.double_dqn = double_dqn
        self.save_frequency = save_frequency
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.num_annealing_steps = num_annealing_steps
        self.gamma = gamma
        self.update_target_freq = update_target_freq

        # Tensorboard setup
        self.writer = tf.summary.FileWriter(tensorboard_logdir)
        self.writer.add_graph(self.live_network.sess.graph)

        # Initialise target network and assign weights to those of the live network
        self.target_network = Network(input_shape=self.state_shape, output_dim=self.num_actions,
                                      hidden_architecture=self.hidden_architecture)
        self._update_target_network()

        # Define and populate the replay memory
        if self.prioritised_experience_replay:
            self.replay_memory = PrioritisedReplayMemory(self.memory_capacity)
        else:
            self.replay_memory = ReplayMemory(self.memory_capacity)
        print("Populating replay memory...")
        self.replay_memory.populate(self.env, self.preprocessor)

        # Main training loop
        self.epsilon = self.max_epsilon
        num_steps = 0
        print("Starting training...")
        for i in range(self.num_training_episodes):
            print("Beginning episode {}/{}".format(i+1, self.num_training_episodes))
            s = self.env.reset()
            if self.preprocessor is not None:
                s = self.preprocessor.process(s)
            s_init = s
            done = False
            episode_reward = 0
            while not done:
                a = self._choose_action(s)
                s_, r, done, info = self.env.step(a)
                if self.preprocessor is not None:
                    s_ = self.preprocessor.process(s_)
                if self.prioritised_experience_replay:
                    priority = self._calc_priority(s, a, r, s_, done)
                    experience = (s,a,r,s_,done,priority)
                else:
                    experience = (s,a,r,s_,done)
                self.replay_memory.add(experience)
                batch = self.replay_memory.sample(self.batch_size)
                targets = self._calc_targets(batch)
                self.live_network.train_on_batch(targets, batch)
                num_steps += 1
                if num_steps % self.update_target_freq == 0:
                    self._update_target_network()
                if num_steps % self.save_frequency == 0:
                    self.live_network.save(self.model_files_path, num_steps)
                self._anneal_epsilon(num_steps)
                self._write_epsilon_summary(num_steps)
                q_val = np.max(self.live_network.predict(s_init))
                self._write_q_summary(q_val, num_steps)
                s = s_
                episode_reward += r
            print("Episode {} reward: {}".format(i+1, episode_reward))
            self._write_reward_summary(episode_reward, num_steps)
        self.live_network.save(self.model_files_path, num_steps)

    def act(self, s):
        """Given current raw state s, act greedily with respect to the agent's estimated Q-values.

        Args:
            s (numpy array): The current state of the environment in its raw form.

        Returns:
            (s_,r,done,info) (tuple):
                s_ (numpy array): The unprocessed state that the environment is in after acting.
                r (float): The reward received by the agent due to its action.
                done (bool): Whether s_ is a terminal state.
                info (dict): Additional information returned by the environment.

        """
        if self.preprocessor is not None:
            s = self.preprocessor.process(s)
        action = np.argmax(self.live_network.predict(s))
        return self.env.step(action)

    def _choose_action(self, s):
        """Act epsilon-greedily. Note that s should be passed in its processed form."""
        if np.random.rand() < self.epsilon:
            a = np.random.randint(self.num_actions)
        else:
            a = np.argmax(self.live_network.predict(s))
        return a

    def _calc_targets(self, batch):
        """Calculate the target value for each experience in batch.

        This is:
        if DDQN:
            reward if done
            reward + gamma*Q_target(s_,argmax_{a'}(Q_live(s_,a'))) otherwise
        else (i.e. vanilla DQN):
            reward if done
            reward + gamma*max_a(Q_target(s_, a)) otherwise
        """
        rewards = [exp[2] for exp in batch]
        next_states = [exp[3] for exp in batch]
        done = [exp[4] for exp in batch]

        # end_modifier is 0 for terminal states and 1 otherwise to implement logic in docstring
        is_not_done = np.logical_not(done)
        end_modifier = is_not_done.astype(float)

        if self.double_dqn:
            actions = np.argmax(self.live_network.predict_batch(next_states), axis=1)
            rows = np.arange(len(batch))
            cols = actions
            q_sa = self.target_network.predict_batch(next_states)[rows, cols]
        else:
            q_sa = np.max(self.target_network.predict_batch(next_states), axis=1)

        targets = rewards + end_modifier*self.gamma*q_sa
        return targets

    def _calc_priority(self, s, a, r, s_, d):
        """Given (s,a,r,s_,d) transition, calculate the associated priority.

        The formula priority = (error + eps)^alpha is used here.

        """
        transition = np.array([(s,a,r,s_,d)], dtype=object)
        target = self._calc_targets(transition)[0]
        predict = self.live_network.predict(s)[a]
        error = abs(predict - target)
        p = (error + self.replay_memory.eps)**self.replay_memory.alpha
        return p

    def _update_target_network(self):
        """Update the target network by assigning its weight values to match those of the live network."""
        self.target_network.assign_weights(self.live_network)

    def _anneal_epsilon(self, step_num):
        """Anneal epsilon according to the number of elapsed training steps, step_num."""
        if self.epsilon > self.min_epsilon:
            self.epsilon = self.max_epsilon - (step_num/self.num_annealing_steps) \
                           * (self.max_epsilon-self.min_epsilon)

    def _write_epsilon_summary(self, num_steps):
        """Represent the value of self.epsilon in tensorboard."""
        summ = tf.Summary(value=[tf.Summary.Value(tag="Epsilon",
                                                  simple_value=self.epsilon),])
        self.writer.add_summary(summ, num_steps)

    def _write_q_summary(self, q_val, num_steps):
        """Represent the Q value in tensorboard."""
        summ = tf.Summary(value=[tf.Summary.Value(tag="Q",
                                                  simple_value=q_val),])
        self.writer.add_summary(summ, num_steps)

    def _write_reward_summary(self, episode_reward, num_steps):
        """Represent episode reward in tensorboard."""
        summ = tf.Summary(value=[tf.Summary.Value(tag="Episode Reward",
                                                  simple_value=episode_reward),])
        self.writer.add_summary(summ, num_steps)
