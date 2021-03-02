"""Recreates an agent trained through dqn_trainer.learn()."""

import numpy as np
import tensorflow as tf


class DQNAgent:
    """Generate a trained agent from saved tensorflow files."""

    def __init__(self, env, model_meta, model_checkpoint, preprocessor=None):
        """Import model specified by model_meta and model_checkpoint for use within environment env.

        Args:
            env (gym environment object): The OpenAI Gym environment the agent should interact with.
            model_meta (str): .meta file of the saved model to be imported, e.g. "./q_model-70000.meta".
            model_checkpoint (str): checkpoint of the saved model to be imported, e.g. "./q_model-70000".
            preprocessor (Preprocessor): Preprocessor used to train the saved model.

        Returns:
            None

        """
        self.env = env
        self.preprocessor = preprocessor
        self.saver = tf.train.import_meta_graph(model_meta)
        self.sess = tf.Session()
        self.saver.restore(self.sess, model_checkpoint)

    def act(self, s):
        """Given current raw state s, act greedily with respect to the agent's estimated Q-values.

        Args:
            s (numpy array): The raw state which is to be used to determine the action.
                             Passed from e.g. s = env.reset() or s, r, done, info = env.step(a).

        Returns:
            s_, r, done, info (tuple):
                s_ (numpy array): State after the agent has acted.
                r (float): Reward received due to the action taken.
                done (bool): Whether s_ is terminal.
                info (dict): Debugging information provided by the environment.

        """
        if self.preprocessor is not None:
            s = self.preprocessor.process(s)
        q_values = self.sess.run("output/BiasAdd:0", feed_dict={"state:0": [s]})[0]
        action = np.argmax(q_values)
        return self.env.step(action)
