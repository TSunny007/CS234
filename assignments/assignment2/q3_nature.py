import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.general import get_logger
from utils.test_env import EnvTest
from q1_schedule import LinearExploration, LinearSchedule
from q2_linear import Linear


from configs.q3_nature import config


class NatureQN(Linear):
  """
  Implementing DeepMind's Nature paper. Here are the relevant urls.
  https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
  https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
  """
  def get_q_values_op(self, state, scope, reuse=False):
    """
    Returns Q values for all actions

    Args:
      state: (tf tensor) 
        shape = (batch_size, img height, img width, nchannels)
      scope: (string) scope name, that specifies if target network or not
      reuse: (bool) reuse of variables in the scope

    Returns:
      out: (tf tensor) of shape = (batch_size, num_actions)
    """
    # this information might be useful
    num_actions = self.env.action_space.n

    ##############################################################
    """
    TODO: implement the computation of Q values like in the paper
        https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

        you may find the section "model architecture" of the appendix of the 
        nature paper particulary useful.

        store your result in out of shape = (batch_size, num_actions)

    HINT: 
      - You may find the following functions useful:
        - tf.layers.conv2d
        - tf.layers.flatten
        - tf.layers.dense

      - Make sure to also specify the scope and reuse

    """
    ##############################################################
    ################ YOUR CODE HERE - 10-15 lines ################ 
    conv1_config = {'filters': 32,
                    'kernel_size': [8,8],
                    'strides': [4,4],
                    'activation': 'relu',
                    'padding': 'same'}
    conv2_config = {'filters': 64,
                    'kernel_size': [4,4],
                    'strides': [2,2],
                    'activation': 'relu',
                    'padding': 'same'}
    conv3_config = {'filters': 64,
                    'kernel_size': [3,3],
                    'strides': [1,1],
                    'activation': 'relu',
                    'padding': 'same'}
    hidden_units = 512
    with tf.variable_scope(scope, reuse=reuse):
      conv1 = tf.keras.layers.Conv2D(**conv1_config) (state)
      conv2 = tf.keras.layers.Conv2D(**conv2_config) (conv1)
      conv3 = tf.keras.layers.Conv2D(**conv3_config) (conv2)
      flattened = tf.keras.layers.Flatten() (conv3)
      hidden = tf.keras.layers.Dense(hidden_units, activation='relu') (flattened)
      out = tf.keras.layers.Dense(num_actions) (hidden)
    ##############################################################
    ######################## END YOUR CODE #######################
    return out


"""
Use deep Q network for test environment.
"""
if __name__ == '__main__':
  env = EnvTest((80, 80, 1))

  # exploration strategy
  exp_schedule = LinearExploration(env, config.eps_begin, 
      config.eps_end, config.eps_nsteps)

  # learning rate schedule
  lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
      config.lr_nsteps)

  # train model
  model = NatureQN(env, config)
  model.run(exp_schedule, lr_schedule)
