import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
import numpy as np
import random
from board import AdaptedGame as Connect4Game
from utils import Player, Synchronizer, epsilon_decay

class Agent(Player):

  def __init__(self, gamma = 0.95, lr=0.1, input_shape=(7,6), action_size = 7, e_init = 1, e_min = 0.1):
    self.lr = lr
    self.input_shape = input_shape
    self.action_size = action_size
    self.epsilon_init = e_init
    self.epsilon_min = e_min
    self.epsilon = e_init
    self.gamma = gamma
    self.memory = []

    self.model = self._build_model()
    self.target_model = self._build_model()
    self.update_target_model()

  def memorize(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))

  def _build_model(self):
    model = Sequential()
    model.add(Flatten(input_shape=self.input_shape))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(self.action_size, activation='linear'))
    model.compile(loss=Huber(), optimizer=Adam(learning_rate=self.lr))

    return model

  def move(self, board, legal_moves, hash):
    values = self.model(np.array([board]), training=False).numpy()[0]

    res = self.max(values, legal_moves)

    action = random.choice(res[0]) if random.random() > self.epsilon \
      else legal_moves[random.randint(0,len(legal_moves) - 1)]
    return action, values[action]

  def max(self, values, actions):
    best_actions = [actions[0]]
    max_val = values[actions[0]]

    for a in actions[1:]:
        if max_val < values[a]:
            best_actions = [a]
            max_val = values[a]
        elif max_val == values[a]:
            best_actions.append(a)

    return best_actions, max_val

  def replay(self, batch_size):
      minibatch = random.sample(self.memory, batch_size)
      targetbatch = np.zeros((batch_size,self.action_size))
      statebatch = np.zeros((batch_size, self.input_shape[0], self.input_shape[1]))
      i = 0
      for state, action, reward, next_state, done in minibatch:
        statebatch[i] = state[0]
        target = self.model(state[0], training=False).numpy()
        if done:
          target[0][action] = reward
        else:
          t = self.model(next_state[0], training=False).numpy()[0]
          res = self.max(t, next_state[1])
          target[0][action] = reward + self.gamma * res[1]
        targetbatch[i] = target
        i += 1
      self.model.fit(statebatch, targetbatch, epochs=1, verbose=0)

  def update_epsilon(self, N, t):
    self.epsilon = epsilon_decay(N,t, self.epsilon_min, self.epsilon_init)
  
  def update_target_model(self):
    self.target_model.set_weights(self.model.get_weights())
  
  def load(self, name):
    self.model.load_weights(name)

  def save(self, name):
    self.model.save_weights(name)

def train_self_play(episodes=5000):
    game = Connect4Game().copy_state()
    p = Agent()
    sync = Synchronizer(game, p, p)

    batch_size = 32
    update_target_ctr = 0

    for t in range(episodes):
        winner, board_state, moves = sync.play()
        
        r = 0
        if board_state != 0:
            r = 1
        feed_moves(moves[winner[0]], r, p)
        feed_moves(moves[winner[0] - 1], -r, p)

        p.update_epsilon(episodes, t)

        game.reset_game()

        if len(p.memory) > batch_size:
            p.replay(batch_size)
            update_target_ctr += 1
        if update_target_ctr == 100:
            p.update_target_model()
            update_target_ctr = 0

        if t % 100 == 0:
            print(t)

    p.save("data/dqn_weights")

def feed_moves(moves, r, agent):
    moves = moves[::-1]
    prev = (np.array([moves[0][0]]), moves[0][1])
    agent.memorize(prev, moves[0][3], r, None, True)
    for i in range(1, len(moves)):
        curr = (np.array([moves[i][0]]), moves[i][1])
        agent.memorize(curr, moves[i][3], 0, prev, False)
        prev = curr

if __name__ == '__main__':
#   train_self_play()
    from humanplayer import play_human
    a = Agent(e_init=0)
    a.load("data/dqn_weights")

    play_human(a)