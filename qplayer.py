import json
import random
import pickle
from board import AdaptedGame as Connect4Game
from utils import Player, Synchronizer, epsilon_decay

class Memory:
    def __init__(self, gamma=0.89, lr=0.5, name="data/data_u.json"):
        self.gamma = gamma
        self.lr = lr
        self.values = {} # Q[s][a]
        self.load(name)
    
    def Q(self,s,a):
        if s in self.values and a in self.values[s]:
            return self.values[s][a]
        return 0

    def max(self, s, legal_actions):
        max = float("-inf")
        for a in legal_actions:
            q_sa = self.Q(s,a)
            if max < q_sa:
                max = q_sa
        
        return max
    
    def init_sa(self, s, a):
        if s not in self.values:
            self.values[s] = {a: 0}
        elif a not in self.values[s]:
            self.values[s].update({a: 0})
    
    def rm_useless_sa(self, s, a):
        if self.values[s][a] == 0:
            del self.values[s][a]
            if self.values[s] == {}:
                del self.values[s]

    def update(self, type, moves):
        r = 1
        if type == 0:
            r = -r

        moves = moves[::-1]
        self.init_sa(moves[0][2], moves[0][3])
        self.values[moves[0][2]][moves[0][3]] = r

        for i in range(1, len(moves)):
            s = moves[i][2]
            a = moves[i][3]
            self.init_sa(s, a)
            self.values[s][a] = self.values[s][a] + self.lr * (r + self.gamma * self.max(moves[i - 1][2], moves[i - 1][1]) - self.values[s][a])
            self.rm_useless_sa(s,a)

    def load(self, name="data/data_u.json"):
        def jsonKeys2int(x):
            if isinstance(x, dict):
                return {int(k):v for k,v in x.items()}
            return x
        try:
            with open(name, 'r') as file:
                self.values = json.load(file, object_hook=jsonKeys2int)
        except IOError:
            pass
    
    def save(self, name="data/data_u.json"):
        try:
            with open(name, 'w') as file:
                json.dump(self.values, file)
        except IOError:
            pass

class QPlayer(Player):
    def __init__(self, memory: Memory, epsilon=1):
        self.epsilon = epsilon
        self.memory = memory

    def choice(self, legal_moves, best):
        return random.choice(best) if random.random() > self.epsilon \
            else legal_moves[random.randint(0,len(legal_moves) - 1)]

    def move(self, board, legal_moves, hash):
        best_moves = None
        for i in legal_moves:
            if not best_moves:
                best_moves = [i]
            elif self.memory.Q(hash,best_moves[0]) == self.memory.Q(hash, i):
                best_moves.append(i)
            elif self.memory.Q(hash,best_moves[0]) < self.memory.Q(hash, i):
                best_moves = [i]
        
        action = self.choice(legal_moves, best_moves)

        return action, self.memory.Q(hash, action)

    def update_epsilon(self, N, t):
        self.epsilon = epsilon_decay(N,t)

def play_against_random(memory, episodes=10000, train=True):
    game = Connect4Game().copy_state()
    epsilon = 0 if not train else 1
    p = QPlayer(memory, epsilon=epsilon)
    sync = Synchronizer(game, p, QPlayer(Memory(name="data/empty.json")))

    win_rate = 0
    t = 0
    while t < episodes:
        winner, board_state, moves = sync.play()
        if board_state != 0:
            if winner[1] == p:
                win_rate += 1
            if train:
                memory.update(1, moves[winner[0]])
                memory.update(0, moves[winner[0] - 1])
        game.reset_game()
        t += 1
        if train:
            p.update_epsilon(episodes, t)
        if t % 10000 == 0:
            print("{}: winning {:.2f}% of games".format(t,win_rate/t * 100))
        if t % 1000000 == 0:
            memory.save("data/data_r.json")

    print("WON: {:.2f}% of games".format(win_rate/episodes * 100))

def train_self_play(memory: Memory, episodes=10000):
    game = Connect4Game().copy_state()
    p1 = QPlayer(memory)
    sync = Synchronizer(game, p1, p1)
    
    t = 0

    while t < episodes:
        winner, board_state, moves = sync.play()
        if board_state != 0:
            memory.update(1, moves[winner[0]])
            memory.update(0, moves[winner[0] - 1])
        game.reset_game()
        t += 1
        p1.update_epsilon(episodes, t)
        if t % 10000 == 0:
            print(t)
        if t % 1000000 == 0:
            memory.save("data/data_s.json")

def train_against_fixed(memory: Memory, episodes=10000):
    game = Connect4Game().copy_state()
    p1 = QPlayer(memory, epsilon=0.2)

    t = 0
    while t < episodes:
        p2 = QPlayer(pickle.loads(pickle.dumps(memory)), epsilon=0.2)
        sync = Synchronizer(game, p1, p2)
        for i in range(t, t + 101):
            winner, board_state, moves = sync.play()
            if board_state != 0:
                if winner[1] == p1:
                    memory.update(1, moves[winner[0]])
                    memory.update(0, moves[winner[0] - 1])
            game.reset_game()
        t = i
        if t % 10000 == 0:
            print(t)
        if t % 1000000 == 0:
            memory.save("data/data_f.json")

def arena(players):
    game = Connect4Game().copy_state()
    scores = [[] for _ in range(len(players))]
    draws = [0] * len(players)
    for i in range(len(players) - 1):
        p1 = players[i]
        for j in range(i+1, len(players)):
            p2 = players[j]
            sync = Synchronizer(game, p1, p2)
            wins = [0, 0]
            for _ in range(100):
                winner, board_state, moves = sync.play()
                if board_state != 0:
                    if winner[1] == p1:
                        wins[0] += 1
                    if winner[1] == p2:
                        wins[1] += 1
                else:
                   draws[i] += 1
                   draws[j] += 1 
                game.reset_game()
            scores[i].append(wins[0])
            scores[j].append(wins[1])
    return scores, draws

if __name__ == '__main__':
    from humanplayer import play_human
    GAMMA = 0.95
    LR = 0.1
    EPISODES = 1000000
    
    # Uncomment to train against a random player 
    # print("TRAINING AGAINST RANDOM")
    # memory = Memory(gamma=GAMMA, lr=LR, name="data/data_r.json")
    # play_against_random(memory, episodes=EPISODES)
    # memory.save("data/data_r.json")
    # print(len(memory.values))
    # play_against_random(memory, episodes=100, train=False)

    print("TRAINING AGAINST SELF")
    self_memory = Memory(gamma=GAMMA, lr=LR, name="data/data_s.json")
    train_self_play(self_memory, episodes=EPISODES)
    self_memory.save(name="data/data_s.json")
    print(len(self_memory.values))
    play_against_random(self_memory, episodes=100, train=False)

    # Uncomment to train against a fixed player
    # print("TRAINING AGAINST FIXED")
    # f_memory = Memory(gamma=GAMMA, lr=LR, name="data/data_f.json")
    # train_against_fixed(f_memory, episodes=EPISODES)
    # f_memory.save(name="data/data_f.json")
    # print(len(f_memory.values))
    # play_against_random(f_memory, episodes=100, train=False)

    # Play against the AI right after training
    play_human(QPlayer(self_memory, epsilon=0))

    # Use this function if you want to see which of the AI is the best
    # print(arena([QPlayer(memory, epsilon=0), QPlayer(self_memory, epsilon=0), QPlayer(f_memory, epsilon=0)]))