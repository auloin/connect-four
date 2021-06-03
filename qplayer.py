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
    
    def update_q_sa(self, s, a, max_q, r = 0):
        if s not in self.values:
            self.values[s] = {a: 0}
        elif a not in self.values[s]:
            self.values[s].update({a: 0})

        if not r:
            self.values[s][a] = r
        else:
            self.values[s][a] = self.values[s][a] + self.lr * (r + self.gamma * max_q - self.values[s][a])

        if self.values[s][a] == 0:
            del self.values[s][a]
            if self.values[s] == {}:
                del self.values[s]

    def update(self, type, moves):
        r = 1
        if type == 0:
            r = -r

        moves = moves[::-1]

        self.update_q_sa(moves[0][2], moves[0][3], 0, r)

        for i in range(1, len(moves)):
            self.update_q_sa(moves[i][2], moves[i][3], self.max(moves[i - 1][2], moves[i - 1][1]))

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

# def play_human(memory):
#     game = Connect4Game()	
#     view = Connect4Viewer(game=game)
#     view.initialize()

#     sync = Synchronizer(game, QPlayer(memory, epsilon=0), HumanPlayer())
    
#     while True:
#         winner, board_state, moves = sync.play(True)
#         if board_state != 0:
#             memory.update(1, moves[winner[0]])
#             memory.update(0, moves[winner[0] - 1])
#         while True:
#             event = pygame.event.wait()
#             if event.type == pygame.QUIT:
#                 pygame.quit()
#                 return
#             if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
#                 game.reset_game()
#                 break

def train_self_play(memory: Memory, episodes=10000):
    game = Connect4Game().copy_state()
    p1 = QPlayer(game, memory)
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
            memory.save()

def train_against_fixed(memory: Memory, episodes=10000):
    game = Connect4Game().copy_state()
    p1 = QPlayer(game, memory, epsilon=0.2)

    t = 0
    while t < episodes:
        p2 = QPlayer(game, pickle.loads(pickle.dumps(memory)), epsilon=0.2)
        sync = Synchronizer(game, p1, p2)
        for i in range(t, t + 101):
            winner, board_state, moves = sync.play()
            if board_state != 0:
                if winner[1] == p1:
                    memory.update(1, moves[winner])
                else:
                    memory.update(0, moves[winner - 1])
            game.reset_game()
        t = i
        if t % 10000 == 0:
            print(t)
        if t % 1000000 == 0:
            memory.save()

if __name__ == '__main__':
    from humanplayer import play_human
    memory = Memory(gamma=0.95, lr=0.1)
    #train_against_fixed(memory, episodes=1000000)
    
    #memory.save()

    play_human(QPlayer(memory, epsilon=0))