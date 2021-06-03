import random
from board import AdaptedGame as Connect4Game

class Player:
    def __init__(self):
        pass
    def move(self, board, legal_moves, hash):
        pass

class Synchronizer:
    def __init__(self, game: Connect4Game,  player1: Player, player2: Player):
        self.players = [player1, player2]
        self.game = game

    def play(self, debug = False):
        moves = [[] for _ in range(2)]
        turn = 0
        while True:
            board = self.game.get_board_copy()
            hash = self.game.hash()
            legal_actions = self.game.get_legal_actions()
            action, val = self.players[turn].move(board, legal_actions, hash)
            if action == None:
                break
            if debug:
                print("{}: {} {}".format(self.players[turn].__class__.__name__, action, val))
            self.game.place(action)
            moves[turn].append((board, legal_actions, hash, action))
            if self.game.get_win() != None:
                break
            turn = not turn
        winner = (turn, self.players[turn])
        self.players = self.players[::-1]
        return winner, self.game.get_win(), moves

def epsilon_decay(N, t, e_min = 0, e_init = 1):
    r = (N - t) / N
    return (e_init - e_min) * r + e_min