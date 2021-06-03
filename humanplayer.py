import pygame
import pygame.gfxdraw
from board import AdaptedGame as Connect4Game, Connect4Viewer, SQUARE_SIZE
from utils import Player, Synchronizer

class HumanPlayer(Player):
    def move(self, board, legal_moves, hash):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None, None
                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    action = pygame.mouse.get_pos()[0] // SQUARE_SIZE
                    if action not in legal_moves:
                        continue
                    return action, None

def play_human(p: Player, callback = None):
    game = Connect4Game()	
    view = Connect4Viewer(game=game)
    view.initialize()

    sync = Synchronizer(game, p, HumanPlayer())
    
    while True:
        winner, board_state, moves = sync.play(True)
        if callback:
            callback(winner, board_state, moves, p)
        while True:
            event = pygame.event.wait()
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                game.reset_game()
                break