from triple_town_game import TripleTownHandler
from triple_town_simulate import TripleTownSim
from agent import TripleTownAgent

def main():
    game = TripleTownHandler()
    gameSim = TripleTownSim()
    agent = TripleTownAgent()
    agent.load("models/triple_town_model_final.pt")
    game.reset()
    action = -1
    if game.screen_center.x == -1:
        exit()
    while True:
        state = game.game_status()
        gameSim.display_board(state)
        if action == 0:
            block = True
        else:
            block = False
        action = agent.select_action(state, block, explore=False)
        game.click_slot(action)
        if gameSim.is_game_over(state):
            break


if __name__ == "__main__":
    main()