import numpy as np
import random

items = {
    "empty": 0,
    "grass": 1,
    "bush": 2,
    "tree": 3,
    "hut": 4,
    "house": 5,
    "mansion": 6,
    "castle": 7,
    "Fcastle": 8,
    "Tcastle": 9,
    "bear": 10,
    "Nbear": 11,
    "tombstone": 12,
    "church": 13,
    "cathedral": 14,
    "treasure": 15,
    "Ltreasure": 16,
    "bot": 17,
    "mountain": 18,
    "rock": 19,
    "crystal": 20,
    "unknown": 21
}

ritem = {value: key for key, value in items.items()}

item_picture = {
    "0": "🔲",
    "1": "🌱",
    "2": "🌳",
    "3": "🌲",
    "4": "🗼",
    "5": "🏠",
    "6": "🏫",
    "7": "🏬",
    "8": "🏯",
    "9": "🏰",
    "10": "🐻",
    "11": "🐼",
    "12": "⚰️",
    "13": "⛪",
    "14": "🕍",
    "15": "💰",
    "16": "👑",
    "17": "🤖",
    "18": "⛰️",
    "19": "🪨 ",
    "20": "💎",
    "21": "? "
}

upgrade = {
    "grass": "bush",
    "bush": "tree",
    "tree": "hut",
    "hut": "house",
    "house": "mansion",
    "mansion": "castle",
    "castle": "Fcastle",
    "Fcastle": "Tcastle",
    "tombstone": "church",
    "church": "cathedral",
    "cathedral": "treasure",
    "rock":"mountain",
    "treasure": "Ltreasure",
    "mountain": "Ltreasure",
}

downgrade = {value: key for key, value in upgrade.items()}

class triple_town_sim:
    def __init__(self, state = None):
        self.memory = None
        self.memory_time = None
        self.random_item = None
        if state is None:
            state = self.random_init()
        self.last_state = state
        self.next_item = state[0]
        self.state_matrix = state[1:].reshape(6, 6)
        self.time_matrix = np.zeros((6, 6), dtype=int)
        self.game_score = 0
        self.last_game_score = 0
        self.reload_time(state)
        # self.last_action = 0

    def random_init(self):
        random_item = np.random.choice(
            [items["grass"], items["bush"], items["tree"], items["hut"], items["bear"], items["Nbear"], items["crystal"], items["bot"]],
            p=[0.605, 0.155, 0.02, 0.005, 0.15, 0.015, 0.025, 0.025]
        )
        state_matrix = np.random.choice(
            [items["empty"], items["grass"], items["bush"], items["tree"], items["hut"], items["bear"], items["rock"]],
            p=[0.34, 0.355, 0.155, 0.02, 0.005, 0.10, 0.025],
            size=(6, 6)
        )
        state = np.zeros(37, dtype=int)
        state[0] = random_item
        state[1:] = state_matrix.flatten()
        return state

    def slot_item_bind(self, slot, item):
        state = np.zeros(37, dtype=int)
        state[0] = int(item)
        for i in range(36):
            state[i + 1] = int(slot.flatten()[i])
        return state
    
    def slot_item_split(self, state_all):
        slot_matrix = state_all[1:].reshape(6, 6)
        item = state_all[0]

        return slot_matrix, item
    
    def reload_time(self, state_all):
        state, item = self.slot_item_split(state_all)
        time = 1
        for i in range(36):
            row, col = divmod(i, 6)
            if state[row, col] != 0:
                self.time_matrix[row, col] = time
                time += 1

    def add_new_item(self, row, col):
        for i in range(36):
            r, c = divmod(i, 6)
            if self.time_matrix[r, c] > 0:
                self.time_matrix[r, c] += 1
        self.time_matrix[row, col] = 1

    def fix_state(self, state):
        state_matrix = state[1:].reshape(6, 6)
        unknown_positions = [(row, col) for row in range(6) for col in range(6) 
                            if state_matrix[row, col] == items["unknown"]]
        
        for row, col in unknown_positions:
            target_bear_count = np.count_nonzero(state_matrix == items["bear"])
            target_nbear_count = np.count_nonzero(state_matrix == items["Nbear"])
            current_bear_count = np.count_nonzero(self.state_matrix == items["bear"])
            current_nbear_count = np.count_nonzero(self.state_matrix == items["Nbear"])

            if target_bear_count != current_bear_count:
                state_matrix[row, col] = items["bear"]
            elif target_nbear_count != current_nbear_count:
                state_matrix[row, col] = items["Nbear"]
            else:
                state_matrix[row, col] = self.state_matrix[row, col]

        return self.slot_item_bind(state_matrix, state[0])
    
    def try_match(self, state1, state2):
        self.fix_state(state2)
        state1_matrix = state1[1:].reshape(6, 6)
        state2_matrix = state2[1:].reshape(6, 6)
        timechange1 = set()
        timechange2 = set()

        for i in range(36):
            row, col = divmod(i, 6)
            if state1_matrix[row, col] != state2_matrix[row, col]:
                if state1_matrix[row, col] in {items["bear"], items["Nbear"]}:
                    timechange1.add((row, col))
                if state2_matrix[row, col] in {items["bear"], items["Nbear"]}:
                    timechange2.add((row, col))
                if not timechange1 and not timechange2:
                    self.reload_time(state2)
                    return
        if len(timechange1) != len(timechange2):
            self.reload_time(state2)
            return
        if not timechange1 and not timechange2:
            return

        bear_time = None
        for r1, c1 in timechange1:
            checktype = state1_matrix[r1, c1]
            for r2, c2 in timechange2:
                if state2_matrix[r2, c2] == checktype:
                    bear_time = self.time_matrix[r1, c1]
                    self.time_matrix[r2, c2] = bear_time
                    self.time_matrix[r1, c1] = 0
                    break
            timechange2.remove((r2, c2))

    def next_state_simulate(self, current_state, action):
        valid_mask = self.valid_action_mask(current_state)
        state, item = self.slot_item_split(current_state)
        
        if sum(valid_mask) == 1:
            print("No valid action")
            return None

        # if action is -1, return to last state
        if action == -1:
            if self.memory is not None:
                self.random_item = current_state[0]
                self.time_matrix = self.memory_time.copy()
                self.last_state = self.memory.copy()
                self.state_matrix = self.memory.copy()[1:].reshape(6, 6)
                self.next_item = self.memory.copy()[0]
                self.game_score = self.last_game_score
                return self.memory
            else:
                return current_state
        else:
            self.random_item = None
            self.memory = current_state.copy()
            self.memory_time = self.time_matrix.copy()
            self.last_game_score = self.game_score

        if valid_mask[action] == 0:
            print("Invalid action")
            return current_state

        self.try_match(self.last_state, current_state)
        self.last_state = current_state
        self.next_item = current_state[0]
        self.state_matrix = current_state[1:].reshape(6, 6)

        # if action is 0, then swap the item
        if action == 0:
            store_item = state[0][0]
            state[0][0] = item
            item = int(store_item)
            if item == items["empty"]:
                item = np.random.choice(
                    [items["grass"], items["bush"], items["tree"], items["hut"], items["bear"], items["Nbear"], items["crystal"], items["bot"]],
                    p=[0.605, 0.155, 0.02, 0.005, 0.15, 0.015, 0.025, 0.025]
                )
            self.state_matrix = state
            self.next_item = item
            self.last_state = self.slot_item_bind(state, item)
            return self.last_state
        
        row, col = divmod(action, 6)

        if state[row, col] == items["treasure"] or state[row, col] == items["Ltreasure"]:
            state[row, col] = 0
            self.time_matrix[row, col] = 0
            return self.slot_item_bind(state, item)
        
        # update bot
        if item == items["bot"]:
            if state[row, col] == items["bear"] or state[row, col] == items["Nbear"]:
                state[row, col] = items["tombstone"]
            elif state[row, col] == items["mountain"]:
                state[row, col] = items["treasure"]
            else:
                state[row, col] = items["empty"]
        else:
            state[row, col] = item
        
        self.add_new_item(row, col)
        
        # update connected elements
        if 0 < state[row, col] < 9:
            state = self.update_connected_elements(state, row, col)

        # update crystal
        if state[row, col] == 20:
            state = self.update_crystal(state, row, col)
        
        # update bear
        state = self.check_bear_marge(state)

        state = self.update_bear_move(state, items["bear"])
        state = self.update_bear_move(state, items["Nbear"])

        self.state_matrix = state
        if self.random_item is None:
            self.next_item = np.random.choice(
                [items["grass"], items["bush"], items["tree"], items["hut"], items["bear"], items["Nbear"], items["crystal"], items["bot"]],
                p=[0.605, 0.155, 0.02, 0.005, 0.15, 0.015, 0.025, 0.025]
            )
        else:
            self.next_item = self.random_item
        self.last_state = self.slot_item_bind(self.state_matrix, self.next_item)
        return self.last_state

    def valid_action_mask(self, state_all, block = False):
        mask = np.zeros(36)
        state, next_item = self.slot_item_split(state_all)
        state_flatten = state.flatten()
        if next_item == 17:
            mask[(state_flatten != 0)] = 1
        else:
            mask[(state_flatten == 15) | (state_flatten == 16) | (state_flatten == 0)] = 1
        if block:
            mask[0] = 0
        else:
            mask[0] = 1
        return mask
    
    def find_connected_elements(self, matrix, start_row, start_col):
        rows, cols = matrix.shape
        target_value = matrix[start_row, start_col]
        visited = set()
        result = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        queue = [(start_row, start_col)]
        visited.add((start_row, start_col))

        is_bear_type = target_value in (items["bear"], items["Nbear"])

        while queue:
            current_row, current_col = queue.pop(0)
            result.append((current_row, current_col))

            for dr, dc in directions:
                new_row, new_col = current_row + dr, current_col + dc
                if (new_row != 0 or new_col != 0):
                    if (0 <= new_row < rows and 
                        0 <= new_col < cols and 
                        (new_row, new_col) not in visited):
                        if is_bear_type:
                            if matrix[new_row, new_col] in (items["bear"], items["Nbear"]):
                                queue.append((new_row, new_col))
                                visited.add((new_row, new_col))
                        elif matrix[new_row, new_col] == target_value:
                            queue.append((new_row, new_col))
                            visited.add((new_row, new_col))
        return result
    
    def update_connected_elements(self, matrix, start_row, start_col):
        connected_list = self.find_connected_elements(matrix, start_row, start_col)
        item = matrix[start_row, start_col]
        if item == items["Tcastle"]:
            return matrix
        while len(connected_list) >= 3:
            if item == items["Fcastle"] and len(connected_list) < 4:
                return matrix
            self.gamescore += 1
            for r, c in connected_list:
                matrix[r, c] = 0
                self.time_matrix[r, c] = 0
            matrix[start_row, start_col] = items[upgrade[ritem[item]]]
            self.add_new_item(start_row, start_col)
            connected_list = self.find_connected_elements(matrix, start_row, start_col)
            item = matrix[start_row, start_col]
        return matrix
    
    def update_crystal(self, matrix, row, col):
        marge_list = []
        for i in ["Fcastle", "castle", "mountain", "treasure", "mansion", "cathedral", "church", "house", "hut", "tombstone", "rock", "tree", "bush", "grass"]:
            matrix[row, col] = items[i]
            connected_list = self.find_connected_elements(matrix, row, col)
            if i == "Fcastle" and len(connected_list) >= 4:
                marge_list.append(i)
                continue
            if len(connected_list) >= 3:
                marge_list.append(i)

        if len(marge_list) == 0:
            matrix[row, col] = items["rock"]
        else:
            marge_item = marge_list[0]
            while downgrade[marge_item] in marge_list:
                marge_item = downgrade[marge_item]
                if marge_item == "grass" or marge_item == "tomstone" or marge_item == "rock":
                    break
            matrix[row, col] = items[marge_item]
            print(f"crystal marge to {marge_item}")
            self.update_connected_elements(matrix, row, col)

        return matrix
    
    def check_bear_marge(self, matrix):
        visited = set()
        movenumber = 0
        for i in range(35):
            row, col = divmod(i+1, 6)
            if (matrix[row, col] == items["bear"] or matrix[row, col] == items["Nbear"]) and (row, col) not in visited:
                connected_list = self.find_connected_elements(matrix, row, col)
                for r, c in connected_list:
                    if matrix[r, c] == items["bear"]:
                        movenumber += len(self.get_bear_moves(matrix, r, c)[2])
                    elif matrix[r, c] == items["Nbear"]:
                        movenumber += len(self.get_Nbear_moves(matrix, r, c)[2])
                    visited.add((r, c))
                if movenumber == 0:
                    for r, c in connected_list:
                        matrix[r, c] = items["tombstone"]
                movenumber = 0
        item_time = 99
        check_r = 0
        check_c = 0
        for i in range(36):
            row, col = divmod(i, 6)
            if matrix[row, col] == items["tombstone"]:
                connected_list = self.find_connected_elements(matrix, row, col)
                if len(connected_list) >= 3:
                    for r, c in connected_list:
                        if self.time_matrix[r, c] < item_time:
                            item_time = self.time_matrix[r, c]
                            check_r = r
                            check_c = c
                    self.update_connected_elements(matrix, check_r, check_c)
        return matrix

    def update_bear_move(self, matrix, beartype):
        visited = set()
        movelist = []
        for i in range(35):
            row, col = divmod(i+1, 6)
            renew_row, renew_col = col, row
            if matrix[renew_row, renew_col] == beartype and (renew_row, renew_col) not in visited:
                if beartype == items["bear"]:
                    movelist = self.get_bear_moves(matrix, renew_row, renew_col)
                elif beartype == items["Nbear"]:
                    movelist = self.get_Nbear_moves(matrix, renew_row, renew_col)
                if movelist[2] != []:
                    moveplace = random.choice(movelist[2])
                    matrix[renew_row, renew_col] = items["empty"]
                    matrix[moveplace[0], moveplace[1]] = beartype
                    beartime = self.time_matrix[renew_row, renew_col]
                    self.time_matrix[moveplace[0], moveplace[1]] = beartime
                    self.time_matrix[renew_row, renew_col] = 0
                    visited.add((moveplace[0], moveplace[1]))
                else:
                    visited.add((renew_row, renew_col))
        return matrix
                
    
    def get_bear_moves(self, matrix, bear_row, bear_col):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        valid_moves = []

        for i in range(len(directions)):
            dx, dy = directions[i]
            new_x, new_y = bear_row + dx, bear_col + dy
            # make sure the new position is within the grid
            if 0 <= new_x < len(matrix) and 0 <= new_y < len(matrix[0]):
                # if the new position is empty
                if matrix[new_x][new_y] == 0:
                    valid_moves.append((new_x, new_y))

        return (bear_row, bear_col, valid_moves)
    
    def get_Nbear_moves(self, matrix, bear_row, bear_col):
        valid_moves = []
        for i in range(35):
            row, col = divmod(i + 1, 6)
            if matrix[row, col] == 0:
                valid_moves.append((row, col))
        return (bear_row, bear_col, valid_moves)
    
    def console_print(self, state):
        state_matrix = state[1:].reshape(6, 6)
        next_item = state[0]
        print("next item:", item_picture[str(next_item)])
        # print action index in front of item
        for i in range(36):
            row, col = divmod(i, 6)
            print(f"{i}:".rjust(3), end=" ")
            print(item_picture[str(state_matrix[row, col])].rjust(1), end=" ")
            if col == 5:
                print()

test = True

if test:
    sim_game = triple_town_sim(
        state = np.array([ 1,
            1 ,0 ,8 ,0 ,0 ,0 ,
            0 ,0 ,0 ,4 ,4 ,1 ,
            0 ,0 ,4 ,0 ,0 ,2 ,
            4 ,0 ,4 ,1 ,1 ,15,
            0 ,1 ,3 ,0 ,10,11,
            11,2 ,0 ,13,13,14]))
    state = sim_game.last_state
    while True:
        sim_game.console_print(state)
        action = int(input("action:"))
        state = sim_game.next_state_simulate(sim_game.last_state, action)
        print("game score:", sim_game.game_score)
        print("=============================================================")
        if state is None:
            print("Game Over")
            break
    


