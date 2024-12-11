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
    "rock": 18,
    "monutain": 19,
    "crystal": 20,
    "unknown": 21
}

ritem = {value: key for key, value in items.items()}

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
    "treasure": "Ltreasure",
    "monutain": "Ltreasure",
}

class triple_town_sim:
    def __init__(self, state = np.zeros(37, dtype=int)):
        self.last_state = state
        self.next_item = state[0]
        self.state_matrix = state[1:].reshape(6, 6)
        self.time_matrix = np.zeros((6, 6), dtype=int)
        self.reload_time(state)
        # self.last_action = 0

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
            if self.time_matrix[row, col] > 0:
                self.time_matrix[i] += 1
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
        self.try_match(self.last_state, current_state)
        self.last_state = current_state
        self.next_item = current_state[0]
        self.state_matrix = current_state[1:].reshape(6, 6)
            
        valid_mask = self.valid_action_mask(current_state)
        state, item = self.slot_item_split(current_state)

        if valid_mask[action] == 0:
            return state
        
        # if action is 0, then swap the item
        if action == 0:
            store_item = state[0][0]
            state[0][0] = item
            item = int(store_item)
            return self.slot_item_bind(state, item)
        
        row, col = divmod(action, 6)
        
        # update bot
        if item == items["bot"]:
            if state[row, col] == items["bear"] or state[row, col] == items["Nbear"]:
                state[row, col] = items["tombstone"]
            elif state[row, col] == items["monutain"]:
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
            rock_check = True
            for i in {"Fcastle", "castle", "mountain", "treasure", "mansion", "cathedral", "church", "house", "hut", "rock", "tomstone", "tree", "bush", "grass"}:
                state[row, col] = items[i]
                connected_list = self.find_connected_elements(state, row, col)
                if len(connected_list) >= 3:
                    state = self.update_connected_elements(state, row, col)
                    rock_check = False
                    break
            if rock_check:
                state[row, col] = items["rock"]
        
        # update bear
        state = self.check_bear_marge(state)

        state = self.update_bear_move(state, items["bear"])
        state = self.update_bear_move(state, items["Nbear"])

        self.last_state = self.slot_item_bind(state, item)
        self.state_matrix = state
        self.next_item = np.random.choice(
            [items["grass"], items["bush"], items["tree"], items["hut"], items["bear"], items["Nbear"], items["crystal"], items["bot"]],
            p=[0.605, 0.155, 0.02, 0.005, 0.15, 0.015, 0.025, 0.025]
        )
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
        while len(connected_list) >= 3:
            for r, c in connected_list:
                matrix[r, c] = 0
                self.time_matrix[r, c] = 0
            matrix[start_row, start_col] = items[upgrade[ritem[item]]]
            self.add_new_item(start_row, start_col)
            connected_list = self.find_connected_elements(matrix, start_row, start_col)
            item = matrix[start_row, start_col]
        return matrix
    
    def check_bear_marge(self, matrix):
        visited = set()
        movelist = []
        for i in range(36):
            row, col = divmod(i, 6)
            if (matrix[row, col] == items["bear"] or matrix[row, col] == items["Nbear"]) and (row, col) not in visited:
                connected_list = self.find_connected_elements(matrix, row, col)
                for r, c in connected_list:
                    if matrix[r, c] == items["bear"]:
                        movelist = movelist.append(self.get_bear_moves(matrix, r, c))
                    elif matrix[r, c] == items["Nbear"]:
                        movelist = movelist.append(self.get_Nbear_moves(matrix, r, c))
                    visited.add((r, c))
                if movelist == []:
                    for r, c in connected_list:
                        matrix[r, c] = items["tombstone"]
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
                            item_time = self.time_matrix(r, c)
                            check_r = r
                            check_c = c
                    self.update_connected_elements(matrix, check_r, check_c)
        return matrix

    def update_bear_move(self, matrix, beartype):
        visited = set()
        movelist = []
        for i in range(36):
            row, col = divmod(i, 6)
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
            slot = i + 1
            row, col = divmod(slot, 6)
            if matrix[row, col] == 0:
                valid_moves.append((row, col))
        return (bear_row, bear_col, valid_moves)

test = False

if test:
    sim_game = triple_town_sim(
        state = np.array([1,
            0,0,0,0,0,0,
            0,4,2,1,0,0,
            0,4,0,0,0,0,
            1,1,19,0,0,19,
            0,0,0,0,1,0,
            1,12,0,0,0,1]))
    while True:
        print("next item:")
        print(sim_game.next_item)
        print("state:")
        print(sim_game.state_matrix)
        action = int(input("action: "))
        state = sim_game.next_state_simulate(sim_game.last_state, action)
    


