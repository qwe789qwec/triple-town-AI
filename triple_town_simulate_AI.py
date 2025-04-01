import numpy as np
import random

class TripleTownSim:
    """Triple Town遊戲模擬器"""
    
    # 物品定義
    ITEMS = {
        "empty": 0,
        "grass": 1,
        "bush": 2,
        "tree": 3,
        "hut": 4,
        "house": 5,
        "mansion": 6,
        "castle": 7,
        "Fcastle": 8,   # 浮動城堡
        "Tcastle": 9,   # 三重城堡
        "bear": 10,
        "Nbear": 11,    # 忍者熊
        "tombstone": 12,
        "church": 13,
        "cathedral": 14,
        "treasure": 15,
        "Ltreasure": 16,  # 巨型寶藏
        "bot": 17,
        "mountain": 18,
        "rock": 19,
        "crystal": 20,
        "unknown": 21
    }
    
    # 物品名稱反查表
    ITEM_NAMES = {value: key for key, value in ITEMS.items()}
    
    # 物品圖示
    ITEM_ICONS = {
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
    
    # 物品升級表
    UPGRADE_MAP = {
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
        "rock": "mountain",
        "treasure": "Ltreasure",
        "mountain": "Ltreasure",
    }
    
    # 物品降級表
    DOWNGRADE_MAP = {value: key for key, value in UPGRADE_MAP.items()}
    
    # 物品生成概率
    ITEM_PROBABILITIES = {
        "grass": 0.605,
        "bush": 0.155,
        "tree": 0.02,
        "hut": 0.005,
        "bear": 0.15,
        "Nbear": 0.015,
        "crystal": 0.025,
        "bot": 0.025
    }
    
    # 初始棋盤生成概率
    BOARD_PROBABILITIES = {
        "empty": 0.34,
        "grass": 0.355,
        "bush": 0.155,
        "tree": 0.02,
        "hut": 0.005,
        "bear": 0.10,
        "rock": 0.025
    }
    
    def __init__(self, state=None):
        """初始化遊戲狀態"""
        self.memory = None
        self.memory_time = None
        self.random_item = None
        
        if state is None:
            state = self._generate_random_state()
        
        self.last_state = state
        self.next_item = state[0]
        self.state_matrix = state[1:].reshape(6, 6)
        self.time_matrix = np.zeros((6, 6), dtype=int)
        self.game_score = 0
        self.last_game_score = 0
        self._reload_time_matrix(state)
    
    def _generate_random_state(self):
        """生成隨機初始狀態"""
        # 生成隨機物品
        random_item_id = self._get_random_item()
        
        # 生成隨機棋盤
        state_matrix = self._get_random_board()
        
        # 組合狀態
        state = np.zeros(37, dtype=int)
        state[0] = random_item_id
        state[1:] = state_matrix.flatten()
        state[1] = 0  # 確保左上角為空
        
        return state
    
    def _get_random_item(self):
        """生成隨機物品"""
        items = [self.ITEMS[item] for item in self.ITEM_PROBABILITIES.keys()]
        probs = list(self.ITEM_PROBABILITIES.values())
        return np.random.choice(items, p=probs)
    
    def _get_random_board(self):
        """生成隨機棋盤"""
        items = [self.ITEMS[item] for item in self.BOARD_PROBABILITIES.keys()]
        probs = list(self.BOARD_PROBABILITIES.values())
        return np.random.choice(items, p=probs, size=(6, 6))
    
    def reset(self):
        """重置遊戲狀態"""
        self.memory = None
        self.memory_time = None
        self.random_item = None
        
        state = self._generate_random_state()
        self.last_state = state
        self.next_item = state[0]
        self.state_matrix = state[1:].reshape(6, 6)
        self.time_matrix = np.zeros((6, 6), dtype=int)
        self.game_score = 0
        self.last_game_score = 0
        self._reload_time_matrix(state)
        
        return state
    
    def get_valid_actions(self, state, block_swap=False):
        """獲取有效的動作掩碼"""
        mask = np.zeros(36)  # 包括交換動作(0)
        board, next_item = self._split_state(state)
        board_flatten = board.flatten()
        
        # 決定有效的放置位置
        if next_item == self.ITEMS["bot"]:
            # 機器人可以放在非空的位置
            mask[(board_flatten != 0)] = 1
        else:
            # 其他物品可以放在空位置或寶藏位置
            valid_cells = (board_flatten == 0) | (board_flatten == self.ITEMS["treasure"]) | (board_flatten == self.ITEMS["Ltreasure"])
            mask[valid_cells] = 1
        
        # 設置swap動作的有效性
        if block_swap:
            mask[0] = 0
        else:
            mask[0] = 1
            
        return mask
    
    def is_game_over(self, state):
        """檢查遊戲是否結束"""
        valid_mask = self.get_valid_actions(state)
        return sum(valid_mask) == 1  # 只有swap動作可用
    
    def display_board(self, state):
        """在控制台显示游戏状态"""
        board, next_item = self._split_state(state)
        print("下一个物品:", self.ITEM_ICONS[str(next_item)])
        
        # 显示带有动作索引的棋盘
        for i in range(36):
            row, col = divmod(i, 6)
            print(f"{i}:".rjust(3), end=" ")
            print(self.ITEM_ICONS[str(board[row, col])].rjust(1), end=" ")
            if col == 5:
                print()  # 换行
    
    def _create_state_from_board_and_item(self, board, item):
        """從棋盤和物品創建狀態向量"""
        state = np.zeros(37, dtype=int)
        state[0] = int(item)
        state[1:] = board.flatten()
        return state
    
    def _split_state(self, state):
        """將狀態向量分割為棋盤和物品"""
        board = state[1:].reshape(6, 6)
        item = state[0]
        return board, item
    
    def _reload_time_matrix(self, state):
        """重新載入時間矩陣"""
        board, _ = self._split_state(state)
        time = 1
        
        for i in range(36):
            row, col = divmod(i, 6)
            if board[row, col] != 0:
                self.time_matrix[row, col] = time
                time += 1
    
    def _update_time_for_new_item(self, row, col):
        """更新時間矩陣以添加新物品"""
        # 增加所有現有物品的時間
        for i in range(36):
            r, c = divmod(i, 6)
            if self.time_matrix[r, c] > 0:
                self.time_matrix[r, c] += 1
        
        # 設置新物品的時間為1
        self.time_matrix[row, col] = 1
    
    def _fix_unknown_state(self, state):
        """修復狀態中的未知物品"""
        board, item = self._split_state(state)
        
        # 查找所有未知位置
        unknown_positions = []
        for row in range(6):
            for col in range(6):
                if board[row, col] == self.ITEMS["unknown"]:
                    unknown_positions.append((row, col))
        
        # 修復未知位置
        for row, col in unknown_positions:
            # 計算目標熊的數量
            target_bear_count = np.count_nonzero(board == self.ITEMS["bear"])
            target_nbear_count = np.count_nonzero(board == self.ITEMS["Nbear"])
            
            # 計算當前熊的數量
            current_bear_count = np.count_nonzero(self.state_matrix == self.ITEMS["bear"])
            current_nbear_count = np.count_nonzero(self.state_matrix == self.ITEMS["Nbear"])
            
            # 根據熊的數量差異確定未知物品類型
            if target_bear_count != current_bear_count:
                board[row, col] = self.ITEMS["bear"]
            elif target_nbear_count != current_nbear_count:
                board[row, col] = self.ITEMS["Nbear"]
            else:
                board[row, col] = self.state_matrix[row, col]
        
        return self._create_state_from_board_and_item(board, item)
    
    def _try_match_bear_movement(self, prev_state, new_state):
        """嘗試匹配熊的移動，並更新時間矩陣"""
        fixed_state = self._fix_unknown_state(new_state)
        prev_board = prev_state[1:].reshape(6, 6)
        new_board = fixed_state[1:].reshape(6, 6)
        
        # 跟踪變化的位置
        bear_changes_prev = set()
        bear_changes_new = set()
        
        # 尋找變化的位置
        for i in range(36):
            row, col = divmod(i, 6)
            if prev_board[row, col] != new_board[row, col]:
                if prev_board[row, col] in {self.ITEMS["bear"], self.ITEMS["Nbear"]}:
                    bear_changes_prev.add((row, col))
                if new_board[row, col] in {self.ITEMS["bear"], self.ITEMS["Nbear"]}:
                    bear_changes_new.add((row, col))
        
        # 如果沒有變化或變化不匹配，重新加載時間矩陣
        if len(bear_changes_prev) != len(bear_changes_new) or (not bear_changes_prev and not bear_changes_new):
            if bear_changes_prev or bear_changes_new:
                self._reload_time_matrix(fixed_state)
            return
        
        # 更新熊的時間
        for r1, c1 in bear_changes_prev:
            bear_type = prev_board[r1, c1]
            for r2, c2 in bear_changes_new:
                if new_board[r2, c2] == bear_type:
                    bear_time = self.time_matrix[r1, c1]
                    self.time_matrix[r2, c2] = bear_time
                    self.time_matrix[r1, c1] = 0
                    bear_changes_new.remove((r2, c2))
                    break
    
    def next_state(self, current_state, action):
        """計算給定動作後的下一個狀態"""
        valid_mask = self.get_valid_actions(current_state)
        board, item = self._split_state(current_state)
        
        # 檢查是否有有效動作
        if sum(valid_mask) == 1:  # 只有swap動作可用
            print("No valid action")
            return None
        
        # 處理返回上一狀態的動作
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
            # 保存當前狀態作為回退點
            self.random_item = None
            self.memory = current_state.copy()
            self.memory_time = self.time_matrix.copy()
            self.last_game_score = self.game_score
        
        # 檢查動作是否有效
        if valid_mask[action] == 0:
            return current_state
        
        # 嘗試匹配熊的移動
        self._try_match_bear_movement(self.last_state, current_state)
        self.last_state = current_state
        self.next_item = current_state[0]
        self.state_matrix = current_state[1:].reshape(6, 6)
        
        # 處理交換物品的特殊動作
        if action == 0:
            return self._handle_swap_action(board, item)
        
        # 處理一般放置動作
        self.game_score += 1
        row, col = divmod(action, 6)
        
        # 處理寶藏
        if board[row, col] in {self.ITEMS["treasure"], self.ITEMS["Ltreasure"]}:
            board[row, col] = 0
            self.time_matrix[row, col] = 0
            return self._create_state_from_board_and_item(board, item)
        
        # 處理機器人放置
        if item == self.ITEMS["bot"]:
            board = self._handle_bot_placement(board, row, col)
        else:
            board[row, col] = item
            
        self._update_time_for_new_item(row, col)
        
        # 更新連接的元素
        if 0 < board[row, col] < 9:
            board = self._update_connected_elements(board, row, col)
        
        # 更新水晶
        if board[row, col] == self.ITEMS["crystal"]:
            board = self._update_crystal_placement(board, row, col)
        
        # 處理熊的合併和移動
        board = self._check_bear_merge(board)
        board = self._update_bear_movement(board, self.ITEMS["bear"])
        board = self._update_bear_movement(board, self.ITEMS["Nbear"])
        
        # 更新遊戲狀態
        self.state_matrix = board
        if self.random_item is None:
            self.next_item = self._get_random_item()
        else:
            self.next_item = self.random_item
            
        self.last_state = self._create_state_from_board_and_item(self.state_matrix, self.next_item)
        return self.last_state
    
    def _handle_swap_action(self, board, item):
        """處理交換物品的特殊動作"""
        # 交換存儲的物品和當前物品
        store_item = board[0][0]
        board[0][0] = item
        new_item = int(store_item)
        
        # 如果交換空物品，則生成新物品
        if new_item == self.ITEMS["empty"]:
            new_item = self._get_random_item()
            
        self.state_matrix = board
        self.next_item = new_item
        self.last_state = self._create_state_from_board_and_item(board, new_item)
        return self.last_state
    
    def _handle_bot_placement(self, board, row, col):
        """處理機器人放置"""
        if board[row, col] in {self.ITEMS["bear"], self.ITEMS["Nbear"]}:
            board[row, col] = self.ITEMS["tombstone"]
        elif board[row, col] == self.ITEMS["mountain"]:
            board[row, col] = self.ITEMS["treasure"]
        else:
            board[row, col] = self.ITEMS["empty"]
        return board
    
    def _find_connected_elements(self, matrix, start_row, start_col):
        """查找與給定位置相連的相同元素"""
        rows, cols = matrix.shape
        target_value = matrix[start_row, start_col]
        visited = set()
        result = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右
        queue = [(start_row, start_col)]
        visited.add((start_row, start_col))
        
        is_bear_type = target_value in (self.ITEMS["bear"], self.ITEMS["Nbear"])
        
        while queue:
            current_row, current_col = queue.pop(0)
            result.append((current_row, current_col))
            
            for dr, dc in directions:
                new_row, new_col = current_row + dr, current_col + dc
                
                # 不搜索左上角的特殊位置 (用於存儲)
                if (new_row != 0 or new_col != 0):
                    # 確保新位置在棋盤內並且未訪問過
                    if (0 <= new_row < rows and 
                        0 <= new_col < cols and 
                        (new_row, new_col) not in visited):
                        
                        # 根據物品類型決定連接條件
                        if is_bear_type:
                            if matrix[new_row, new_col] in (self.ITEMS["bear"], self.ITEMS["Nbear"]):
                                queue.append((new_row, new_col))
                                visited.add((new_row, new_col))
                        elif matrix[new_row, new_col] == target_value:
                            queue.append((new_row, new_col))
                            visited.add((new_row, new_col))
        
        return result
    
    def _update_connected_elements(self, matrix, start_row, start_col):
        """更新連接元素 - 當有3個或更多相同元素相連時，合併成更高級物品"""
        connected_list = self._find_connected_elements(matrix, start_row, start_col)
        item_id = matrix[start_row, start_col]
        
        # 三重城堡是最高級別，無法升級
        if item_id == self.ITEMS["Tcastle"]:
            return matrix
            
        # 处理升级逻辑
        while len(connected_list) >= 3:
            # 浮動城堡需要4个才能升级
            if item_id == self.ITEMS["Fcastle"] and len(connected_list) < 4:
                return matrix
                
            # 清除连接元素
            for r, c in connected_list:
                matrix[r, c] = 0
                self.time_matrix[r, c] = 0
                
            # 升级中心位置
            item_name = self.ITEM_NAMES[item_id]
            matrix[start_row, start_col] = self.ITEMS[self.UPGRADE_MAP[item_name]]
            self._update_time_for_new_item(start_row, start_col)
            
            # 检查升级后是否还能继续合并
            connected_list = self._find_connected_elements(matrix, start_row, start_col)
            item_id = matrix[start_row, start_col]
            
        return matrix
    
    def _update_crystal_placement(self, matrix, row, col):
        """处理水晶放置 - 水晶会变成可以形成合并的最佳物品"""
        merge_candidates = []
        
        # 检查所有可能的物品类型
        merge_item_names = [
            "Fcastle", "castle", "mountain", "treasure", "mansion", 
            "cathedral", "church", "house", "hut", "tombstone", 
            "rock", "tree", "bush", "grass"
        ]
        
        for item_name in merge_item_names:
            matrix[row, col] = self.ITEMS[item_name]
            connected_list = self._find_connected_elements(matrix, row, col)
            
            # 浮動城堡需要4个才能形成合并
            if item_name == "Fcastle" and len(connected_list) >= 4:
                merge_candidates.append(item_name)
                continue
                
            # 其他物品需要3个才能形成合并
            if len(connected_list) >= 3:
                merge_candidates.append(item_name)
        
        # 如果没有能形成合并的物品，变成石头
        if not merge_candidates:
            matrix[row, col] = self.ITEMS["rock"]
        else:
            # 选择最优的合并物品（优先选择较低级别的物品）
            merge_item = merge_candidates[0]
            
            # 尝试找到最低级别的能合并的物品
            if merge_item not in ["grass", "tombstone", "rock"]:
                while self.DOWNGRADE_MAP[merge_item] in merge_candidates:
                    merge_item = self.DOWNGRADE_MAP[merge_item]
                    if merge_item in ["grass", "tombstone", "rock"]:
                        break
                        
            # 变成选定的物品并尝试合并
            matrix[row, col] = self.ITEMS[merge_item]
            self._update_connected_elements(matrix, row, col)
            
        return matrix
    
    def _check_bear_merge(self, matrix):
        """检查熊是否应该合并成墓碑（当熊无法移动时）"""
        visited = set()
        
        for i in range(35):  # 跳过左上角
            row, col = divmod(i+1, 6)
            
            # 检查是否是熊且尚未访问
            if (matrix[row, col] in {self.ITEMS["bear"], self.ITEMS["Nbear"]}) and (row, col) not in visited:
                connected_list = self._find_connected_elements(matrix, row, col)
                move_count = 0
                
                # 计算所有连接的熊的可能移动数
                for r, c in connected_list:
                    if matrix[r, c] == self.ITEMS["bear"]:
                        moves = self._get_bear_moves(matrix, r, c)
                        move_count += len(moves[2])
                    elif matrix[r, c] == self.ITEMS["Nbear"]:
                        moves = self._get_ninja_bear_moves(matrix, r, c)
                        move_count += len(moves[2])
                    visited.add((r, c))
                
                # 如果没有可能的移动，转换为墓碑
                if move_count == 0:
                    for r, c in connected_list:
                        matrix[r, c] = self.ITEMS["tombstone"]
        
        # 检查墓碑合并
        min_time = 99
        check_row = 0
        check_col = 0
        
        for i in range(36):
            row, col = divmod(i, 6)
            if matrix[row, col] == self.ITEMS["tombstone"]:
                connected_list = self._find_connected_elements(matrix, row, col)
                
                # 如果有足够的墓碑合并
                if len(connected_list) >= 3:
                    # 找到最早放置的墓碑
                    for r, c in connected_list:
                        if self.time_matrix[r, c] < min_time:
                            min_time = self.time_matrix[r, c]
                            check_row = r
                            check_col = c
                            
                    # 合并墓碑
                    self._update_connected_elements(matrix, check_row, check_col)
                    
        return matrix
    
    def _update_bear_movement(self, matrix, bear_type):
        """更新熊的移动"""
        visited = set()
        
        for i in range(35):  # 跳过左上角
            row, col = divmod(i+1, 6)
            
            # 使用行列坐标而不是转置
            if matrix[row, col] == bear_type and (row, col) not in visited:
                # 根据熊类型获取可能的移动
                if bear_type == self.ITEMS["bear"]:
                    _, _, valid_moves = self._get_bear_moves(matrix, row, col)
                elif bear_type == self.ITEMS["Nbear"]:
                    _, _, valid_moves = self._get_ninja_bear_moves(matrix, row, col)
                
                # 如果有有效移动，随机选择一个
                if valid_moves:
                    move_row, move_col = random.choice(valid_moves)
                    
                    # 执行移动
                    matrix[row, col] = self.ITEMS["empty"]
                    matrix[move_row, move_col] = bear_type
                    
                    # 更新时间矩阵
                    bear_time = self.time_matrix[row, col]
                    self.time_matrix[move_row, move_col] = bear_time
                    self.time_matrix[row, col] = 0
                    
                    # 标记新位置为已访问
                    visited.add((move_row, move_col))
                else:
                    visited.add((row, col))
                    
        return matrix
    
    def _get_bear_moves(self, matrix, bear_row, bear_col):
        """获取普通熊的可能移动位置 - 只能移动到相邻的空位置"""
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右
        valid_moves = []
        
        for dx, dy in directions:
            new_row, new_col = bear_row + dx, bear_col + dy
            
            # 检查新位置是否在棋盘内且为空
            if 0 <= new_row < 6 and 0 <= new_col < 6 and matrix[new_row, new_col] == 0:
                valid_moves.append((new_row, new_col))
                
        return (bear_row, bear_col, valid_moves)
    
    def _get_ninja_bear_moves(self, matrix, bear_row, bear_col):
        """获取忍者熊的可能移动位置 - 可以移动到任何空位置"""
        valid_moves = []
        
        # 检查棋盘上所有空位置
        for i in range(36):
            row, col = divmod(i, 6)
            if matrix[row, col] == 0:
                valid_moves.append((row, col))
                
        return (bear_row, bear_col, valid_moves)


# 测试游戏
def test_game():
    """运行游戏测试"""
    game = TripleTownSim(
        state = np.array([ 1,
            1 ,0 ,8 ,0 ,0 ,0 ,
            0 ,0 ,0 ,4 ,4 ,1 ,
            0 ,0 ,4 ,0 ,0 ,2 ,
            4 ,0 ,4 ,1 ,1 ,15,
            0 ,1 ,3 ,0 ,10,11,
            11,2 ,0 ,13,13,14]))
    
    state = game.last_state
    
    while True:
        game.display_board(state)
        action = int(input("请输入动作:"))
        state = game.next_state(game.last_state, action)
        print("得分:", game.game_score)
        print("=============================================================")
        
        if state is None:
            print("游戏结束")
            break


# 是否运行测试
# if __name__ == "__main__":
    # test_game()