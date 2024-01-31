import random
import numpy as np
import pandas as pd
import pygame

import gymnasium as gymnasium
from gymnasium import spaces

class MinesweeperEnv(gymnasium.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, width, height, n_mines,
                 rewards={'win':1, 'lose':-1, 'progress':0.3, 'guess':-0.3, 'no_progress' : -0.3}):
        super(MinesweeperEnv, self).__init__()


        self.nrows, self.ncols = width, height
        self.ntiles = self.nrows * self.ncols
        self.n_mines = n_mines
        self.rewards = rewards

        pygame.init()
        self.screen = pygame.display.set_mode((self.nrows * 40, self.ncols * 40))  

        self.grid = self.init_grid()
        self.board = self.get_board()
        self.state, self.state_im = self.init_state()
        self.n_clicks = 0
        self.n_progress = 0
        self.n_noprogress = 0
        self.n_wins = 0


        self.action_space = spaces.Discrete(self.ntiles)
        self.observation_space = spaces.Box(low=-0.125, high=1, shape=(self.nrows, self.ncols,1), dtype=np.float16)

        self.reset()

    def init_grid(self):
        board = np.zeros((self.nrows, self.ncols), dtype='object')
        mines = self.n_mines

        while mines > 0:
            row, col = random.randint(0, self.nrows-1), random.randint(0, self.ncols-1)
            if board[row][col] != 'B':
                board[row][col] = 'B'
                mines -= 1

        return board

    def get_neighbors(self, coord):
        x, y = coord[0], coord[1]

        neighbors = []
        for col in range(y - 1, y + 2):
            for row in range(x - 1, x + 2):
                if ((x != row or y != col) and
                    (0 <= col < self.ncols) and
                    (0 <= row < self.nrows)):
                    cell_value = self.grid[row, col]
                    if cell_value != 'B':
                        # Convert numeric zeros to string '0' for consistency
                        cell_value = '0'
                    neighbors.append(cell_value)

        return np.array(neighbors)

    def count_bombs(self, coord):
        neighbors = self.get_neighbors(coord)
        return np.sum(neighbors=='B')

    def get_board(self):
        board = self.grid.copy()

        coords = []
        for x in range(self.nrows):
            for y in range(self.ncols):
                if self.grid[x,y] != 'B':
                    coords.append((x,y))

        for coord in coords:
            board[coord] = self.count_bombs(coord)

        return board

    def get_state_im(self, state):
        state_im = [t['value'] for t in state]
        
        state_im = np.reshape(state_im, (self.nrows, self.ncols, 1)).astype(object)

        state_im[state_im=='U'] = -1
        state_im[state_im=='B'] = -2

        state_im = state_im.astype(np.int8) / 8
        state_im = state_im.astype(np.float16) 
        return state_im

    def init_state(self):
        unsolved_array = np.full((self.nrows, self.ncols), 'U', dtype='object')

        state = []
        for (x, y), value in np.ndenumerate(unsolved_array):
            state.append({'coord': (x, y), 'value':value})

        state_im = self.get_state_im(state)

        return state, state_im
  
    def get_action(self):
        board = self.state_im.reshape(1,self.ntiles)
        unsolved = [i for i, x in enumerate(board[0]) if x==-0.125]

        return np.random.choice(unsolved)
    
    def click(self, action_index):
        coord = self.state[action_index]['coord']
        value = self.board[coord]

        # ensure first move is not a bomb
        if (value == 'B') and (self.n_clicks == 0):
            grid = self.grid.reshape(1, self.ntiles)
            move = np.random.choice(np.nonzero(grid!='B')[1])
            coord = self.state[move]['coord']
            value = self.board[coord]
            self.state[move]['value'] = value
        else:
            # make state equal to board at given coordinates
            self.state[action_index]['value'] = value

        # reveal all neighbors if value is 0
        if value == 0.0:
            self.reveal_neighbors(coord, clicked_tiles=[])

        self.n_clicks += 1

    def reveal_neighbors(self, coord, clicked_tiles):
        processed = clicked_tiles
        state_df = pd.DataFrame(self.state)
        x,y = coord[0], coord[1]

        for col in range(y-1, y+2):
            for row in range(x-1, x+2):
                if ((x != row or y != col) and
                    (0 <= col < self.ncols) and
                    (0 <= row < self.nrows) and
                    ((row, col) not in processed)):

                    # prevent redundancy for adjacent zeros
                    processed.append((row,col))

                    index = state_df.index[state_df['coord'] == (row,col)].tolist()[0]

                    self.state[index]['value'] = self.board[row, col]

                    # recursion in case neighbors are also 0
                    if self.board[row, col] == 0.0:
                        self.reveal_neighbors((row, col), clicked_tiles=processed)

    def reset(self, **kwargs):

        self.n_clicks = 0
        self.n_progress = 0
        self.n_noprogress = 0
        self.grid = self.init_grid()
        self.board = self.get_board()
        self.state, self.state_im = self.init_state()

        return self.state_im, {}

    def step(self, action_index):
        done = False
        truncated = False
        coords = self.state[action_index]['coord']

        current_state = self.state_im

        # get neighbors before action
        neighbors = self.get_neighbors(coords)

        self.click(action_index)

        # update state image
        new_state_im = self.get_state_im(self.state)
        self.state_im = new_state_im

        all_neighbors_unsolved = all(t == -0.125 for t in neighbors)

        info = {"is_success": False}

        if self.state[action_index]['value']=='B': # if lose
            reward = self.rewards['lose']
            done = True
            info["is_success"] = False

        elif np.sum(new_state_im==-0.125) == self.n_mines: # if win
            reward = self.rewards['win']
            done = True
            self.n_progress += 1
            self.n_wins += 1
            info["is_success"] = True
            

        elif np.sum(self.state_im == -0.125) == np.sum(current_state == -0.125):
            reward = self.rewards['no_progress']
            self.n_noprogress += 1
            if (self.n_noprogress == 20):
                truncated = True

        else: # if progress
            if all_neighbors_unsolved: # if guess (all neighbors are unsolved)
                reward = self.rewards['guess']

            else:
                reward = self.rewards['progress']
                self.n_progress += 1 # track n of non-isoloated click

        return self.state_im, reward, done, truncated, info
    

    def color_state(self, value):
        color_map = {
            -1: (255, 255, 255),  # Unrevealed - white
            0: (128, 128, 128),   # Empty - slategrey
            1: (0, 64, 255),      # 1 - blue
            2: (0, 128, 0),       # 2 - green
            3: (255, 0, 0),       # 3 - red
            4: (0, 0, 255),       # 4 - midnightblue
            5: (179, 45, 0),      # 5 - brown
            6: (0, 255, 255),     # 6 - aquamarine
            7: (0, 0, 0),         # 7 - black
            8: (192, 192, 192),   # 8 - silver
        }
        return color_map.get(value, (255, 0, 255))
    
    def draw_state_pygame(self, state_im):
        for x in range(self.nrows):
            for y in range(self.ncols):

                value = int(state_im[x, y] * 8)
                color = self.color_state(value)
                pygame.draw.rect(self.screen, color, pygame.Rect(x*40, y*40, 40, 40))
        pygame.display.flip()
   
    def render(self, mode='human', close=False):
        if mode == 'human':
            if not hasattr(self, 'screen'):
                pygame.init()
                self.screen = pygame.display.set_mode((self.nrows * 40, self.ncols * 40))
            
            self.draw_state_pygame(self.state_im)
            # running = True
            # while running:
            #     for event in pygame.event.get():
            #         if event.type == pygame.QUIT:
            #             running = False
            #         elif event.type == pygame.KEYDOWN:
            #             running = False

            #     # Update the display
            #     pygame.display.flip()

            # # Close the Pygame window if close is True
            # if close:
            #     pygame.quit()
        else:
            super(MinesweeperEnv, self).render(mode=mode)