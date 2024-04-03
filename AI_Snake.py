import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random 
import numpy as np
import os
from collections import deque
import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt 
from IPython import display as dp
import cv2
pygame.init()
font = pygame.font.Font('arial.ttf',25)

WIDTH, HEIGHT = 640, 480
FPS = 40
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

# Set up OpenCV video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('snake.mp4', fourcc, FPS, (WIDTH, HEIGHT))

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
 
Point = namedtuple('Point','x , y')

BLOCK_SIZE=20
SPEED = 40
WHITE = (255,255,255)
RED = (200,0,0)
BLUE1 = (0,0,255)
BLUE2 = (0,100,255)
BLACK = (0,0,0)

class Game:
    def __init__(self,w=640,h=480):
        self.w=w
        self.h=h
        #init display
        self.display = pygame.display.set_mode((self.w,self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        
        #init game state
        self.reset()
    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2,self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE,self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE),self.head.y)]
        self.score = 0
        self.food = None
        self._place__food()
        self.frame_iteration = 0
      

    def _place__food(self):
        x = random.randint(0,(self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0,(self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x,y)
        if(self.food in self.snake):
            self._place__food()


    def play_step(self,action):
        self.frame_iteration+=1
        for event in pygame.event.get():
            if(event.type == pygame.QUIT):
                pygame.quit()
                quit()
        self._move(action)
        self.snake.insert(0,self.head)
        reward = 0  # eat food: +10 , game over: -10 , else: 0
        game_over = False 
        if(self.is_collision() or self.frame_iteration > 100*len(self.snake) ):
            game_over=True
            reward = -10
            return reward,game_over,self.score
        if(self.head == self.food):
            self.score+=1
            reward=10
            self._place__food()
        else:
            self.snake.pop()
        
        self._update_ui()
        self.clock.tick(SPEED)
        
        return reward,game_over,self.score

    def _update_ui(self):
        global out
        self.display.fill(WHITE)
        for pt in self.snake:
            pygame.draw.rect(self.display,BLACK,pygame.Rect(pt.x,pt.y,BLOCK_SIZE,BLOCK_SIZE))
        pygame.draw.rect(self.display,RED,pygame.Rect(self.food.x,self.food.y,BLOCK_SIZE,BLOCK_SIZE))
        text = font.render("Score: "+str(self.score),True,BLACK)
        self.display.blit(text,[0,0])
        pygame.display.flip()
        # Capture Pygame screen as OpenCV image and write to video
        img = pygame.surfarray.array3d(self.display)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = img.swapaxes(0,1)
        out.write(img)
        if self.score > 60:
            cv2.imwrite('image{}.png'.format(self.frame_iteration), img)

    def _move(self,action):
        clock_wise = [Direction.RIGHT,Direction.DOWN,Direction.LEFT,Direction.UP]
        idx = clock_wise.index(self.direction)
        if np.array_equal(action,[1,0,0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action,[0,1,0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right Turn
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # Left Turn
        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if(self.direction == Direction.RIGHT):
            x+=BLOCK_SIZE
        elif(self.direction == Direction.LEFT):
            x-=BLOCK_SIZE
        elif(self.direction == Direction.DOWN):
            y+=BLOCK_SIZE
        elif(self.direction == Direction.UP):
            y-=BLOCK_SIZE
        self.head = Point(x,y)

    def pseudo_move(self,action):
        clock_wise = [Direction.RIGHT,Direction.DOWN,Direction.LEFT,Direction.UP]
        idx = clock_wise.index(self.direction)
        if np.array_equal(action,[1,0,0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action,[0,1,0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right Turn
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # Left Turn

        x = self.head.x
        y = self.head.y
        if(new_dir== Direction.RIGHT):
            x+=BLOCK_SIZE
        elif(new_dir == Direction.LEFT):
            x-=BLOCK_SIZE
        elif(new_dir == Direction.DOWN):
            y+=BLOCK_SIZE
        elif(new_dir == Direction.UP):
            y-=BLOCK_SIZE
        pseudo_head = Point(x,y)
        return pseudo_head

    def is_collision(self,pt=None):
        if(pt is None):
            pt = self.head
        #hit boundary
        if(pt.x>self.w-BLOCK_SIZE or pt.x<0 or pt.y>self.h - BLOCK_SIZE or pt.y<0):
            return True
        if(pt in self.snake[1:]):
            return True
        return False


class Linear_QNet(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.linear2 = nn.Linear(hidden_size,output_size)
        
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    def save(self, file_name='model.pth'):
        current_directory = os.getcwd()
        model_folder_path = '{}\{}'.format(current_directory, 'SnakeAI')
        file_name = os.path.join(model_folder_path,file_name)
        torch.save(self.state_dict(),file_name)

class QTrainer:
    def __init__(self,model,lr,gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimer = optim.Adam(model.parameters(),lr = self.lr)    
        self.criterion = nn.MSELoss()

    def train_step(self,state,action,reward,next_state,done):
        state = torch.tensor(state,dtype=torch.float)
        next_state = torch.tensor(next_state,dtype=torch.float)
        action = torch.tensor(action,dtype=torch.long)
        reward = torch.tensor(reward,dtype=torch.float)


        if(len(state.shape) == 1):
            state = torch.unsqueeze(state,0)
            next_state = torch.unsqueeze(next_state,0)
            action = torch.unsqueeze(action,0)
            reward = torch.unsqueeze(reward,0)
            done = (done, )
        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action).item()] = Q_new
        self.optimer.zero_grad()
        loss = self.criterion(target,pred)
        loss.backward()

        self.optimer.step()

class Agent:
    def __init__(self):
        self.n_game = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11,256,3) 
        self.trainer = QTrainer(self.model,lr=LR,gamma=self.gamma)

    def get_state(self,game):
        head = game.snake[0]
        point_l=Point(head.x - BLOCK_SIZE, head.y)
        point_r=Point(head.x + BLOCK_SIZE, head.y)
        point_u=Point(head.x, head.y - BLOCK_SIZE)
        point_d=Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger Straight
            (dir_u and game.is_collision(point_u))or
            (dir_d and game.is_collision(point_d))or
            (dir_l and game.is_collision(point_l))or
            (dir_r and game.is_collision(point_r)),

            # Danger right
            (dir_u and game.is_collision(point_r))or
            (dir_d and game.is_collision(point_l))or
            (dir_u and game.is_collision(point_u))or
            (dir_d and game.is_collision(point_d)),

            #Danger Left
            (dir_u and game.is_collision(point_r))or
            (dir_d and game.is_collision(point_l))or
            (dir_r and game.is_collision(point_u))or
            (dir_l and game.is_collision(point_d)),

            dir_l,
            dir_r,
            dir_u,
            dir_d,

            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y 
        ]
        return np.array(state,dtype=int)

    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))

    def train_long_memory(self):
        if (len(self.memory) > BATCH_SIZE):
            mini_sample = random.sample(self.memory,BATCH_SIZE)
        else:
            mini_sample = self.memory
        states,actions,rewards,next_states,dones = zip(*mini_sample)
        self.trainer.train_step(states,actions,rewards,next_states,dones)

    def train_short_memory(self,state,action,reward,next_state,done):
        self.trainer.train_step(state,action,reward,next_state,done)

    def get_action(self,state):
        self.epsilon = 80 - self.n_game
        final_move = [0,0,0]
        if(random.randint(0,200)<self.epsilon):
            move = random.randint(0,2)
            final_move[move]=1
        else:
            state0 = torch.tensor(state,dtype=torch.float)
            prediction = self.model(state0)# prediction by model 
            move = torch.argmax(prediction).item()
            final_move[move]=1 
        return final_move

    def get_move(self,state):
        self.epsilon = 80 - self.n_game
        final_move = [0,0,0]
        state0 = torch.tensor(state,dtype=torch.float)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        final_move[move]=1 
        return final_move
    
def plot(scores, mean_scores):
    dp.clear_output(wait=True)
    dp.display(plt.gcf())
    plt.clf()
    plt.title("Training...")
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1,scores[-1],str(scores[-1]))
    plt.text(len(mean_scores)-1,mean_scores[-1],str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)
    plt.savefig('Training_plot.png')

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = Game()
    while agent.n_game < 150:
        # Get Old state
        state_old = agent.get_state(game)
        # get move
        final_move = agent.get_action(state_old)
        # get move
        final_move = agent.get_move(state_old)
        for i in range(2):
        # code to be executed in each iteration
            pseudo_head = game.pseudo_move(final_move)
            if(pseudo_head.x>game.w-20 or pseudo_head.x<0 or pseudo_head.y>game.h - 20 or pseudo_head.y<0 or pseudo_head in game.snake[1:]):
                if final_move == [1,0,0]:
                    final_move = [0,1,0]
                elif final_move == [0,1,0]:
                    final_move = [0,0,1]
                elif final_move == [0,0,1]:
                    final_move = [1,0,0]
                else:
                    pass
            else:
                pass
        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old,final_move,reward,state_new,done)

        #remember
        agent.remember(state_old,final_move,reward,state_new,done)

        if done:
            # Train long memory,plot result
            game.reset()
            agent.n_game += 1
            agent.train_long_memory()
            if(score > reward): # new High score 
                reward = score
                agent.model.save()
            print('Game:',agent.n_game,'Score:',score,'Record:',record)
            
            plot_scores.append(score)
            total_score+=score
            mean_score = total_score / agent.n_game
            plot_mean_scores.append(mean_score)
            plot(plot_scores,plot_mean_scores)
            #plot.savefig('my_plot.png')
    out.release()

def play():
    agent = Agent()
    game = Game()
    #agent.model.load_state_dict(torch.load('C:\Workspace\old_documents\EDUCATION\design_of_AI\Game-main\model.pth'))
    current_directory = os.getcwd()
    agent.model.load_state_dict(torch.load('{}\\{}'.format(current_directory,'SnakeAI\model_final.pth')))
    while agent.n_game < 1:
        # Get Old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_move(state_old)
        for i in range(2):
        # code to be executed in each iteration
            pseudo_head = game.pseudo_move(final_move)
            if(pseudo_head.x>game.w-20 or pseudo_head.x<0 or pseudo_head.y>game.h - 20 or pseudo_head.y<0 or pseudo_head in game.snake[1:]):
                if final_move == [1,0,0]:
                    final_move = [0,1,0]
                elif final_move == [0,1,0]:
                    final_move = [0,0,1]
                elif final_move == [0,0,1]:
                    final_move = [1,0,0]
                else:
                    pass
            else:
                pass
        
        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        #remember
        agent.remember(state_old,final_move,reward,state_new,done)

        if done:
            agent.n_game += 1
    out.release()

if(__name__=="__main__"):
    #train()
    play()