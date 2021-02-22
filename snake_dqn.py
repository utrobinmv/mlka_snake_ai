import torch
import copy
import random
import numpy as np

grid_change = {}

class DQN():
    def __init__(self, n_state, n_action, n_hidden, lr=0.05):
        '''
        Инициализирует нейронную сеть
        '''
        self.criterion = torch.nn.MSELoss()
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_state, n_hidden[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden[0], n_hidden[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden[1], n_action)
        )
        
        self.model_target = copy.deepcopy(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        
    def copy_target(self):
        '''
        Производит копирование весов обученной сети от model в model_target
        '''
        self.model_target.load_state_dict(self.model.state_dict())    
        
    def predict(self, s):
        """
        Вычисляет значения Q-функции состояния для всех действий,
        применяя обученную модель
        @param s: входное состояние
        @return: значения Q для всех действий
        """
        with torch.no_grad():
            state = torch.Tensor(s)
            model_state = self.model(state)
            return model_state
    
    def target_predict(self, s):
        """
        Вычисляет значения Q-функции состояния для всех действий
        с помощью целевой сети
        @param s: входное состояние212  Глубокие Q-сети в действии
        @return: целевые ценности состояния для всех действий
        """
        with torch.no_grad():
            return self.model_target(torch.Tensor(s))
    
    def replay(self, memory, replay_size, gamma):
        """
        Воспроизведение опыта с целевой сетью    
        @param memory: буфер воспроизведения опыта
        @param replay_size: количество выбираемых из буфера примеров при каждом обновлении модели
        @param gamma: коэффициент обесценивания
        """
        if len(memory) >= replay_size:
            replay_data = random.sample(memory, replay_size)
            states = []
            td_targets = []
            for state, action, next_state, reward, is_done in replay_data:
                
                #Приведем к одномерному массиву
                state = torch.Tensor(state).reshape(-1)
                next_state = torch.Tensor(next_state).reshape(-1)
                
                states.append(state.tolist())
                q_values = self.predict(state).tolist()
                if is_done:
                    q_values[action] = reward
                else:
                    q_values_next = self.target_predict(next_state).detach()
                    q_values[action] = reward + gamma * torch.max(q_values_next).item()
                td_targets.append(q_values)
            self.update(states, td_targets)
            
    def update(self, s, y):
        """
        Обновляет веса DQN, получив обучающий пример
        @param s: состояние
        @param y: целевое значение
        """
        y_pred = self.model(torch.Tensor(s))
        loss = self.criterion(y_pred, torch.autograd.Variable(torch.Tensor(y)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()    

            
def gen_epsilon_greedy_policy(estimator, epsilon, n_action):
    def policy_function(state):
        if random.random() < epsilon:
            action = random.randint(0, n_action - 1)
            return action
        else:
            state = torch.Tensor(state).reshape(-1)
            q_values = estimator.predict(state)
            action = torch.argmax(q_values).item()
            return action
    return policy_function 


def sum_np_3rd(BODY_COLOR):
    return (BODY_COLOR[0:1] + (255 * BODY_COLOR[1:2].astype(np.int32)) + (255*255*BODY_COLOR[2:3].astype(np.int32))).sum()

def find_state_food(state):
    
    FOOD_COLOR = grid_change['FOOD_COLOR']
    
    n = np.where(state.numpy() == FOOD_COLOR)
    
    x = n[1][0]
    y = n[0][0]
    
    return x,y

def np_array_change_value(a):
    d = grid_change
    b = a.copy()
    for k, v in d.items():
        b[a==k] = v
    return b

def convert_game_grid(grid_pixels):
    body = grid_pixels[:,:,0:1] + (255 * grid_pixels[:,:,1:2].astype(np.float32)) + (255*255*grid_pixels[:,:,2:3].astype(np.float32))
    body = body.reshape(grid_pixels.shape[:2])
    
    body_new = np_array_change_value(body)
    return body_new 


def init_state(controller):
    global grid_change
    
    grid_object = controller.grid
    
    BODY_COLOR = sum_np_3rd(grid_object.BODY_COLOR)
    HEAD_COLOR = sum_np_3rd(np.array(controller.snakes[0].head_color))
    FOOD_COLOR = sum_np_3rd(grid_object.FOOD_COLOR)
    SPACE_COLOR = sum_np_3rd(grid_object.SPACE_COLOR)
    
    grid_change = {BODY_COLOR: 0.3, HEAD_COLOR: 0.2, FOOD_COLOR: 0.1, SPACE_COLOR: 0}
    
    grid_change['FOOD_COLOR'] = 0.1


def get_state(grid_pixels):
    convert_grid = convert_game_grid(grid_pixels)
    state = torch.from_numpy(convert_grid)

    return state

