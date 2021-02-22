from snake_dqn import gen_epsilon_greedy_policy, get_state, find_state_food

def game_shore_learn(param_state, state, action):
    '''
    Это учитель по обучению, который выставляет баллы в зависимости от выбранного хода от -1 до 1
    Выставляемые баллы:
    -1 - ход повлек смерть, ход либо в стену либо в тело змеи, за исключением кончика хвоста
    +0.5 - ход приближает нас к яблоку
    +0 - ход не приближает и не отдаляет нас от яблока
    +1 - съели яблоко
    -0.2 - ход привел нас к западне тела
    +0.4 - ход приближает нас к хвосту если мы в западне (либо самой хвостовой точке тела)
    -0.2 - ход отдаляет нас от яблока
    
    actions {DOWN = 2, LEFT = 3, RIGHT = 1, UP = 0}
    
    '''
    
    x_food, y_food = find_state_food(state)

    a = 1
    
    pass

def save_state_param(controller):
    
    param = {}
    
    # Grid
    grid_object = controller.grid
    #grid_pixels = grid_object.grid

    # Snake(s)
    snakes_array = controller.snakes
    snake_object1 = snakes_array[0]
    
    param['snake_body'] = snake_object1.body.copy()
    param['snake_head'] = snake_object1.head.copy()
    
    
    
    #a = 1
    
    return param

def q_learning(env, estimator, n_episode, ACTIONS, total_reward_episode, memory, replay_size,
        target_update=10, gamma=1.0, epsilon=0.1, epsilon_decay=.99):
    """
    Глубокое Q-обучение с применением DDQN и буфера воспроизведения опыта
    @param env: имя окружающей среды Gym
    @param estimator: объекта класса DQN
    @param replay_size: сколько примеров использовать при каждом
    обновлении модели
    @param target_update: через сколько эпизодов обновлять целевую сеть
    @param n_episode: количество эпизодов
    @param gamma: коэффициент обесценивания
    @param epsilon: параметр ε-жад­ной стратегии
    @param epsilon_decay: коэффициент затухания epsilon
    """
    n_action = len(ACTIONS)
    
    for episode in range(n_episode):
        if episode % target_update == 0:
            estimator.copy_target()
            
        policy = gen_epsilon_greedy_policy(estimator, epsilon, n_action)
        
        env.reset()
         
        #state = get_state(obs).view(image_size * image_size)
        
        is_done = False
        
        while not is_done:
            
            # Controller
            game_controller = env.controller
            
            param_state = save_state_param(game_controller)
            
            # Grid
            grid_object = game_controller.grid
            grid_pixels = grid_object.grid

            # Snake(s)
            snakes_array = game_controller.snakes
            snake_object1 = snakes_array[0]    
            
            if snake_object1 == None:
                is_done = True
            
            state = get_state(grid_pixels)
            
            action = policy(state)
            
            env.step(ACTIONS[action])
            
            snake_after = snakes_array[0]

            next_state = get_state(grid_pixels)

            new_reword = game_shore_learn(param_state, state, action)

            if snake_after == None: #Ход оказался последним
                is_done = True
            else:
                reward = 1
            
            total_reward_episode[episode] += reward
            
            
            memory.append((state, action, next_state, reward, is_done))
            
            reward = 0
            
            if is_done:
                break
            
            estimator.replay(memory, replay_size, gamma)
            
            #state = next_state
            
        print('Эпизод: {}, полное вознаграждение: {}, epsilon:{}'.
          format(episode, total_reward_episode[episode], epsilon))
        epsilon = max(epsilon * epsilon_decay, 0.01)
