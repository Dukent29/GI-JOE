"""
Snake AI
Made with PyGame and PyTorch
"""

import pygame
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


# --- Hyperparamètres ---
difficulty = 25  # Vitesse du jeu
max_memory_size = 100000
batch_size = 1000
gamma = 0.9  # Facteur d'escompte
epsilon = 1  # Taux d'exploration initial
epsilon_decay = 0.995
min_epsilon = 0.01
learning_rate = 0.001
target_update = 10  # Fréquence de mise à jour du réseau cible

# --- Dimensions de la fenêtre ---
frame_size_x = 720
frame_size_y = 480

# --- Initialisation de Pygame ---
pygame.init()
pygame.display.set_caption('Snake AI')
game_window = pygame.display.set_mode((frame_size_x, frame_size_y))

# --- Couleurs ---
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)

# --- Contrôleur FPS ---
fps_controller = pygame.time.Clock()

# --- Classes pour le DQN ---
class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=max_memory_size)
        self.epsilon = epsilon
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.n_games = 0
        self.reward_total = 0

    def build_model(self):
        # Réseau de neurones simple
        model = nn.Sequential(
            nn.Linear(11, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # Modifié de 4 à 3
        )
        return model

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_state(self, game):
        # État du jeu sous forme de tableau binaire
        head = game.snake_pos
        point_l = [head[0] - 10, head[1]]
        point_r = [head[0] + 10, head[1]]
        point_u = [head[0], head[1] - 10]
        point_d = [head[0], head[1] + 10]

        dir_l = game.direction == 'LEFT'
        dir_r = game.direction == 'RIGHT'
        dir_u = game.direction == 'UP'
        dir_d = game.direction == 'DOWN'

        state = [
            # Danger tout droit
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger à droite
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger à gauche
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Direction de mouvement
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Nourriture à gauche
            game.food_pos[0] < game.snake_pos[0],
            # Nourriture à droite
            game.food_pos[0] > game.snake_pos[0],
            # Nourriture en haut
            game.food_pos[1] < game.snake_pos[1],
            # Nourriture en bas
            game.food_pos[1] > game.snake_pos[1]
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 2)  # Modifié de 3 à 2
        state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        prediction = self.model(state0)
        return torch.argmax(prediction).item()

    def replay(self):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.tensor(np.array(states), dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.bool)

        # Q valeurs actuelles
        pred = self.model(states)
        target = pred.clone()
        for idx in range(len(dones)):
            Q_new = rewards[idx]
            if not dones[idx]:
                Q_new = rewards[idx] + gamma * torch.max(self.target_model(next_states[idx]))
            target[idx][actions[idx]] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()

    def train_short_memory(self, state, action, reward, next_state, done):
        state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        next_state0 = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)
        target = self.model(state0)
        Q_new = reward
        if not done:
            Q_new = reward + gamma * torch.max(self.target_model(next_state0))
        target[0][action] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(self.model(state0), target)
        loss.backward()
        self.optimizer.step()

# --- Classe du jeu ---
class SnakeGameAI:
    def __init__(self):
        self.reset()

    def reset(self):
        self.snake_pos = [frame_size_x / 2, frame_size_y / 2]
        self.snake_body = [self.snake_pos[:], [self.snake_pos[0] - 10, self.snake_pos[1]], [self.snake_pos[0] - 20, self.snake_pos[1]]]
        self.food_pos = [random.randrange(1, (frame_size_x // 10)) * 10,
                         random.randrange(1, (frame_size_y // 10)) * 10]
        self.food_spawn = True
        self.direction = 'RIGHT'
        self.score = 0
        self.frame_iteration = 0

    def is_collision(self, point=None):
        if point is None:
            point = self.snake_pos
        # Limites de la fenêtre
        if point[0] < 0 or point[0] > frame_size_x - 10 or point[1] < 0 or point[1] > frame_size_y - 10:
            return True
        # Collision avec le corps
        if point in self.snake_body[1:]:
            return True
        return False

    def play_step(self, action):
        self.frame_iteration += 1
        # Récupérer l'événement de fermeture de la fenêtre
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Mouvement
        self.move(action)
        self.snake_body.insert(0, self.snake_pos[:])

        # Récompense
        reward = 0
        done = False

        if self.is_collision() or self.frame_iteration > 100 * len(self.snake_body):
            done = True
            reward = -10
            return reward, done, self.score

        if self.snake_pos == self.food_pos:
            self.score += 1
            reward = 10
            self.food_spawn = False
        else:
            self.snake_body.pop()

        if not self.food_spawn:
            self.food_pos = [random.randrange(1, (frame_size_x // 10)) * 10,
                             random.randrange(1, (frame_size_y // 10)) * 10]
        self.food_spawn = True

        # Mise à jour de l'interface
        self.update_ui(reward)
        fps_controller.tick(difficulty)

        return reward, done, self.score

    def move(self, action):
        clock_wise = ['RIGHT', 'DOWN', 'LEFT', 'UP']
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # Pas de changement
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # Tourne à droite
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # Tourne à gauche

        self.direction = new_dir

        if self.direction == 'RIGHT':
            self.snake_pos[0] += 10
        elif self.direction == 'LEFT':
            self.snake_pos[0] -= 10
        elif self.direction == 'UP':
            self.snake_pos[1] -= 10
        elif self.direction == 'DOWN':
            self.snake_pos[1] += 10

    def update_ui(self, reward):
        game_window.fill(black)

        for pos in self.snake_body:
            pygame.draw.rect(game_window, green, pygame.Rect(pos[0], pos[1], 10, 10))

        pygame.draw.rect(game_window, white, pygame.Rect(self.food_pos[0], self.food_pos[1], 10, 10))

        # Affichage des informations
        font = pygame.font.SysFont('consolas', 20)
        text_score = font.render('Score: ' + str(self.score), True, white)
        text_generation = font.render('Génération: ' + str(agent.n_games), True, white)
        text_reward_total = font.render('Récompense Totale: ' + str(agent.reward_total), True, white)

        game_window.blit(text_score, [0, 0])
        game_window.blit(text_generation, [0, 20])
        game_window.blit(text_reward_total, [0, 40])

        # Indicateur de récompense ou punition
        if reward > 0:
            pygame.draw.circle(game_window, green, (frame_size_x - 20, 20), 10)
        elif reward < 0:
            pygame.draw.circle(game_window, red, (frame_size_x - 20, 20), 10)

        pygame.display.flip()

# --- Fonction principale ---
def train():
    global agent
    agent = DQNAgent()
    game = SnakeGameAI()
    while True:
        # Récupérer l'état actuel
        state_old = agent.get_state(game)

        # Décider de l'action à prendre
        action = [0, 0, 0]
        move = agent.act(state_old)
        action[move] = 1

        # Effectuer l'action et obtenir la récompense
        reward, done, score = game.play_step(action)
        state_new = agent.get_state(game)

        # Entraîner la mémoire courte
        agent.train_short_memory(state_old, move, reward, state_new, done)

        # Stocker dans la mémoire
        agent.remember(state_old, move, reward, state_new, done)

        if done:
            # Entraîner la mémoire longue (replay)
            game.reset()
            agent.n_games += 1
            agent.reward_total += reward
            agent.replay()
            agent.epsilon = max(min_epsilon, agent.epsilon * epsilon_decay)
            if agent.n_games % target_update == 0:
                agent.update_target_model()

if __name__ == '__main__':
    train()
