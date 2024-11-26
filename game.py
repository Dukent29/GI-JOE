"""
Snake AI with Persistent Memory and Model
Made with PyGame and PyTorch
"""

import pygame
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import pickle

# --- Hyperparamètres ---
difficulty = 50  # Vitesse du jeu
max_memory_size = 100000
batch_size = 1000
gamma = 0.9  # Facteur d'escompte
epsilon_start = 1  # Taux d'exploration initial
epsilon_decay = 0.995
min_epsilon = 0.01
learning_rate = 0.001
target_update = 10  # Fréquence de mise à jour du réseau cible

# --- Dimensions de la fenêtre ---
single_frame_size_x = 720
single_frame_size_y = 480
total_frame_size_x = single_frame_size_x * 2  # Pour deux jeux côte à côte
total_frame_size_y = single_frame_size_y

# --- Initialisation de Pygame ---
pygame.init()
pygame.display.set_caption('Snake AI - Generation Training')
game_window = pygame.display.set_mode((total_frame_size_x, total_frame_size_y))

# --- Couleurs ---
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)

# --- Contrôleur FPS ---
fps_controller = pygame.time.Clock()

# --- Classes pour le DQN ---
class DQNAgent:
    def __init__(self, model=None, memory=None, epsilon=epsilon_start):
        self.memory = deque(maxlen=max_memory_size) if memory is None else memory
        self.epsilon = epsilon
        self.model = self.build_model() if model is None else model
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
            nn.Linear(256, 3)
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
            return random.randint(0, 2)
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
    def __init__(self, agent, surface, offset_x=0):
        self.agent = agent
        self.surface = surface
        self.offset_x = offset_x
        self.reset()

    def reset(self):
        self.snake_pos = [single_frame_size_x / 2, single_frame_size_y / 2]
        self.snake_body = [self.snake_pos[:],
                           [self.snake_pos[0] - 10, self.snake_pos[1]],
                           [self.snake_pos[0] - 20, self.snake_pos[1]]]
        self.food_spawn = True
        self.direction = 'RIGHT'
        self.score = 0
        self.frame_iteration = 0
        self.time_limit = 20  # Temps limite par partie, en secondes
        self.start_time = pygame.time.get_ticks()  # Temps de début en millisecondes

        # Générer une position de nourriture qui ne chevauche pas le serpent
        while True:
            self.food_pos = [random.randrange(1, (single_frame_size_x // 10)) * 10,
                             random.randrange(1, (single_frame_size_y // 10)) * 10]
            if self.food_pos not in self.snake_body:
                break

    def is_collision(self, point=None):
        if point is None:
            point = self.snake_pos
        if point[0] < 0 or point[0] > single_frame_size_x - 10 or point[1] < 0 or point[1] > single_frame_size_y - 10:
            return True
        if point in self.snake_body[1:]:
            return True
        return False

    def play_step(self):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Sauvegarder avant de quitter
                pygame.quit()
                sys.exit()

        state_old = self.agent.get_state(self)
        action = [0, 0, 0]
        move = self.agent.act(state_old)
        action[move] = 1

        # Calcul de la distance avant le mouvement
        distance_old = np.linalg.norm(np.array(self.snake_pos) - np.array(self.food_pos))

        self.move(action)
        self.snake_body.insert(0, self.snake_pos[:])

        reward = 0  # Initialiser la récompense à 0
        done = False

        # Calcul de la distance après le mouvement
        distance_new = np.linalg.norm(np.array(self.snake_pos) - np.array(self.food_pos))

        # Récompense ou punition basée sur le déplacement vers/au-delà de la nourriture
        delta_distance = distance_old - distance_new
        reward += delta_distance * 0.1  # Récompense proportionnelle

        # Gestion du chronomètre
        current_time = pygame.time.get_ticks()  # Temps actuel en millisecondes
        elapsed_time = (current_time - self.start_time) / 1000  # Temps écoulé en secondes
        time_remaining = max(0, self.time_limit - elapsed_time)  # Temps restant en secondes

        if time_remaining == 0:
            done = True
            reward -= 20  # Pénalité pour le temps écoulé
            self.agent.reward_total += reward  # Mettre à jour la récompense totale
            print(f"Time's up! Reward: {reward}")
            self.update_ui(reward, time_remaining)
            return reward, done, self.score

        if self.is_collision():
            done = True
            reward -= 100  # Pénalité pour la mort
            self.agent.reward_total += reward  # Mettre à jour la récompense totale
            print(f"Collision! Reward: {reward}")
            self.update_ui(reward, time_remaining)
            return reward, done, self.score

        if self.snake_pos == self.food_pos:
            self.score += 1
            reward += 50  # Récompense pour avoir mangé la nourriture
            self.food_spawn = False
            print(f"Ate food! Reward: {reward}")
        else:
            self.snake_body.pop()

        if not self.food_spawn:
            # Assurez-vous que la nourriture ne spawn pas sur le corps du serpent
            while True:
                new_food_pos = [random.randrange(1, (single_frame_size_x // 10)) * 10,
                                random.randrange(1, (single_frame_size_y // 10)) * 10]
                if new_food_pos not in self.snake_body:
                    self.food_pos = new_food_pos
                    break
            self.food_spawn = True

        self.agent.reward_total += reward  # Mettre à jour la récompense totale

        # Entraînement à court terme
        state_new = self.agent.get_state(self)
        self.agent.train_short_memory(state_old, move, reward, state_new, done)
        self.agent.remember(state_old, move, reward, state_new, done)

        self.update_ui(reward, time_remaining)
        fps_controller.tick(difficulty)
        return reward, done, self.score

    def move(self, action):
        clock_wise = ['RIGHT', 'DOWN', 'LEFT', 'UP']
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # Aller tout droit
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # Tourner à droite
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # Tourner à gauche

        self.direction = new_dir

        if self.direction == 'RIGHT':
            self.snake_pos[0] += 10
        elif self.direction == 'LEFT':
            self.snake_pos[0] -= 10
        elif self.direction == 'UP':
            self.snake_pos[1] -= 10
        elif self.direction == 'DOWN':
            self.snake_pos[1] += 10

    def update_ui(self, reward, time_remaining=None):
        # Dessiner uniquement sur la surface assignée
        self.surface.fill(black)
        for pos in self.snake_body:
            pygame.draw.rect(self.surface, green, pygame.Rect(pos[0], pos[1], 10, 10))
        pygame.draw.rect(self.surface, white, pygame.Rect(self.food_pos[0], self.food_pos[1], 10, 10))

        font = pygame.font.SysFont('consolas', 20)
        text_score = font.render('Score: ' + str(self.score), True, white)
        text_generation = font.render('Génération: ' + str(self.agent.n_games), True, white)
        text_reward_total = font.render('Récompense Totale: ' + str(round(self.agent.reward_total, 2)), True, white)

        self.surface.blit(text_score, [0, 0])
        self.surface.blit(text_generation, [0, 20])
        self.surface.blit(text_reward_total, [0, 40])

        if time_remaining is not None:
            text_time = font.render('Temps restant: ' + str(int(time_remaining)) + 's', True, white)
            self.surface.blit(text_time, [0, 60])

        # Indicateur de récompense/punition
        if reward > 0:
            pygame.draw.circle(self.surface, green, (single_frame_size_x - 20, 20), 10)
        elif reward < 0:
            pygame.draw.circle(self.surface, red, (single_frame_size_x - 20, 20), 10)

        pygame.display.update()


# --- Fonction principale ---
def train():
    # Charger le modèle et la mémoire du meilleur agent précédent
    try:
        with open("best_agent.pkl", "rb") as f:
            best_agent_data = pickle.load(f)
        print("Meilleur agent chargé depuis best_agent.pkl")
        best_model_state_dict = best_agent_data['model_state_dict']
        best_memory = best_agent_data['memory']
        best_epsilon = best_agent_data['epsilon']
    except FileNotFoundError:
        print("Aucun agent précédent trouvé, création d'un nouvel agent.")
        best_model_state_dict = None
        best_memory = None
        best_epsilon = epsilon_start

    generation = 0

    while True:
        try:
            generation += 1
            print(f"\n--- Génération {generation} ---")

            # Initialiser les deux agents avec le meilleur modèle
            agent1 = DQNAgent(model=None, memory=None, epsilon=best_epsilon)
            agent2 = DQNAgent(model=None, memory=None, epsilon=best_epsilon)

            if best_model_state_dict is not None:
                agent1.model.load_state_dict(best_model_state_dict)
                agent1.update_target_model()
                agent2.model.load_state_dict(best_model_state_dict)
                agent2.update_target_model()

            if best_memory is not None:
                agent1.memory = deque(best_memory, maxlen=max_memory_size)
                agent2.memory = deque(best_memory, maxlen=max_memory_size)

            # Créer deux surfaces pour les deux jeux
            surface1 = game_window.subsurface((0, 0, single_frame_size_x, single_frame_size_y))
            surface2 = game_window.subsurface((single_frame_size_x, 0, single_frame_size_x, single_frame_size_y))

            # Créer deux jeux pour les deux agents
            game1 = SnakeGameAI(agent1, surface1, offset_x=0)
            game2 = SnakeGameAI(agent2, surface2, offset_x=single_frame_size_x)

            # Réinitialiser les récompenses totales pour chaque agent
            agent1.reward_total = 0
            agent2.reward_total = 0

            # Jouer les deux parties jusqu'à ce qu'elles se terminent
            done1 = False
            done2 = False
            while not done1 or not done2:
                if not done1:
                    reward1, done1, score1 = game1.play_step()
                if not done2:
                    reward2, done2, score2 = game2.play_step()

            # Entraîner les agents
            agent1.replay()
            agent2.replay()

            # Déterminer le meilleur agent
            if agent1.reward_total >= agent2.reward_total:
                best_agent = agent1
                print(f"L'agent 1 est le meilleur avec une récompense totale de {agent1.reward_total}")
            else:
                best_agent = agent2
                print(f"L'agent 2 est le meilleur avec une récompense totale de {agent2.reward_total}")

            # Mettre à jour le meilleur modèle et la mémoire
            best_model_state_dict = best_agent.model.state_dict()
            best_memory = best_agent.memory
            best_epsilon = max(min_epsilon, best_agent.epsilon * epsilon_decay)

            # Sauvegarder le meilleur agent
            with open("best_agent.pkl", "wb") as f:
                pickle.dump({
                    'model_state_dict': best_model_state_dict,
                    'memory': best_memory,
                    'epsilon': best_epsilon
                }, f)
            print("Meilleur agent sauvegardé dans best_agent.pkl")

        except KeyboardInterrupt:
            print("Interruption détectée, sauvegarde en cours...")
            # Sauvegarder le meilleur agent
            with open("best_agent.pkl", "wb") as f:
                pickle.dump({
                    'model_state_dict': best_model_state_dict,
                    'memory': best_memory,
                    'epsilon': best_epsilon
                }, f)
            print("Meilleur agent sauvegardé. Programme terminé.")
            break


if __name__ == '__main__':
    train()
