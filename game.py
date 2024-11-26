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
difficulty = 30  # Réduire la vitesse du jeu pour améliorer les performances
max_memory_size = 50000  # Réduire la taille de la mémoire de replay
batch_size = 64  # Réduire la taille des batchs
gamma = 0.9  # Facteur d'escompte
epsilon_start = 1  # Taux d'exploration initial
epsilon_decay = 0.995
min_epsilon = 0.01
learning_rate = 0.001
target_update = 10  # Fréquence de mise à jour du réseau cible

# --- Dimensions de la fenêtre ---
rows = 2  # Nombre de lignes
cols = 3  # Nombre de colonnes
screen_width = 1536  # Assurez-vous que la largeur est divisible par le nombre de colonnes
screen_height = 864  # Assurez-vous que la hauteur est divisible par le nombre de lignes
single_frame_size_x = screen_width // cols  # 512
single_frame_size_y = screen_height // rows  # 432
total_frame_size_x = single_frame_size_x * cols  # 1536
total_frame_size_y = single_frame_size_y * rows  # 864

# --- Initialisation de Pygame ---
pygame.init()
pygame.display.set_caption('Snake AI - Generation Training')
# Mode fenêtre avec taille exacte
game_window = pygame.display.set_mode((screen_width, screen_height))
print(f"Window size set to: {game_window.get_size()}")

# --- Couleurs ---
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)

# --- Contrôleur FPS ---
fps_controller = pygame.time.Clock()

# --- Flag de Débogage ---
DEBUG = False  # Mettre à True pour activer les impressions de débogage

# --- Classes pour le DQN ---
class DQNAgent:
    def __init__(self, model=None, memory=None, epsilon=epsilon_start):
        self.memory = deque(maxlen=max_memory_size) if memory is None else memory
        self.epsilon = epsilon
        self.model = self.build_model() if model is None else model
        self.target_model = self.build_model()
        if model is not None:
            self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.reward_total = 0

    def build_model(self):
        # Réseau de neurones simplifié
        model = nn.Sequential(
            nn.Linear(11, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
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
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.bool).unsqueeze(1)

        # Prédictions actuelles
        pred = self.model(states).gather(1, actions)

        # Calcul des cibles
        with torch.no_grad():
            Q_new = self.target_model(next_states).max(1)[0].unsqueeze(1)
            Q_new[dones] = 0.0
            target = rewards + gamma * Q_new

        # Calcul de la perte et optimisation
        loss = self.criterion(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_short_memory(self, state, action, reward, next_state, done):
        state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        next_state0 = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)
        target = self.model(state0)
        with torch.no_grad():
            if not done:
                Q_new = reward + gamma * torch.max(self.target_model(next_state0))
            else:
                Q_new = reward
        target[0][action] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(self.model(state0), target)
        loss.backward()
        self.optimizer.step()

# --- Classe du jeu ---
class SnakeGameAI:
    def __init__(self, agent, surface):
        self.agent = agent
        self.surface = surface
        self.reset()

    def reset(self):
        # Aligner la position initiale sur la grille de 10 pixels
        self.snake_pos = [
            (single_frame_size_x // 2) // 10 * 10,
            (single_frame_size_y // 2) // 10 * 10
        ]
        self.snake_body = [
            self.snake_pos[:],
            [self.snake_pos[0] - 10, self.snake_pos[1]],
            [self.snake_pos[0] - 20, self.snake_pos[1]]
        ]
        self.food_spawn = True
        self.direction = 'RIGHT'
        self.score = 0
        self.frame_iteration = 0
        self.time_limit = 20  # Temps limite par partie, en secondes
        self.start_time = pygame.time.get_ticks()  # Temps de début en millisecondes

        # Générer une position de nourriture qui ne chevauche pas le serpent
        while True:
            self.food_pos = [
                random.randrange(0, single_frame_size_x // 10) * 10,
                random.randrange(0, single_frame_size_y // 10) * 10
            ]
            if self.food_pos not in self.snake_body:
                break

    def is_collision(self, point=None):
        if point is None:
            point = self.snake_pos
        if point[0] < 0 or point[0] >= single_frame_size_x or point[1] < 0 or point[1] >= single_frame_size_y:
            return True
        if point in self.snake_body[1:]:
            return True
        return False

    def play_step(self, current_generation):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Sauvegarder avant de quitter
                pygame.quit()
                sys.exit()

        state_old = self.agent.get_state(self)
        move = self.agent.act(state_old)
        action = [0, 0, 0]
        action[move] = 1

        if DEBUG:
            print(f"[Gen {current_generation}] Snake Position Avant Mouvement: {self.snake_pos}")
            print(f"[Gen {current_generation}] Food Position: {self.food_pos}")

        # Calcul de la distance avant le mouvement
        distance_old = np.linalg.norm(np.array(self.snake_pos) - np.array(self.food_pos))

        self.move(action)
        self.snake_body.insert(0, self.snake_pos[:])

        if DEBUG:
            print(f"[Gen {current_generation}] Snake Position Après Mouvement: {self.snake_pos}")

        # Vérifier l'alignement des positions
        assert self.snake_pos[0] % 10 == 0 and self.snake_pos[1] % 10 == 0, "Snake position is not aligned to grid!"
        assert self.food_pos[0] % 10 == 0 and self.food_pos[1] % 10 == 0, "Food position is not aligned to grid!"

        reward = 0
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

        if time_remaining <= 0:
            done = True
            reward -= 20  # Pénalité pour le temps écoulé
            self.agent.reward_total += reward
            if DEBUG:
                print(f"[Gen {current_generation}] Time's up! Reward: {reward}")
            self.update_ui(reward, time_remaining, current_generation)
            return reward, done, self.score

        if self.is_collision():
            done = True
            reward -= 100  # Pénalité pour la mort
            self.agent.reward_total += reward
            if DEBUG:
                print(f"[Gen {current_generation}] Collision! Reward: {reward}")
            self.update_ui(reward, time_remaining, current_generation)
            return reward, done, self.score

        if self.snake_pos == self.food_pos:
            self.score += 1
            reward += 50  # Récompense pour avoir mangé la nourriture
            self.food_spawn = False
            if DEBUG:
                print(f"[Gen {current_generation}] Ate food! Reward: {reward}")
        else:
            self.snake_body.pop()

        if not self.food_spawn:
            # Assurez-vous que la nourriture ne spawn pas sur le corps du serpent
            while True:
                new_food_pos = [
                    random.randrange(0, single_frame_size_x // 10) * 10,
                    random.randrange(0, single_frame_size_y // 10) * 10
                ]
                if new_food_pos not in self.snake_body:
                    self.food_pos = new_food_pos
                    break
            self.food_spawn = True

        self.agent.reward_total += reward

        # Entraînement à court terme
        state_new = self.agent.get_state(self)
        self.agent.train_short_memory(state_old, move, reward, state_new, done)
        # Mémoriser
        self.agent.remember(state_old, move, reward, state_new, done)

        self.update_ui(reward, time_remaining, current_generation)
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

    def update_ui(self, reward, time_remaining=None, current_generation=1):
        # Dessiner uniquement sur la surface assignée
        self.surface.fill(black)
        for pos in self.snake_body:
            pygame.draw.rect(self.surface, green, pygame.Rect(pos[0], pos[1], 10, 10))
        pygame.draw.rect(self.surface, white, pygame.Rect(self.food_pos[0], self.food_pos[1], 10, 10))

        font = pygame.font.SysFont('consolas', 20)
        text_score = font.render('Score: ' + str(self.score), True, white)
        text_generation = font.render(f'Génération: {current_generation}', True, white)
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

    # Initialiser un seul agent avec le meilleur modèle
    if best_model_state_dict is not None:
        agent = DQNAgent(model=None, memory=None, epsilon=best_epsilon)
        agent.model.load_state_dict(best_model_state_dict)
        agent.update_target_model()
        if best_memory is not None:
            agent.memory = deque(best_memory, maxlen=max_memory_size)
    else:
        agent = DQNAgent()

    generation = 0

    while True:
        try:
            generation += 1
            print(f"\n--- Génération {generation} ---")

            # Créer 6 sous-surfaces pour les 6 jeux arrangés en grille 3x2
            games = []
            for row in range(rows):
                for col in range(cols):
                    x = col * single_frame_size_x
                    y = row * single_frame_size_y
                    w = single_frame_size_x
                    h = single_frame_size_y

                    surface_rect = (x, y, w, h)
                    print(f"Creating subsurface: {surface_rect}")

                    # Vérifier que le rectangle est à l'intérieur de la fenêtre principale
                    if (x + w > total_frame_size_x) or (y + h > total_frame_size_y):
                        print(f"Erreur: Le rectangle {surface_rect} dépasse la fenêtre principale.")
                        pygame.quit()
                        sys.exit()

                    surface = game_window.subsurface(surface_rect)
                    game = SnakeGameAI(agent, surface)
                    games.append(game)

            # Réinitialiser les récompenses totales pour l'agent
            agent.reward_total = 0

            # Jouer les 6 parties jusqu'à ce qu'elles se terminent
            done_flags = [False] * (rows * cols)
            while not all(done_flags):
                for i, game in enumerate(games):
                    if not done_flags[i]:
                        reward, done, score = game.play_step(current_generation=generation)
                        done_flags[i] = done
                        agent.reward_total += reward  # Accumuler la récompense totale

            # Entraîner l'agent une fois après avoir joué toutes les parties
            agent.replay()

            # Mettre à jour le réseau cible périodiquement
            if generation % target_update == 0:
                print("Updating target network...")
                agent.update_target_model()

            # Déterminer le meilleur agent parmi les 6 (un seul agent partagé)
            best_agent = agent
            best_reward = agent.reward_total
            best_index = 1  # Un seul agent

            print(f"Agent {best_index} est le meilleur avec une récompense totale de {best_reward}")

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
