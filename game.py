"""
Snake AI avec DQN et LSTM
Créé avec Arcade et PyTorch
"""

import arcade
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import pickle
from torch.utils.tensorboard import SummaryWriter
import time
import platform
import ctypes
import subprocess
import os

# --- Hyperparamètres ---
difficulty = 0.05  # Temps en secondes entre chaque mise à jour (20 FPS)
max_memory_size = 100000
batch_size = 64
gamma = 0.95
epsilon_start = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
learning_rate = 0.001
target_update = 10
sequence_length = 4  # Longueur des séquences pour le LSTM

# --- Dimensions de la fenêtre ---
rows = 2
cols = 3
screen_width = 1536
screen_height = 864
single_frame_size_x = screen_width // cols  # 512
single_frame_size_y = screen_height // rows  # 432

# --- TensorBoard ---
writer = SummaryWriter('runs/snake_ai_arcade')

# --- Couleurs ---
COLOR_BACKGROUND = arcade.color.BLACK
COLOR_SNAKE = arcade.color.GREEN
COLOR_FOOD = arcade.color.RED
COLOR_TEXT = arcade.color.WHITE
COLOR_REWARD_POSITIVE = arcade.color.GREEN
COLOR_REWARD_NEGATIVE = arcade.color.RED
COLOR_BEST_GAME_BORDER = arcade.color.YELLOW
COLOR_WHITE_BORDER = arcade.color.WHITE
COLOR_RAY_SAFE = arcade.color.BLUE
COLOR_RAY_DANGER = arcade.color.RED
COLOR_FOOD_DIRECTION = arcade.color.ORANGE
COLOR_DANGER_ZONE = arcade.color.RED
COLOR_Q_VALUES = [arcade.color.GREEN, arcade.color.YELLOW, arcade.color.RED]  # Pour les valeurs Q

# --- Variables Globales pour Empêcher la Mise en Veille ---
caffeinate_process = None  # Pour macOS

# --- Fonctions pour Empêcher la Mise en Veille ---
def prevent_sleep():
    os_name = platform.system()
    if os_name == "Windows":
        # ES_CONTINUOUS = 0x80000000
        # ES_SYSTEM_REQUIRED = 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    elif os_name == "Darwin":  # macOS
        global caffeinate_process
        caffeinate_process = subprocess.Popen(['caffeinate'])
    elif os_name == "Linux":
        pass

def allow_sleep():
    os_name = platform.system()
    if os_name == "Windows":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    elif os_name == "Darwin":  # macOS
        global caffeinate_process
        if caffeinate_process:
            caffeinate_process.terminate()
    elif os_name == "Linux":
        pass

# --- Classes pour le DQN avec LSTM ---
class DQNAgent:
    def __init__(self, model=None, memory=None, epsilon=epsilon_start, device='cpu'):
        self.memory = deque(maxlen=max_memory_size) if memory is None else memory
        self.epsilon = epsilon
        self.device = device
        self.model = self.build_model().to(self.device) if model is None else model.to(self.device)
        self.target_model = self.build_model().to(self.device)
        if model is not None:
            self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.step_count = 0  # Compteur pour TensorBoard
        self.hidden_state = None  # État caché initial du LSTM
        self.q_values = None  # Initialiser q_values à None

    def build_model(self):
        class LSTMNet(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(LSTMNet, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.relu = nn.ReLU()
                self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
                self.fc2 = nn.Linear(hidden_size, output_size)

            def forward(self, x, hidden_state):
                x = self.fc1(x)
                x = self.relu(x)
                x, hidden_state = self.lstm(x, hidden_state)
                x = self.fc2(x)
                return x, hidden_state

        input_size = 16  # Taille de l'état
        hidden_size = 128
        output_size = 3  # Nombre d'actions possibles

        model = LSTMNet(input_size, hidden_size, output_size)
        return model

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_state(self, game):
        head = game.snake_pos

        # Directions pour le ray casting
        directions = [
            (0, 10),    # Haut
            (10, 10),   # Haut-Droite
            (10, 0),    # Droite
            (10, -10),  # Bas-Droite
            (0, -10),   # Bas
            (-10, -10), # Bas-Gauche
            (-10, 0),   # Gauche
            (-10, 10)   # Haut-Gauche
        ]

        # Calcul des distances aux obstacles dans chaque direction
        obstacle_distances = []
        max_distance = max(single_frame_size_x, single_frame_size_y) // 10  # Distance maximale possible en cases
        for dx, dy in directions:
            distance = 0
            pos = head.copy()
            while True:
                pos[0] += dx
                pos[1] += dy
                distance += 1
                if game.is_collision(pos) or not self.is_within_frame(pos, game.x_offset, game.y_offset):
                    break
            # Normaliser la distance
            obstacle_distances.append(distance / max_distance)

        # Direction actuelle du serpent
        dir_l = game.direction == 'LEFT'
        dir_r = game.direction == 'RIGHT'
        dir_u = game.direction == 'UP'
        dir_d = game.direction == 'DOWN'

        # Position relative de la nourriture
        food_direction = [
            game.food_pos[0] < game.snake_pos[0],  # Nourriture à gauche
            game.food_pos[0] > game.snake_pos[0],  # Nourriture à droite
            game.food_pos[1] > game.snake_pos[1],  # Nourriture en haut
            game.food_pos[1] < game.snake_pos[1],  # Nourriture en bas
        ]

        state = obstacle_distances + [dir_l, dir_r, dir_u, dir_d] + food_direction

        return np.array(state, dtype=float)

    def is_within_frame(self, pos, x_offset, y_offset):
        return x_offset <= pos[0] < x_offset + single_frame_size_x and y_offset <= pos[1] < y_offset + single_frame_size_y

    def remember(self, state, action, reward, next_state, done):
        # Stocker une séquence d'états pour le LSTM
        if hasattr(self, 'state_sequence'):
            self.state_sequence.append(state)
            self.action_sequence.append(action)
            self.reward_sequence.append(reward)
            self.next_state_sequence.append(next_state)
            self.done_sequence.append(done)
            if len(self.state_sequence) == sequence_length:
                self.memory.append((
                    list(self.state_sequence),
                    list(self.action_sequence),
                    list(self.reward_sequence),
                    list(self.next_state_sequence),
                    list(self.done_sequence)
                ))
                self.state_sequence.pop(0)
                self.action_sequence.pop(0)
                self.reward_sequence.pop(0)
                self.next_state_sequence.pop(0)
                self.done_sequence.pop(0)
        else:
            self.state_sequence = [state]
            self.action_sequence = [action]
            self.reward_sequence = [reward]
            self.next_state_sequence = [next_state]
            self.done_sequence = [done]

    def act(self, state):
        state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0).unsqueeze(0).to(self.device)
        if random.uniform(0, 1) < self.epsilon:
            # Action aléatoire (exploration)
            self.hidden_state = None  # Réinitialiser l'état caché pour l'action aléatoire
            with torch.no_grad():
                prediction, _ = self.model(state0, self.hidden_state)
            action = random.randint(0, 2)
        else:
            # Action basée sur le modèle (exploitation)
            with torch.no_grad():
                prediction, self.hidden_state = self.model(state0, self.hidden_state)
            action = torch.argmax(prediction).item()
        self.q_values = prediction.cpu().numpy()[0]
        return action

    def replay(self):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states_seq, actions_seq, rewards_seq, next_states_seq, dones_seq = zip(*minibatch)

        # Convertir en tenseurs
        states_seq = torch.tensor(states_seq, dtype=torch.float).to(self.device)
        actions_seq = torch.tensor(actions_seq, dtype=torch.long).to(self.device)
        rewards_seq = torch.tensor(rewards_seq, dtype=torch.float).to(self.device)
        next_states_seq = torch.tensor(next_states_seq, dtype=torch.float).to(self.device)
        dones_seq = torch.tensor(dones_seq, dtype=torch.float).to(self.device)

        h0 = None  # Initialiser l'état caché

        pred, _ = self.model(states_seq, h0)
        target_pred, _ = self.target_model(next_states_seq, h0)

        pred = pred.gather(2, actions_seq.unsqueeze(2)).squeeze(2)

        target = rewards_seq + (1 - dones_seq) * gamma * torch.max(target_pred, dim=2)[0]

        loss = self.criterion(pred, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Enregistrer la perte dans TensorBoard
        if self.step_count % 10 == 0:
            writer.add_scalar('Loss/train', loss.item(), self.step_count)
        self.step_count += 1

    def train_short_memory(self, state, action, reward, next_state, done):
        # Pour l'entraînement à court terme, nous utilisons des séquences de longueur 1
        state_seq = torch.tensor([state], dtype=torch.float).unsqueeze(0).to(self.device)
        action_seq = torch.tensor([[action]], dtype=torch.long).to(self.device)
        reward_seq = torch.tensor([[reward]], dtype=torch.float).to(self.device)
        next_state_seq = torch.tensor([next_state], dtype=torch.float).unsqueeze(0).to(self.device)
        done_seq = torch.tensor([[done]], dtype=torch.float).to(self.device)

        h0 = None  # État caché initial pour l'entraînement

        pred, _ = self.model(state_seq, h0)
        target_pred, _ = self.target_model(next_state_seq, h0)

        pred = pred.gather(2, action_seq.unsqueeze(2)).squeeze(2)

        target = reward_seq + (1 - done_seq) * gamma * torch.max(target_pred, dim=2)[0]
        target.detach_()

        loss = self.criterion(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Enregistrer la perte dans TensorBoard
        if self.step_count % 10 == 0:
            writer.add_scalar('Loss/train_short_memory', loss.item(), self.step_count)
        self.step_count += 1

# --- Classe du jeu avec Arcade ---
class SnakeGameAI:
    def __init__(self, agent, app, x_offset=0, y_offset=0):
        self.agent = agent
        self.app = app  # Référence à l'application principale
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.reward_total = 0
        self.last_reward = 0  # Pour stocker la dernière récompense
        self.score = 0
        self.reset()

        # Variables pour suivre les métriques
        self.bonbons_manges = 0
        self.intervalles_bonbons = []
        self.temps_last_bonbon = time.time()

        # Variables pour la perception visuelle
        self.obstacle_distances = []
        self.food_direction = []
        self.q_values = [0, 0, 0]
        self.next_action = 0

    def reset(self):
        # Aligner la position initiale sur la grille de 10 pixels
        self.snake_pos = [
            self.x_offset + ((single_frame_size_x // 2) // 10 * 10),
            self.y_offset + ((single_frame_size_y // 2) // 10 * 10)
        ]
        self.snake_body = [
            self.snake_pos[:],
            [self.snake_pos[0] - 10, self.snake_pos[1]],
            [self.snake_pos[0] - 20, self.snake_pos[1]]
        ]
        self.food_spawn = True
        self.direction = 'RIGHT'
        self.reward_total = 0
        self.last_reward = 0
        self.score = 0
        self.time_limit = 20  # Temps limite par partie, en secondes
        self.start_time = time.time()
        self.time_remaining = self.time_limit  # Initialiser le temps restant

        # Réinitialiser les métriques
        self.bonbons_manges = 0
        self.intervalles_bonbons = []
        self.temps_last_bonbon = time.time()

        # Générer une position de nourriture qui ne chevauche pas le serpent
        while True:
            self.food_pos = [
                self.x_offset + random.randrange(0, single_frame_size_x // 10) * 10,
                self.y_offset + random.randrange(0, single_frame_size_y // 10) * 10
            ]
            if self.food_pos not in self.snake_body:
                break

    def is_collision(self, point=None):
        if point is None:
            point = self.snake_pos
        if point[0] < self.x_offset or point[0] >= self.x_offset + single_frame_size_x or \
                point[1] < self.y_offset or point[1] >= self.y_offset + single_frame_size_y:
            return True
        if point in self.snake_body[1:]:
            return True
        return False

    def move(self, action):
        clock_wise = ['RIGHT', 'DOWN', 'LEFT', 'UP']
        idx = clock_wise.index(self.direction)

        if action == 0:
            new_dir = clock_wise[idx]  # Aller tout droit
        elif action == 1:
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
            self.snake_pos[1] += 10
        elif self.direction == 'DOWN':
            self.snake_pos[1] -= 10

        # Mettre à jour le temps restant après le déplacement
        self.update_time()

    def update_time(self):
        elapsed_time = time.time() - self.start_time
        self.time_remaining = max(0, self.time_limit - elapsed_time)

    def play_step(self, delta_time, current_generation):
        state_old = self.agent.get_state(self)
        # Stocker les distances aux obstacles et la direction de la nourriture pour la visualisation
        self.obstacle_distances = state_old[:8]
        self.food_direction = state_old[12:]

        move = self.agent.act(state_old)
        self.next_action = move  # Stocker l'action pour la visualisation
        self.q_values = self.agent.q_values  # Récupérer les valeurs Q pour la visualisation

        action = move  # 0: Tout droit, 1: Droite, 2: Gauche

        self.move(action)
        self.snake_body.insert(0, self.snake_pos[:])

        # Initialisation de la récompense avec la récompense de survie
        reward = self.app.rewards['survie']
        done = False

        # Vérifier les collisions
        if self.is_collision():
            done = True
            reward += self.app.penalties['collision']
            self.reward_total += reward
            self.last_reward = reward
            return reward, done, self.score

        # Vérifier si le serpent a mangé un bonbon
        if self.snake_pos == self.food_pos:
            self.score += 1
            reward += self.app.rewards['manger_bonbon']
            self.bonbons_manges += 1
            self.food_spawn = False
            interval = time.time() - self.temps_last_bonbon
            self.intervalles_bonbons.append(interval)
            self.temps_last_bonbon = time.time()
        else:
            # Retirer le dernier segment pour maintenir la taille constante
            self.snake_body.pop()

        # Générer une nouvelle nourriture si nécessaire
        if not self.food_spawn:
            while True:
                new_food_pos = [
                    self.x_offset + random.randrange(0, single_frame_size_x // 10) * 10,
                    self.y_offset + random.randrange(0, single_frame_size_y // 10) * 10
                ]
                if new_food_pos not in self.snake_body:
                    self.food_pos = new_food_pos
                    break
            self.food_spawn = True

        # Vérifier si le temps est écoulé
        if self.time_remaining <= 0:
            done = True
            reward += self.app.penalties['temps_expire']
            self.reward_total += reward
            self.last_reward = reward
            return reward, done, self.score

        self.reward_total += reward
        self.last_reward = reward

        # Entraînement à court terme
        state_new = self.agent.get_state(self)
        self.agent.train_short_memory(state_old, move, reward, state_new, done)
        # Mémoriser
        self.agent.remember(state_old, move, reward, state_new, done)

        return reward, done, self.score

# --- Classe principale de l'application ---
class SnakeAIApp(arcade.Window):
    def __init__(self):
        super().__init__(screen_width, screen_height, "Snake AI - Arcade Version", update_rate=difficulty)
        arcade.set_background_color(COLOR_BACKGROUND)

        # Empêcher la mise en veille du système
        prevent_sleep()

        # Définir le périphérique (GPU si disponible)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialiser les variables de génération et de meilleur score
        self.generation = 0
        self.best_score = 0  # Meilleur score atteint jusqu'à présent

        # Définir les récompenses et pénalités initiales
        self.rewards = {
            'manger_bonbon': 40,
            'survie': 0.1
        }
        self.penalties = {
            'collision': -150,
            'temps_expire': -20
        }

        # Charger le modèle, la mémoire, la génération et le meilleur score du meilleur agent précédent
        try:
            with open("best_agent.pkl", "rb") as f:
                best_agent_data = pickle.load(f)
            print("Meilleur agent chargé depuis best_agent.pkl")
            best_model_state_dict = best_agent_data['model_state_dict']
            best_memory = best_agent_data['memory']
            best_epsilon = best_agent_data['epsilon']
            loaded_generation = best_agent_data.get('generation', 0)
            loaded_best_score = best_agent_data.get('best_score', 0)
            self.generation = loaded_generation
            self.best_score = loaded_best_score
        except FileNotFoundError:
            print("Aucun agent précédent trouvé, création d'un nouvel agent.")
            best_model_state_dict = None
            best_memory = None
            best_epsilon = epsilon_start

        # Initialiser l'agent
        if best_model_state_dict is not None:
            self.agent = DQNAgent(model=None, memory=None, epsilon=best_epsilon, device=self.device)
            self.agent.model.load_state_dict(best_model_state_dict)
            self.agent.update_target_model()
            if best_memory is not None:
                self.agent.memory = deque(best_memory, maxlen=max_memory_size)
        else:
            self.agent = DQNAgent(device=self.device)

        # Créer les jeux
        self.games = []
        for row in range(rows):
            for col in range(cols):
                x_offset = col * single_frame_size_x
                y_offset = row * single_frame_size_y
                game = SnakeGameAI(self.agent, self, x_offset, y_offset)
                self.games.append(game)

        self.done_flags = [False] * len(self.games)

        # Charger le meilleur score si disponible
        self.best_game_index = None  # Index du meilleur jeu de la génération

    def on_draw(self):
        arcade.start_render()
        # Dessiner tous les jeux
        for i, game in enumerate(self.games):
            # Dessiner une bordure blanche autour de chaque jeu
            arcade.draw_rectangle_outline(
                game.x_offset + single_frame_size_x / 2,
                game.y_offset + single_frame_size_y / 2,
                single_frame_size_x,
                single_frame_size_y,
                COLOR_WHITE_BORDER,
                border_width=2
            )

            # Dessiner le cadre du meilleur jeu avec une bordure spéciale
            if i == self.best_game_index:
                arcade.draw_rectangle_outline(
                    game.x_offset + single_frame_size_x / 2,
                    game.y_offset + single_frame_size_y / 2,
                    single_frame_size_x - 4,
                    single_frame_size_y - 4,
                    COLOR_BEST_GAME_BORDER,
                    border_width=4
                )

            # Dessiner le serpent
            for pos in game.snake_body:
                arcade.draw_rectangle_filled(pos[0] + 5, pos[1] + 5, 10, 10, COLOR_SNAKE)
            # Dessiner la nourriture
            arcade.draw_rectangle_filled(game.food_pos[0] + 5, game.food_pos[1] + 5, 10, 10, COLOR_FOOD)
            # Afficher le score
            arcade.draw_text(f"Score: {game.score}", game.x_offset + 10, game.y_offset + single_frame_size_y - 20,
                             COLOR_TEXT, 12)

            # Afficher le temps restant
            arcade.draw_text(f"Temps restant: {int(game.time_remaining)}s",
                             game.x_offset + 10, game.y_offset + single_frame_size_y - 40,
                             COLOR_TEXT, 12)

            # Afficher la récompense totale
            arcade.draw_text(f"Récompense Totale: {round(game.reward_total, 2)}",
                             game.x_offset + 10, game.y_offset + single_frame_size_y - 60,
                             COLOR_TEXT, 12)

            # Indicateur de récompense/punition
            if game.last_reward > 0:
                arcade.draw_circle_filled(game.x_offset + single_frame_size_x - 20,
                                          game.y_offset + single_frame_size_y - 20,
                                          10, COLOR_REWARD_POSITIVE)
            elif game.last_reward < 0:
                arcade.draw_circle_filled(game.x_offset + single_frame_size_x - 20,
                                          game.y_offset + single_frame_size_y - 20,
                                          10, COLOR_REWARD_NEGATIVE)
            else:
                # Si la dernière récompense est neutre, dessiner un cercle gris
                arcade.draw_circle_filled(game.x_offset + single_frame_size_x - 20,
                                          game.y_offset + single_frame_size_y - 20,
                                          10, arcade.color.GRAY)

            # Dessiner les rayons de détection
            head_x, head_y = game.snake_pos
            directions = [
                (0, 10),    # Haut
                (10, 10),   # Haut-Droite
                (10, 0),    # Droite
                (10, -10),  # Bas-Droite
                (0, -10),   # Bas
                (-10, -10), # Bas-Gauche
                (-10, 0),   # Gauche
                (-10, 10)   # Haut-Gauche
            ]

            for idx, (dx, dy) in enumerate(directions):
                distance = game.obstacle_distances[idx] * (max(single_frame_size_x, single_frame_size_y))
                end_x = head_x + dx * distance
                end_y = head_y + dy * distance

                # Limiter les rayons aux limites du cadre du jeu
                end_x, end_y = self.clip_line_to_frame(head_x, head_y, end_x, end_y, game.x_offset, game.y_offset)

                color = COLOR_RAY_SAFE if distance > 50 else COLOR_RAY_DANGER
                arcade.draw_line(head_x + 5, head_y + 5, end_x + 5, end_y + 5, color, 1)

            # Indiquer la direction de la nourriture avec des flèches
            food_dir = game.food_direction
            arrow_size = 15
            arrow_head_size = 5
            if food_dir[0]:  # Nourriture à gauche
                self.draw_arrow(head_x + 5, head_y + 5, head_x - arrow_size + 5, head_y + 5, COLOR_FOOD_DIRECTION)
            if food_dir[1]:  # Nourriture à droite
                self.draw_arrow(head_x + 5, head_y + 5, head_x + arrow_size + 5, head_y + 5, COLOR_FOOD_DIRECTION)
            if food_dir[2]:  # Nourriture en haut
                self.draw_arrow(head_x + 5, head_y + 5, head_x + 5, head_y + arrow_size + 5, COLOR_FOOD_DIRECTION)
            if food_dir[3]:  # Nourriture en bas
                self.draw_arrow(head_x + 5, head_y + 5, head_x + 5, head_y - arrow_size + 5, COLOR_FOOD_DIRECTION)

            # Indiquer si le danger est proche
            if min(game.obstacle_distances) < 0.1:
                arcade.draw_circle_outline(head_x + 5, head_y + 5, 20, COLOR_DANGER_ZONE, 2)

            # Afficher les valeurs Q pour chaque action
            q_values = game.q_values
            if q_values is not None and len(q_values) > 0:
                q_values = np.array(q_values).flatten()
                max_q = np.max(np.abs(q_values))
            else:
                max_q = 1  # Éviter la division par zéro

            bar_width = 50
            bar_height = 10
            actions = ['Tout droit', 'Droite', 'Gauche']
            for idx, q in enumerate(q_values):
                if max_q != 0:
                    bar_length = (q / max_q) * bar_width
                else:
                    bar_length = 0
                bar_x = game.x_offset + single_frame_size_x - 70
                bar_y = game.y_offset + single_frame_size_y - (idx + 1) * (bar_height + 5) - 20
                color = COLOR_Q_VALUES[idx]
                # Dessiner la barre
                arcade.draw_rectangle_filled(
                    bar_x + bar_length / 2,
                    bar_y,
                    abs(bar_length),
                    bar_height,
                    color
                )
                # Afficher le nom de l'action et la Q-valeur
                arcade.draw_text(f"{actions[idx]}: {q:.2f}", bar_x - 60, bar_y - 5, COLOR_TEXT, 10)

            # Indiquer l'action prévue
            next_action = game.next_action
            if next_action == 0:
                action_text = "Tout droit"
            elif next_action == 1:
                action_text = "Tourner à droite"
            else:
                action_text = "Tourner à gauche"
            arcade.draw_text(f"Action: {action_text}", game.x_offset + 10, game.y_offset + 10, COLOR_TEXT, 12)

        # Afficher la génération en haut au centre de la fenêtre
        arcade.draw_text(f"Génération: {self.generation}",
                         self.width // 2, self.height - 30,
                         COLOR_TEXT, 20, anchor_x="center")

        # Afficher le meilleur score atteint jusqu'à présent
        arcade.draw_text(f"Meilleur Score: {self.best_score}",
                         self.width // 2, self.height - 60,
                         COLOR_TEXT, 16, anchor_x="center")

    def clip_line_to_frame(self, x1, y1, x2, y2, x_offset, y_offset):
        # Limiter la ligne aux limites du cadre du jeu
        min_x = x_offset
        max_x = x_offset + single_frame_size_x
        min_y = y_offset
        max_y = y_offset + single_frame_size_y

        # Implémentation de l'algorithme de Liang-Barsky
        p = [-(x2 - x1), x2 - x1, -(y2 - y1), y2 - y1]
        q = [x1 - min_x, max_x - x1, y1 - min_y, max_y - y1]
        u1 = 0.0
        u2 = 1.0
        for i in range(4):
            if p[i] == 0:
                if q[i] < 0:
                    return x1, y1  # Trivialement hors limites
            else:
                t = q[i] / p[i]
                if p[i] < 0:
                    if t > u1:
                        u1 = t
                else:
                    if t < u2:
                        u2 = t
        if u1 > u2:
            return x1, y1  # Hors limites
        x1_clip = x1 + u1 * (x2 - x1)
        y1_clip = y1 + u1 * (y2 - y1)
        x2_clip = x1 + u2 * (x2 - x1)
        y2_clip = y1 + u2 * (y2 - y1)
        return x2_clip, y2_clip

    def draw_arrow(self, x_start, y_start, x_end, y_end, color):
        # Dessiner une flèche pour indiquer la direction
        arcade.draw_line(x_start, y_start, x_end, y_end, color, 2)
        # Calculer le vecteur directionnel
        dx = x_end - x_start
        dy = y_end - y_start
        angle = np.arctan2(dy, dx)
        # Définir les pointes de la flèche
        arrow_length = 10
        angle1 = angle + np.pi / 6
        angle2 = angle - np.pi / 6
        x1 = x_end - arrow_length * np.cos(angle1)
        y1 = y_end - arrow_length * np.sin(angle1)
        x2 = x_end - arrow_length * np.cos(angle2)
        y2 = y_end - arrow_length * np.sin(angle2)
        # Dessiner les pointes de la flèche
        arcade.draw_triangle_filled(x_end, y_end, x1, y1, x2, y2, color)

    def on_update(self, delta_time):
        # Jouer les parties
        if not all(self.done_flags):
            for i, game in enumerate(self.games):
                if not self.done_flags[i]:
                    reward, done, score = game.play_step(delta_time, current_generation=self.generation)
                    self.done_flags[i] = done
        else:
            # Entraîner l'agent une fois après avoir joué toutes les parties
            self.agent.replay()

            # Mettre à jour le réseau cible périodiquement
            if self.generation % target_update == 0:
                print("Updating target network...")
                self.agent.update_target_model()

            # Collecter les récompenses totales et les scores de chaque jeu
            rewards = [game.reward_total for game in self.games]
            scores = [game.score for game in self.games]
            avg_reward = sum(rewards) / len(rewards)
            max_reward = max(rewards)
            max_score = max(scores)
            print(f"Récompenses des jeux: {rewards}")
            print(f"Scores des jeux: {scores}")
            print(f"Récompense moyenne: {avg_reward}")
            print(f"Récompense maximale: {max_reward}")
            print(f"Meilleur score de la génération: {max_score}")

            # Mettre à jour le meilleur score global si nécessaire
            if max_score > self.best_score:
                self.best_score = max_score

            # Identifier l'index du meilleur jeu
            self.best_game_index = scores.index(max_score)

            # Sauvegarder les récompenses moyennes et maximales dans TensorBoard
            writer.add_scalar('Reward/avg', avg_reward, self.generation)
            writer.add_scalar('Reward/max', max_reward, self.generation)
            writer.add_scalar('Score/max', max_score, self.generation)

            # Mettre à jour les paramètres de l'agent
            best_model_state_dict = self.agent.model.state_dict()
            best_memory = self.agent.memory
            best_epsilon = max(min_epsilon, self.agent.epsilon * epsilon_decay)
            self.agent.epsilon = best_epsilon

            # Sauvegarder l'agent partagé avec génération et meilleur score
            with open("best_agent.pkl", "wb") as f:
                pickle.dump({
                    'model_state_dict': best_model_state_dict,
                    'memory': best_memory,
                    'epsilon': best_epsilon,
                    'generation': self.generation,
                    'best_score': self.best_score
                }, f)
            print("Meilleur agent sauvegardé dans best_agent.pkl")

            # Sauvegarder les statistiques détaillées
            writer.add_scalar('Stats/Generation', self.generation, self.generation)
            writer.add_scalar('Stats/Best_score', self.best_score, self.generation)

            # Sauvegarder le meilleur score dans un fichier séparé pour une récupération rapide
            with open("best_score.txt", "w") as f:
                f.write(str(self.best_score))

            # Préparer la prochaine génération
            self.generation += 1
            self.done_flags = [False] * len(self.games)
            for game in self.games:
                game.reset()

    def on_close(self):
        print("Fermeture du jeu, sauvegarde en cours...")
        # Sauvegarder l'agent partagé avec génération et meilleur score
        best_model_state_dict = self.agent.model.state_dict()
        best_memory = self.agent.memory
        best_epsilon = self.agent.epsilon
        with open("best_agent.pkl", "wb") as f:
            pickle.dump({
                'model_state_dict': best_model_state_dict,
                'memory': best_memory,
                'epsilon': best_epsilon,
                'generation': self.generation,
                'best_score': self.best_score
            }, f)
        print("Meilleur agent sauvegardé dans best_agent.pkl")
        writer.close()

        # Sauvegarder le meilleur score dans un fichier séparé
        with open("best_score.txt", "w") as f:
            f.write(str(self.best_score))

        # Permettre la mise en veille du système
        allow_sleep()

        super().on_close()

# --- Fonction principale ---
def main():
    # Vérifier si "best_score.txt" existe pour charger le meilleur score
    if os.path.exists("best_score.txt"):
        with open("best_score.txt", "r") as f:
            best_score = int(f.read())
    else:
        best_score = 0

    app = SnakeAIApp()
    app.best_score = best_score  # Assigner le meilleur score chargé
    arcade.run()

if __name__ == '__main__':
    main()