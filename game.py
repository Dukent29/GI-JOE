"""
Snake AI avec DQN et LSTM (passage conditionnel basé sur les performances avec ajustement dynamique des récompenses)
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
batch_size = 128
gamma = 0.95
epsilon_start = 1.0
epsilon_decay = 0.995  # Décroissance plus rapide
min_epsilon = 0.01
learning_rate = 0.001
target_update = 10

# --- Critères pour le passage au LSTM ---
window_size = 10  # Nombre de générations pour le calcul des moyennes
score_threshold = 5  # Score moyen minimal pour le passage
reward_threshold = 50  # Récompense moyenne minimale pour le passage
loss_threshold = 0.1  # Perte moyenne maximale pour le passage
success_score = 3  # Score à atteindre pour considérer une partie réussie
success_rate_threshold = 0.7  # Taux de succès minimal pour le passage
consistency_threshold = 5  # Nombre de générations consécutives où les critères sont satisfaits

# --- Limites pour les récompenses et pénalités ---
reward_limits = {
    'manger_bonbon': (10, 100),
    'survie': (0.01, 1),
}
penalty_limits = {
    'collision': (-300, -50),
    'temps_expire': (-50, -10),
}

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

# --- Classes pour le DQN ---
class DQNAgent:
    def __init__(self, model=None, memory=None, epsilon=epsilon_start, device='cpu', use_lstm=False):
        self.memory = deque(maxlen=max_memory_size) if memory is None else memory
        self.epsilon = epsilon
        self.device = device
        self.use_lstm = use_lstm  # Indicateur pour utiliser le LSTM

        # Choisir le modèle en fonction de use_lstm
        if not self.use_lstm:
            self.model = self.build_simple_model().to(self.device) if model is None else model.to(self.device)
            self.target_model = self.build_simple_model().to(self.device)
        else:
            self.model = self.build_lstm_model().to(self.device) if model is None else model.to(self.device)
            self.target_model = self.build_lstm_model().to(self.device)

        if model is not None:
            self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.step_count = 0  # Compteur pour TensorBoard
        self.q_values = None  # Initialiser q_values à None
        self.hidden_state = None  # Pour le LSTM

    def build_simple_model(self):
        class DQNNet(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(DQNNet, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.relu1 = nn.ReLU()
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.relu2 = nn.ReLU()
                self.fc3 = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                x = self.fc1(x)
                x = self.relu1(x)
                x = self.fc2(x)
                x = self.relu2(x)
                x = self.fc3(x)
                return x

        input_size = 16  # Taille de l'état
        hidden_size = 128
        output_size = 3  # Nombre d'actions possibles

        model = DQNNet(input_size, hidden_size, output_size)
        return model

    def build_lstm_model(self):
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
        if not self.use_lstm:
            self.memory.append((state, action, reward, next_state, done))
        else:
            # Pour le LSTM, on peut utiliser des séquences
            if hasattr(self, 'state_sequence'):
                self.state_sequence.append(state)
                self.action_sequence.append(action)
                self.reward_sequence.append(reward)
                self.next_state_sequence.append(next_state)
                self.done_sequence.append(done)
                if len(self.state_sequence) == 4:  # Longueur de séquence choisie
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
        if not self.use_lstm:
            state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
            if random.uniform(0, 1) < self.epsilon:
                # Action aléatoire (exploration)
                with torch.no_grad():
                    prediction = self.model(state0)
                action = random.randint(0, 2)
            else:
                # Action basée sur le modèle (exploitation)
                with torch.no_grad():
                    prediction = self.model(state0)
                action = torch.argmax(prediction).item()
            self.q_values = prediction.cpu().numpy()[0]
            return action
        else:
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

        if not self.use_lstm:
            minibatch = random.sample(self.memory, batch_size)
            states_mb, actions_mb, rewards_mb, next_states_mb, dones_mb = zip(*minibatch)

            # Convertir en tenseurs
            states_mb = torch.tensor(states_mb, dtype=torch.float).to(self.device)
            actions_mb = torch.tensor(actions_mb, dtype=torch.long).unsqueeze(1).to(self.device)
            rewards_mb = torch.tensor(rewards_mb, dtype=torch.float).unsqueeze(1).to(self.device)
            next_states_mb = torch.tensor(next_states_mb, dtype=torch.float).to(self.device)
            dones_mb = torch.tensor(dones_mb, dtype=torch.float).unsqueeze(1).to(self.device)

            pred = self.model(states_mb).gather(1, actions_mb)
            target_next = self.target_model(next_states_mb).detach().max(1)[0].unsqueeze(1)
            target = rewards_mb + (1 - dones_mb) * gamma * target_next

            loss = self.criterion(pred, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Enregistrer la perte dans TensorBoard
            if self.step_count % 10 == 0:
                writer.add_scalar('Loss/train', loss.item(), self.step_count)
            self.step_count += 1

            # Stocker la perte pour l'évaluation des critères
            self.current_loss = loss.item()
        else:
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

            # Stocker la perte pour l'évaluation des critères
            self.current_loss = loss.item()

    def train_short_memory(self, state, action, reward, next_state, done):
        if not self.use_lstm:
            state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
            next_state0 = torch.tensor(next_state, dtype=torch.float).unsqueeze(0).to(self.device)
            action0 = torch.tensor([action], dtype=torch.long).unsqueeze(1).to(self.device)
            reward0 = torch.tensor([reward], dtype=torch.float).unsqueeze(1).to(self.device)
            done0 = torch.tensor([done], dtype=torch.float).unsqueeze(1).to(self.device)

            pred = self.model(state0).gather(1, action0)
            target_next = self.target_model(next_state0).detach().max(1)[0].unsqueeze(1)
            target = reward0 + (1 - done0) * gamma * target_next

            loss = self.criterion(pred, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Enregistrer la perte dans TensorBoard
            if self.step_count % 10 == 0:
                writer.add_scalar('Loss/train_short_memory', loss.item(), self.step_count)
            self.step_count += 1

            # Stocker la perte pour l'évaluation des critères
            self.current_loss = loss.item()
        else:
            # Pour le LSTM, l'entraînement à court terme peut être similaire
            state_seq = torch.tensor([state], dtype=torch.float).unsqueeze(0).to(self.device)
            next_state_seq = torch.tensor([next_state], dtype=torch.float).unsqueeze(0).to(self.device)
            action_seq = torch.tensor([[action]], dtype=torch.long).to(self.device)
            reward_seq = torch.tensor([[reward]], dtype=torch.float).to(self.device)
            done_seq = torch.tensor([[done]], dtype=torch.float).to(self.device)

            h0 = None  # État caché initial

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

            # Stocker la perte pour l'évaluation des critères
            self.current_loss = loss.item()

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

        # Initialisation de la récompense
        reward = 0
        done = False

        # Vérifier les collisions
        if self.is_collision():
            done = True
            reward += self.app.penalties['collision']
            self.reward_total += reward
            self.last_reward = reward
            return reward, done, self.score

        # Calcul de la distance à la nourriture avant et après le mouvement
        prev_distance = abs(self.snake_body[1][0] - self.food_pos[0]) + abs(self.snake_body[1][1] - self.food_pos[1])
        current_distance = abs(self.snake_pos[0] - self.food_pos[0]) + abs(self.snake_pos[1] - self.food_pos[1])

        # Vérifier si le serpent a mangé un bonbon
        if self.snake_pos == self.food_pos:
            self.score += 1
            time_since_last_food = time.time() - self.temps_last_bonbon

            # Récompense pour manger la nourriture
            reward += self.app.rewards['manger_bonbon']

            # Bonus pour avoir atteint la nourriture rapidement
            if time_since_last_food < 5:
                reward += 10  # Bonus pour rapidité

            self.bonbons_manges += 1
            self.food_spawn = False
            interval = time.time() - self.temps_last_bonbon
            self.intervalles_bonbons.append(interval)
            self.temps_last_bonbon = time.time()
        else:
            # Récompense ou pénalité pour se rapprocher ou s'éloigner de la nourriture
            if current_distance < prev_distance:
                reward += 0.5  # Se rapproche de la nourriture
            else:
                reward -= 0.5  # S'éloigne de la nourriture

            # Récompense pour survie
            reward += self.app.rewards['survie']

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
            'manger_bonbon': 40,  # Récompense de base pour manger la nourriture
            'survie': 0.1,        # Récompense pour chaque action sans mourir
        }
        self.penalties = {
            'collision': -150,    # Pénalité pour collision
            'temps_expire': -20   # Pénalité si le temps est écoulé
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
            best_use_lstm = best_agent_data.get('use_lstm', False)
        except FileNotFoundError:
            print("Aucun agent précédent trouvé, création d'un nouvel agent.")
            best_model_state_dict = None
            best_memory = None
            best_epsilon = epsilon_start
            best_use_lstm = False

        # Initialiser l'agent
        if best_model_state_dict is not None:
            self.agent = DQNAgent(model=None, memory=None, epsilon=best_epsilon, device=self.device, use_lstm=best_use_lstm)
            self.agent.model.load_state_dict(best_model_state_dict)
            self.agent.update_target_model()
            if best_memory is not None:
                self.agent.memory = deque(best_memory, maxlen=max_memory_size)
        else:
            self.agent = DQNAgent(device=self.device, use_lstm=False)

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

        # Variables pour stocker les métriques
        self.scores_history = []
        self.rewards_history = []
        self.loss_history = []
        self.success_rates = []
        self.consistency_count = 0

        # --- Nouvelle Variable pour le Speed Up ---
        self.speed_multiplier = 1  # 1x par défaut

        # --- Mapping des touches numériques à leurs multiplicateurs ---
        self.speed_key_mapping = {
            arcade.key.KEY_0: 0,
            arcade.key.KEY_1: 1,
            arcade.key.KEY_2: 2,
            arcade.key.KEY_3: 3,
            arcade.key.KEY_4: 4,
            arcade.key.KEY_5: 5,
            arcade.key.KEY_6: 6,
            arcade.key.KEY_7: 7,
            arcade.key.KEY_8: 8,
            arcade.key.KEY_9: 9,
        }

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

        # Indiquer si le LSTM est utilisé
        model_type = "LSTM" if self.agent.use_lstm else "Feedforward"
        arcade.draw_text(f"Modèle: {model_type}",
                         self.width // 2, self.height - 90,
                         COLOR_TEXT, 16, anchor_x="center")

        # --- Afficher la Vitesse Actuelle ---
        speed_display = "Pause" if self.speed_multiplier == 0 else f"Vitesse: {self.speed_multiplier}x"
        arcade.draw_text(speed_display,
                         10, 10,
                         arcade.color.WHITE, 14)

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

    def adjust_rewards_penalties(self):
        """
        Ajuste les récompenses et pénalités en fonction des performances de l'agent.
        """
        # Calculer les moyennes sur la fenêtre
        if len(self.scores_history) >= window_size:
            avg_score = sum(self.scores_history[-window_size:]) / window_size
            avg_reward = sum(self.rewards_history[-window_size:]) / window_size

            # Ajustement des récompenses
            # Par exemple, si l'agent performe bien, réduire les récompenses pour augmenter la difficulté
            if avg_score > score_threshold:
                # Réduire les récompenses
                self.rewards['manger_bonbon'] = max(
                    reward_limits['manger_bonbon'][0],
                    self.rewards['manger_bonbon'] * 0.9
                )
                self.rewards['survie'] = max(
                    reward_limits['survie'][0],
                    self.rewards['survie'] * 0.9
                )
            else:
                # Augmenter les récompenses
                self.rewards['manger_bonbon'] = min(
                    reward_limits['manger_bonbon'][1],
                    self.rewards['manger_bonbon'] * 1.1
                )
                self.rewards['survie'] = min(
                    reward_limits['survie'][1],
                    self.rewards['survie'] * 1.1
                )

            # Ajustement des pénalités
            if avg_score < score_threshold:
                # Augmenter les pénalités (plus négatives)
                self.penalties['collision'] = max(
                    penalty_limits['collision'][0],
                    self.penalties['collision'] * 1.1
                )
                self.penalties['temps_expire'] = max(
                    penalty_limits['temps_expire'][0],
                    self.penalties['temps_expire'] * 1.1
                )
            else:
                # Réduire les pénalités (moins négatives)
                self.penalties['collision'] = min(
                    penalty_limits['collision'][1],
                    self.penalties['collision'] * 0.9
                )
                self.penalties['temps_expire'] = min(
                    penalty_limits['temps_expire'][1],
                    self.penalties['temps_expire'] * 0.9
                )

            # Enregistrer les nouvelles valeurs dans TensorBoard
            writer.add_scalar('Rewards/manger_bonbon', self.rewards['manger_bonbon'], self.generation)
            writer.add_scalar('Rewards/survie', self.rewards['survie'], self.generation)
            writer.add_scalar('Penalties/collision', self.penalties['collision'], self.generation)
            writer.add_scalar('Penalties/temps_expire', self.penalties['temps_expire'], self.generation)

    def on_update(self, delta_time):
        # Jouer les parties en fonction du speed_multiplier
        if self.speed_multiplier > 0:
            for _ in range(self.speed_multiplier):
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

                    # Enregistrer les métriques
                    self.scores_history.append(avg_reward)
                    self.rewards_history.append(avg_reward)
                    self.loss_history.append(self.agent.current_loss)
                    success_rate = sum([1 for s in scores if s >= success_score]) / len(scores)
                    self.success_rates.append(success_rate)

                    # Limiter la taille des historiques
                    if len(self.scores_history) > window_size:
                        self.scores_history.pop(0)
                        self.rewards_history.pop(0)
                        self.loss_history.pop(0)
                        self.success_rates.pop(0)

                    # Ajuster les récompenses et pénalités en fonction des performances
                    self.adjust_rewards_penalties()

                    # Vérifier les critères pour le passage au LSTM
                    if not self.agent.use_lstm:
                        criteria_met = False
                        if len(self.scores_history) == window_size:
                            avg_score = sum(self.scores_history) / window_size
                            avg_reward_window = sum(self.rewards_history) / window_size
                            avg_loss = sum(self.loss_history) / window_size
                            avg_success_rate = sum(self.success_rates) / window_size

                            criteria_met = (
                                    avg_score >= score_threshold and
                                    avg_reward_window >= reward_threshold and
                                    avg_loss <= loss_threshold and
                                    avg_success_rate >= success_rate_threshold
                            )

                            if criteria_met:
                                self.consistency_count += 1
                            else:
                                self.consistency_count = 0

                            print(f"Critères satisfaits : {criteria_met}, Compteur de consistance : {self.consistency_count}")

                            if self.consistency_count >= consistency_threshold:
                                print("Passage au modèle LSTM.")
                                self.agent.use_lstm = True
                                self.agent.model = self.agent.build_lstm_model().to(self.device)
                                self.agent.target_model = self.agent.build_lstm_model().to(self.device)
                                # Transférer les poids si possible
                                # Ici, nous pouvons initialiser les poids du LSTM avec ceux du modèle simple (selon la compatibilité)
                                self.agent.optimizer = optim.Adam(self.agent.model.parameters(), lr=learning_rate)
                                self.consistency_count = 0  # Réinitialiser le compteur
                        else:
                            self.consistency_count = 0

                    # Mettre à jour le meilleur score global si nécessaire
                    if max_score > self.best_score:
                        self.best_score = max_score

                    # Identifier l'index du meilleur jeu
                    self.best_game_index = scores.index(max_score)

                    # Sauvegarder les récompenses moyennes et maximales dans TensorBoard
                    writer.add_scalar('Reward/avg', avg_reward, self.generation)
                    writer.add_scalar('Reward/max', max_reward, self.generation)
                    writer.add_scalar('Score/max', max_score, self.generation)
                    writer.add_scalar('Epsilon', self.agent.epsilon, self.generation)
                    writer.add_scalar('Loss', self.agent.current_loss, self.generation)
                    writer.add_scalar('Success Rate', success_rate, self.generation)

                    # Sauvegarder le modèle si le meilleur score est atteint
                    if max_score >= self.best_score:
                        best_model_state_dict = self.agent.model.state_dict()
                        best_memory = self.agent.memory
                        best_epsilon = self.agent.epsilon
                        with open("best_agent.pkl", "wb") as f:
                            pickle.dump({
                                'model_state_dict': best_model_state_dict,
                                'memory': best_memory,
                                'epsilon': best_epsilon,
                                'generation': self.generation,
                                'best_score': self.best_score,
                                'use_lstm': self.agent.use_lstm
                            }, f)
                        print("Meilleur agent sauvegardé dans best_agent.pkl")

                    # Sauvegarder les statistiques détaillées
                    writer.add_scalar('Stats/Generation', self.generation, self.generation)
                    writer.add_scalar('Stats/Best_score', self.best_score, self.generation)

                    # Sauvegarder le meilleur score dans un fichier séparé pour une récupération rapide
                    with open("best_score.txt", "w") as f:
                        f.write(str(self.best_score))

                    # Mettre à jour l'epsilon
                    self.agent.epsilon = max(min_epsilon, self.agent.epsilon * epsilon_decay)

                    # Préparer la prochaine génération
                    self.generation += 1
                    self.done_flags = [False] * len(self.games)
                    for game in self.games:
                        game.reset()

    def on_key_press(self, key, modifiers):
        """
        Gère les pressions sur les touches pour ajuster la vitesse du jeu.
        Les touches numériques de 0 à 9 ajustent le multiplicateur de vitesse.
        0 met le jeu en pause.
        """
        if key in self.speed_key_mapping:
            # Récupérer le multiplicateur basé sur la touche pressée
            multiplier = self.speed_key_mapping[key]
            if multiplier == 0:
                self.speed_multiplier = 0  # Pause
            else:
                self.speed_multiplier = multiplier  # 1x à 9x
            print(f"Vitesse ajustée à {self.speed_multiplier}x")

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
                'best_score': self.best_score,
                'use_lstm': self.agent.use_lstm
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
