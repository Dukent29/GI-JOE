"""
Snake AI with Persistent Memory and Adaptive Reward System
Made with Arcade and PyTorch
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
import os  # Ajout de l'import manquant

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
COLOR_BEST_GAME_BORDER = arcade.color.YELLOW  # Changement pour mieux distinguer
COLOR_WHITE_BORDER = arcade.color.WHITE  # Nouvelle couleur pour les bordures blanches

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
        # Sous Linux, la mise en veille dépend de l'environnement de bureau
        # Vous pouvez ajouter des commandes spécifiques si nécessaire
        pass

def allow_sleep():
    os_name = platform.system()
    if os_name == "Windows":
        # Réinitialiser l'état pour permettre la mise en veille
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    elif os_name == "Darwin":  # macOS
        global caffeinate_process
        if caffeinate_process:
            caffeinate_process.terminate()
    elif os_name == "Linux":
        # Sous Linux, aucune action spécifique
        pass

# --- Classes pour le DQN ---
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

    def build_model(self):
        # Réseau de neurones simplifié et optimisé
        model = nn.Sequential(
            nn.Linear(11, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        return model

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_state(self, game):
        # État du jeu sous forme de tableau normalisé
        head = game.snake_pos
        point_l = [head[0] - 10, head[1]]
        point_r = [head[0] + 10, head[1]]
        point_u = [head[0], head[1] + 10]  # En Arcade, l'axe Y augmente vers le haut
        point_d = [head[0], head[1] - 10]

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

            # Nourriture relative à la position du serpent
            game.food_pos[0] < game.snake_pos[0],  # nourriture à gauche
            game.food_pos[0] > game.snake_pos[0],  # nourriture à droite
            game.food_pos[1] > game.snake_pos[1],  # nourriture en haut
            game.food_pos[1] < game.snake_pos[1]   # nourriture en bas
        ]

        # Conversion en float et normalisation
        return np.array(state, dtype=float)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 2)
        state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        prediction = self.model(state0)
        return torch.argmax(prediction).item()

    def replay(self):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.tensor(np.array(states), dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).unsqueeze(1).to(self.device)

        # Prédictions actuelles
        pred = self.model(states).gather(1, actions)

        # Calcul des cibles avec Double DQN
        with torch.no_grad():
            best_actions = self.model(next_states).argmax(1).unsqueeze(1)
            Q_new = self.target_model(next_states).gather(1, best_actions)
            Q_new[dones] = 0.0
            target = rewards + gamma * Q_new

        # Calcul de la perte et optimisation
        loss = self.criterion(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Enregistrer la perte dans TensorBoard
        if self.step_count % 10 == 0:
            writer.add_scalar('Loss/train', loss.item(), self.step_count)
        self.step_count += 1

    def train_short_memory(self, state, action, reward, next_state, done):
        state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        next_state0 = torch.tensor(next_state, dtype=torch.float).unsqueeze(0).to(self.device)
        target = self.model(state0).clone()

        with torch.no_grad():
            if not done:
                Q_new = reward + gamma * self.target_model(next_state0).max(1)[0].unsqueeze(1)
            else:
                Q_new = torch.tensor([[reward]], device=self.device)
        target[0][action] = Q_new

        # Calcul de la perte et optimisation
        loss = self.criterion(self.model(state0), target)
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
        self.app = app  # Référence à l'application principale pour accéder aux récompenses dynamiques
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
        self.proximite_danger = 0
        self.deplacements_inutiles = 0
        self.deplacements_vers_bonbon = 0
        self.deplacements_eloigne_bonbon = 0

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
        self.start_time = time.time()  # Utiliser time.time() au lieu de arcade.get_time()

        # Réinitialiser les métriques
        self.bonbons_manges = 0
        self.intervalles_bonbons = []
        self.temps_last_bonbon = time.time()
        self.proximite_danger = 0
        self.deplacements_inutiles = 0
        self.deplacements_vers_bonbon = 0
        self.deplacements_eloigne_bonbon = 0

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

    def play_step(self, delta_time, current_generation):
        state_old = self.agent.get_state(self)
        move = self.agent.act(state_old)
        action = move  # 0: Tout droit, 1: Droite, 2: Gauche

        # Calcul de la distance avant le mouvement
        distance_old = np.linalg.norm(np.array(self.snake_pos) - np.array(self.food_pos))

        self.move(action)
        self.snake_body.insert(0, self.snake_pos[:])

        # Initialisation de la récompense avec la pénalité pour chaque déplacement
        reward = self.app.rewards['deplacement']
        done = False

        # Calcul de la distance après le mouvement
        distance_new = np.linalg.norm(np.array(self.snake_pos) - np.array(self.food_pos))

        # Récompense ou punition basée sur le déplacement
        delta_distance = distance_old - distance_new

        if delta_distance > 0:
            # Déplacement vers le bonbon
            reduction_steps = int((delta_distance / distance_old) * 10) if distance_old != 0 else 0
            reward += self.app.rewards['deplacement_vers_bonbon'] * reduction_steps
            self.deplacements_vers_bonbon += 1
        elif delta_distance < 0:
            # Déplacement s'éloignant du bonbon
            increase_steps = int((-delta_distance / distance_old) * 10) if distance_old != 0 else 0
            reward += self.app.penalties['deplacement_eloigne_bonbon'] * increase_steps
            self.deplacements_eloigne_bonbon += 1
        else:
            # Déplacement inutile
            self.deplacements_inutiles += 1

        # Vérifier la proximité avec les dangers
        if self.is_danger_close():
            reward += self.app.penalties['proximite_danger']
            self.proximite_danger += 1

        # Gestion du chronomètre
        elapsed_time = time.time() - self.start_time
        time_remaining = max(0, self.time_limit - elapsed_time)
        self.time_remaining = time_remaining  # Stocker le temps restant pour l'affichage

        if time_remaining <= 0:
            done = True
            reward += self.app.penalties['temps_expire']
            self.reward_total += reward
            self.last_reward = reward  # Stocker la dernière récompense
            return reward, done, self.score

        if self.is_collision():
            done = True
            reward += self.app.penalties['collision']
            self.reward_total += reward
            self.last_reward = reward
            return reward, done, self.score

        if self.snake_pos == self.food_pos:
            self.score += 1
            reward += self.app.rewards['manger_bonbon']
            self.bonbons_manges += 1
            # Calculer l'intervalle de temps entre les bonbons
            interval = time.time() - self.temps_last_bonbon
            self.intervalles_bonbons.append(interval)
            self.temps_last_bonbon = time.time()
            self.food_spawn = False
            # Ne pas supprimer le dernier segment pour que le serpent grandisse
        else:
            # Toujours supprimer le dernier segment pour maintenir la taille constante
            self.snake_body.pop()

        if not self.food_spawn:
            # Générer une nouvelle nourriture
            while True:
                new_food_pos = [
                    self.x_offset + random.randrange(0, single_frame_size_x // 10) * 10,
                    self.y_offset + random.randrange(0, single_frame_size_y // 10) * 10
                ]
                if new_food_pos not in self.snake_body:
                    self.food_pos = new_food_pos
                    break
            self.food_spawn = True

        self.reward_total += reward
        self.last_reward = reward

        # Entraînement à court terme
        state_new = self.agent.get_state(self)
        self.agent.train_short_memory(state_old, move, reward, state_new, done)
        # Mémoriser
        self.agent.remember(state_old, move, reward, state_new, done)

        return reward, done, self.score

    def is_danger_close(self):
        # Vérifie s'il y a un danger à moins de 2 déplacements
        look_ahead = 2
        next_pos = self.snake_pos.copy()
        for _ in range(look_ahead):
            if self.direction == 'RIGHT':
                next_pos[0] += 10
            elif self.direction == 'LEFT':
                next_pos[0] -= 10
            elif self.direction == 'UP':
                next_pos[1] += 10
            elif self.direction == 'DOWN':
                next_pos[1] -= 10
            if self.is_collision(next_pos):
                return True
        return False

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
            'manger_bonbon': 50,
            'intervalle_bonbon': 25,
            'deplacement_vers_bonbon': 10,  # Par diminution de 10% de la distance
            'deplacement': -0.2
        }
        self.penalties = {
            'proximite_danger': -10,
            'deplacement_eloigne_bonbon': -10,  # Par augmentation de 10% de la distance
            'collision': -100,
            'temps_expire': -20
        }

        # Objectifs pour l'ajustement des récompenses
        self.target_bonbons_per_episode = 5
        self.target_avg_interval = 2.0  # En secondes
        self.target_proximite_danger = 5  # Maximum de proximité dangereuse par épisode
        self.target_deplacements_vers_bonbon = 10
        self.target_deplacements_eloigne_bonbon = 2

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

        # Afficher la génération en haut au centre de la fenêtre
        arcade.draw_text(f"Génération: {self.generation}",
                         self.width // 2, self.height - 30,
                         COLOR_TEXT, 20, anchor_x="center")

        # Afficher le meilleur score atteint jusqu'à présent
        arcade.draw_text(f"Meilleur Score: {self.best_score}",
                         self.width // 2, self.height - 60,
                         COLOR_TEXT, 16, anchor_x="center")

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

            # Sauvegarder des statistiques détaillées dans TensorBoard
            total_bonbons = sum(game.bonbons_manges for game in self.games)
            avg_bonbons = total_bonbons / len(self.games)
            avg_interval = np.mean([np.mean(game.intervalles_bonbons) if game.intervalles_bonbons else self.target_avg_interval
                                    for game in self.games])
            total_proximite_danger = sum(game.proximite_danger for game in self.games)
            avg_proximite_danger = total_proximite_danger / len(self.games)
            total_deplacements_vers = sum(game.deplacements_vers_bonbon for game in self.games)
            total_deplacements_eloigne = sum(game.deplacements_eloigne_bonbon for game in self.games)

            writer.add_scalar('Stats/Total_bonbons', total_bonbons, self.generation)
            writer.add_scalar('Stats/Average_bonbons', avg_bonbons, self.generation)
            writer.add_scalar('Stats/Average_interval_between_bonbons', avg_interval, self.generation)
            writer.add_scalar('Stats/Total_proximity_danger', total_proximite_danger, self.generation)
            writer.add_scalar('Stats/Average_proximity_danger', avg_proximite_danger, self.generation)
            writer.add_scalar('Stats/Total_deplacements_vers_bonbon', total_deplacements_vers, self.generation)
            writer.add_scalar('Stats/Total_deplacements_eloigne_bonbon', total_deplacements_eloigne, self.generation)
            writer.add_scalar('Stats/Deplacements_inutiles', sum(game.deplacements_inutiles for game in self.games), self.generation)

            # Ajouter les dynamiques des récompenses et pénalités
            writer.add_scalar('Rewards/manger_bonbon', self.rewards['manger_bonbon'], self.generation)
            writer.add_scalar('Rewards/intervalle_bonbon', self.rewards['intervalle_bonbon'], self.generation)
            writer.add_scalar('Rewards/deplacement_vers_bonbon', self.rewards['deplacement_vers_bonbon'], self.generation)
            writer.add_scalar('Rewards/deplacement', self.rewards['deplacement'], self.generation)
            writer.add_scalar('Penalties/proximite_danger', self.penalties['proximite_danger'], self.generation)
            writer.add_scalar('Penalties/deplacement_eloigne_bonbon', self.penalties['deplacement_eloigne_bonbon'], self.generation)
            writer.add_scalar('Penalties/collision', self.penalties['collision'], self.generation)
            writer.add_scalar('Penalties/temps_expire', self.penalties['temps_expire'], self.generation)

            # Enregistrer les métriques dans TensorBoard
            writer.add_scalar('Generation/Average_bonbons', avg_bonbons, self.generation)
            writer.add_scalar('Generation/Average_interval', avg_interval, self.generation)
            writer.add_scalar('Generation/Average_proximity_danger', avg_proximite_danger, self.generation)
            writer.add_scalar('Generation/Total_deplacements_vers_bonbon', total_deplacements_vers, self.generation)
            writer.add_scalar('Generation/Total_deplacements_eloigne_bonbon', total_deplacements_eloigne, self.generation)
            writer.add_scalar('Generation/Deplacements_inutiles', sum(game.deplacements_inutiles for game in self.games), self.generation)

            # Ajuster les récompenses et pénalités automatiquement
            self.adjust_rewards()

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

    def adjust_rewards(self):
        """
        Ajuste automatiquement les récompenses et pénalités en fonction des performances de l'agent.
        """
        total_bonbons = sum(game.bonbons_manges for game in self.games)
        avg_bonbons = total_bonbons / len(self.games)
        avg_interval = np.mean([np.mean(game.intervalles_bonbons) if game.intervalles_bonbons else self.target_avg_interval
                                for game in self.games])
        total_proximite_danger = sum(game.proximite_danger for game in self.games)
        avg_proximite_danger = total_proximite_danger / len(self.games)
        total_deplacements_vers = sum(game.deplacements_vers_bonbon for game in self.games)
        total_deplacements_eloigne = sum(game.deplacements_eloigne_bonbon for game in self.games)

        print(f"Ajustement des récompenses et pénalités pour la génération {self.generation}")
        print(f"Bonbons moyens par jeu: {avg_bonbons}")
        print(f"Intervalle moyen entre bonbons: {avg_interval}")
        print(f"Proximité danger moyenne: {avg_proximite_danger}")
        print(f"Total de déplacements vers bonbon: {total_deplacements_vers}")
        print(f"Total de déplacements éloignés du bonbon: {total_deplacements_eloigne}")

        # Ajuster la récompense pour manger un bonbon
        if avg_bonbons < self.target_bonbons_per_episode:
            self.rewards['manger_bonbon'] = min(self.rewards['manger_bonbon'] + 5, 100)
            print("Augmentation de la récompense pour manger un bonbon")
        elif avg_bonbons > self.target_bonbons_per_episode:
            self.rewards['manger_bonbon'] = max(self.rewards['manger_bonbon'] - 5, 20)
            print("Diminution de la récompense pour manger un bonbon")

        # Ajuster la récompense pour réduire l'intervalle entre les bonbons
        if avg_interval > self.target_avg_interval:
            self.rewards['intervalle_bonbon'] = min(self.rewards['intervalle_bonbon'] + 5, 50)
            print("Augmentation de la récompense pour réduire l'intervalle entre bonbons")
        elif avg_interval < self.target_avg_interval:
            self.rewards['intervalle_bonbon'] = max(self.rewards['intervalle_bonbon'] - 5, 10)
            print("Diminution de la récompense pour réduire l'intervalle entre bonbons")

        # Ajuster la pénalité pour proximité dangereuse
        if avg_proximite_danger > self.target_proximite_danger:
            self.penalties['proximite_danger'] = max(self.penalties['proximite_danger'] - 2, -50)
            print("Augmentation de la pénalité pour proximité dangereuse")
        elif avg_proximite_danger < self.target_proximite_danger:
            self.penalties['proximite_danger'] = min(self.penalties['proximite_danger'] + 2, -5)
            print("Diminution de la pénalité pour proximité dangereuse")

        # Ajuster la récompense pour les déplacements vers le bonbon
        expected_deplacements_vers = self.target_deplacements_vers_bonbon * len(self.games)
        if total_deplacements_vers < expected_deplacements_vers:
            self.rewards['deplacement_vers_bonbon'] = min(self.rewards['deplacement_vers_bonbon'] + 2, 20)
            print("Augmentation de la récompense pour les déplacements vers le bonbon")
        elif total_deplacements_vers > expected_deplacements_vers:
            self.rewards['deplacement_vers_bonbon'] = max(self.rewards['deplacement_vers_bonbon'] - 2, 5)
            print("Diminution de la récompense pour les déplacements vers le bonbon")

        # Ajuster la pénalité pour les déplacements s'éloignant du bonbon
        expected_deplacements_eloigne = self.target_deplacements_eloigne_bonbon * len(self.games)
        if total_deplacements_eloigne > expected_deplacements_eloigne:
            self.penalties['deplacement_eloigne_bonbon'] = max(self.penalties['deplacement_eloigne_bonbon'] - 2, -50)
            print("Augmentation de la pénalité pour les déplacements éloignés du bonbon")
        elif total_deplacements_eloigne < expected_deplacements_eloigne:
            self.penalties['deplacement_eloigne_bonbon'] = min(self.penalties['deplacement_eloigne_bonbon'] + 2, -5)
            print("Diminution de la pénalité pour les déplacements éloignés du bonbon")

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
