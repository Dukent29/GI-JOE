"""
Snake AI avec Vision Améliorée et Représentation Matricielle de l'Environnement
Réalisé avec Arcade et PyTorch
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

# --- Taille de la grille ---
grid_size = 10  # Chaque cellule de la grille fait 10 pixels
grid_width = single_frame_size_x // grid_size
grid_height = single_frame_size_y // grid_size

# --- Taille de la vue du serpent ---
view_size = 11  # La grille vue par le serpent sera de 11x11

# --- TensorBoard ---
writer = SummaryWriter('runs/snake_ai_arcade')

# --- Couleurs ---
COLOR_BACKGROUND = arcade.color.BLACK
COLOR_SNAKE = arcade.color.GREEN
COLOR_FOOD = arcade.color.RED
COLOR_TEXT = arcade.color.WHITE
COLOR_REWARD_POSITIVE = arcade.color.GREEN
COLOR_REWARD_NEGATIVE = arcade.color.RED
COLOR_BEST_GAME_BORDER = arcade.color.GREEN  # Couleur pour le contour du meilleur jeu
COLOR_WALL_BORDER = arcade.color.WHITE  # Couleur pour les murs
COLOR_TRAJECTORY = arcade.color.CYAN  # Couleur pour la trajectoire choisie

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
    def __init__(self, model=None, memory=None, epsilon=epsilon_start, device='cpu', writer=None):
        self.memory = deque(maxlen=max_memory_size) if memory is None else memory
        self.epsilon = epsilon
        self.device = device
        self.writer = writer  # Stocker le writer comme variable d'instance
        self.model = self.build_model().to(self.device) if model is None else model.to(self.device)
        self.target_model = self.build_model().to(self.device)
        if model is not None:
            self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.step_count = 0  # Compteur pour TensorBoard

    def build_model(self):
        # Nouveau modèle avec CNN pour traiter la grille
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # Entrée : 3 canaux (mur, corps, nourriture)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * view_size * view_size, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # Sortie : 3 actions possibles
        )
        return model

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_state(self, game):
        return game.get_state()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 2)
        state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)  # Ajouter batch_size dimension
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
        if self.step_count % 10 == 0 and self.writer:
            self.writer.add_scalar('Perte/entraînement', loss.item(), self.step_count)
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
        if self.step_count % 10 == 0 and self.writer:
            self.writer.add_scalar('Perte/entraînement_court', loss.item(), self.step_count)
        self.step_count += 1

# --- Classe du jeu avec Arcade ---
class SnakeGameAI:
    def __init__(self, agent, x_offset=0, y_offset=0):
        self.agent = agent
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.reward_total = 0
        self.last_reward = 0  # Pour stocker la dernière récompense
        self.reset()

    def reset(self):
        # Aligner la position initiale sur la grille de 10 pixels
        self.snake_pos = [
            self.x_offset + ((single_frame_size_x // 2) // grid_size * grid_size),
            self.y_offset + ((single_frame_size_y // 2) // grid_size * grid_size)
        ]
        self.snake_body = [
            self.snake_pos[:],
            [self.snake_pos[0] - grid_size, self.snake_pos[1]],
            [self.snake_pos[0] - 2 * grid_size, self.snake_pos[1]]
        ]
        self.food_spawn = True
        self.direction = 'RIGHT'
        self.score = 0
        self.time_limit = 20  # Augmenter le temps limite du jeu
        self.start_time = time.time()
        self.next_direction = self.direction  # Initialiser la prochaine direction

        # Générer une position de nourriture qui ne chevauche pas le serpent
        while True:
            self.food_pos = [
                self.x_offset + random.randrange(0, grid_width) * grid_size,
                self.y_offset + random.randrange(0, grid_height) * grid_size
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
            self.snake_pos[0] += grid_size
        elif self.direction == 'LEFT':
            self.snake_pos[0] -= grid_size
        elif self.direction == 'UP':
            self.snake_pos[1] += grid_size
        elif self.direction == 'DOWN':
            self.snake_pos[1] -= grid_size

    def get_state(self):
        # Créer une grille centrée sur la tête du serpent
        state = np.zeros((3, view_size, view_size), dtype=int)  # 3 canaux : mur, corps, nourriture
        center = view_size // 2

        for i in range(view_size):
            for j in range(view_size):
                x = self.snake_pos[0] + (j - center) * grid_size
                y = self.snake_pos[1] + (i - center) * grid_size

                # Vérifier les limites du cadre
                if x < self.x_offset or x >= self.x_offset + single_frame_size_x or \
                        y < self.y_offset or y >= self.y_offset + single_frame_size_y:
                    state[0, i, j] = 1  # Mur
                elif [x, y] in self.snake_body[1:]:
                    state[1, i, j] = 1  # Corps du serpent
                elif [x, y] == self.food_pos:
                    state[2, i, j] = 1  # Nourriture

        return state

    def calculate_distance_to_food(self, position=None):
        if position is None:
            position = self.snake_pos
        return np.linalg.norm(np.array(position) - np.array(self.food_pos))

    def play_step(self, delta_time, current_generation):
        state_old = self.agent.get_state(self)
        move = self.agent.act(state_old)
        self.last_action = move  # Stocker la dernière action

        previous_distance = self.calculate_distance_to_food()

        self.move(move)
        self.snake_body.insert(0, self.snake_pos[:])

        reward = 0
        done = False

        # Calculer la nouvelle distance à la nourriture
        new_distance = self.calculate_distance_to_food()

        # Récompense si on se rapproche de la nourriture
        if new_distance < previous_distance:
            reward += 1
        else:
            reward -= 1

        # Pénalité pour chaque mouvement
        reward -= 0.1

        # Gestion du chronomètre
        elapsed_time = time.time() - self.start_time
        time_remaining = max(0, self.time_limit - elapsed_time)
        self.time_remaining = time_remaining  # Stocker le temps restant pour l'affichage

        if time_remaining <= 0:
            done = True
            reward -= 20
            self.reward_total += reward
            self.last_reward = reward  # Stocker la dernière récompense
            return reward, done, self.score

        if self.is_collision():
            done = True
            reward -= 100
            self.reward_total += reward
            self.last_reward = reward
            return reward, done, self.score

        if self.snake_pos == self.food_pos:
            self.score += 1
            reward += 50
            self.food_spawn = False
            # Ne pas supprimer le dernier segment pour que le serpent grandisse
        else:
            # Supprimer le dernier segment pour maintenir la taille si pas de nourriture mangée
            self.snake_body.pop()

        if not self.food_spawn:
            # Générer une nouvelle nourriture
            while True:
                new_food_pos = [
                    self.x_offset + random.randrange(0, grid_width) * grid_size,
                    self.y_offset + random.randrange(0, grid_height) * grid_size
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

# --- Classe principale de l'application ---
class SnakeAIApp(arcade.Window):
    def __init__(self):
        super().__init__(screen_width, screen_height, "Snake AI - Version Arcade", update_rate=difficulty)
        arcade.set_background_color(COLOR_BACKGROUND)

        # Empêcher la mise en veille du système
        prevent_sleep()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Utilisation du périphérique : {self.device}")

        # Initialiser les variables
        self.generation = 0
        self.best_score = 0  # Meilleur score atteint jusqu'à présent
        self.best_game_index = None  # Index du meilleur jeu de la génération

        # Charger l'agent et l'état si le fichier existe
        self.agent = None
        self.load_agent()

        # Créer les jeux
        self.games = []
        for row in range(rows):
            for col in range(cols):
                x_offset = col * single_frame_size_x
                y_offset = row * single_frame_size_y
                game = SnakeGameAI(self.agent, x_offset, y_offset)
                self.games.append(game)

        self.done_flags = [False] * len(self.games)

        # Pour calculer le temps par génération
        self.generation_start_time = time.time()

    def load_agent(self):
        try:
            with open("best_agent.pkl", "rb") as f:
                data = pickle.load(f)
                model_state_dict = data['model_state_dict']
                memory = data['memory']
                epsilon = data['epsilon']
                self.generation = data.get('generation', 0)
                self.best_score = data.get('best_score', 0)
                # Initialiser l'agent avec le modèle chargé
                model = self.build_model().to(self.device)
                model.load_state_dict(model_state_dict)
                self.agent = DQNAgent(model=model, memory=memory, epsilon=epsilon, device=self.device, writer=writer)
                print("Agent chargé depuis le fichier.")
        except FileNotFoundError:
            print("Aucun agent précédent trouvé, création d'un nouvel agent.")
            self.agent = DQNAgent(device=self.device, writer=writer)

    def build_model(self):
        # Doit correspondre à la méthode build_model de DQNAgent
        model = DQNAgent().build_model()
        return model

    def on_draw(self):
        arcade.start_render()
        # Dessiner tous les jeux
        for i, game in enumerate(self.games):
            # Dessiner le cadre du meilleur jeu
            if i == self.best_game_index:
                arcade.draw_rectangle_outline(
                    game.x_offset + single_frame_size_x / 2,
                    game.y_offset + single_frame_size_y / 2,
                    single_frame_size_x - 2,
                    single_frame_size_y - 2,
                    COLOR_BEST_GAME_BORDER,
                    border_width=5
                )

            # Dessiner les murs (bordures blanches)
            arcade.draw_rectangle_outline(
                game.x_offset + single_frame_size_x / 2,
                game.y_offset + single_frame_size_y / 2,
                single_frame_size_x,
                single_frame_size_y,
                COLOR_WALL_BORDER,
                border_width=2
            )

            # Dessiner le serpent
            for pos in game.snake_body:
                arcade.draw_rectangle_filled(pos[0] + grid_size / 2, pos[1] + grid_size / 2, grid_size, grid_size, COLOR_SNAKE)
            # Dessiner la nourriture
            arcade.draw_rectangle_filled(game.food_pos[0] + grid_size / 2, game.food_pos[1] + grid_size / 2, grid_size, grid_size, COLOR_FOOD)
            # Afficher le score
            arcade.draw_text(f"Score : {game.score}", game.x_offset + 10, game.y_offset + single_frame_size_y - 20,
                             COLOR_TEXT, 12)

            # Afficher le temps restant
            arcade.draw_text(f"Temps restant : {int(game.time_remaining)}s",
                             game.x_offset + 10, game.y_offset + single_frame_size_y - 40,
                             COLOR_TEXT, 12)

            # Afficher la récompense totale
            arcade.draw_text(f"Récompense totale : {round(game.reward_total, 2)}",
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

            # Dessiner la trajectoire choisie
            direction_vectors = {
                'RIGHT': (grid_size, 0),
                'DOWN': (0, -grid_size),
                'LEFT': (-grid_size, 0),
                'UP': (0, grid_size)
            }
            dx, dy = direction_vectors[game.direction]
            start_x = game.snake_pos[0] + grid_size / 2
            start_y = game.snake_pos[1] + grid_size / 2
            end_x = start_x + dx
            end_y = start_y + dy
            arcade.draw_line(start_x, start_y, end_x, end_y, COLOR_TRAJECTORY, 2)

        # Afficher la génération en haut au centre de la fenêtre
        arcade.draw_text(f"Génération : {self.generation}",
                         self.width // 2, self.height - 30,
                         COLOR_TEXT, 20, anchor_x="center")

        # Afficher le meilleur score atteint jusqu'à présent
        arcade.draw_text(f"Meilleur Score : {self.best_score}",
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
                print("Mise à jour du réseau cible...")
                self.agent.update_target_model()

            # Collecter les récompenses totales et les scores de chaque jeu
            rewards = [game.reward_total for game in self.games]
            scores = [game.score for game in self.games]
            avg_reward = sum(rewards) / len(rewards)
            max_reward = max(rewards)
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            deaths = len([1 for game in self.games if game.score == 0])

            generation_time = time.time() - self.generation_start_time

            print(f"Récompenses des jeux : {rewards}")
            print(f"Scores des jeux : {scores}")
            print(f"Récompense moyenne : {avg_reward}")
            print(f"Récompense maximale : {max_reward}")
            print(f"Score moyen de la génération : {avg_score}")
            print(f"Meilleur score de la génération : {max_score}")
            print(f"Morts : {deaths}")
            print(f"Temps pour la génération : {generation_time} secondes")

            # Mettre à jour le meilleur score global si nécessaire
            if max_score > self.best_score:
                self.best_score = max_score

            # Identifier l'index du meilleur jeu
            self.best_game_index = scores.index(max_score)

            # Enregistrer les récompenses moyennes et maximales dans TensorBoard
            if writer:
                writer.add_scalar('Récompense/moyenne', avg_reward, self.generation)
                writer.add_scalar('Récompense/maximale', max_reward, self.generation)
                writer.add_scalar('Score/moyen', avg_score, self.generation)
                writer.add_scalar('Score/maximal', max_score, self.generation)
                writer.add_scalar('Morts', deaths, self.generation)
                writer.add_scalar('Temps/génération', generation_time, self.generation)

            # Mettre à jour les paramètres de l'agent
            self.agent.epsilon = max(min_epsilon, self.agent.epsilon * epsilon_decay)

            # Préparer la prochaine génération
            self.generation += 1
            self.done_flags = [False] * len(self.games)
            for game in self.games:
                game.reward_total = 0
                game.last_reward = 0
                game.reset()

            # Redémarrer le temps pour la nouvelle génération
            self.generation_start_time = time.time()

    def on_close(self):
        print("Fermeture du jeu, sauvegarde en cours...")
        # Sauvegarder l'agent partagé
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
        print(f"Agent sauvegardé à la génération {self.generation} avec le meilleur score {self.best_score}.")
        if writer:
            writer.close()

        # Permettre la mise en veille du système
        allow_sleep()

        super().on_close()

# --- Fonction principale ---
def main():
    app = SnakeAIApp()
    arcade.run()

if __name__ == '__main__':
    main()
