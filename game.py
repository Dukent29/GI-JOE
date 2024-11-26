"""
Snake AI with Persistent Memory and Model
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
import time  # Importer le module time pour utiliser time.time()

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
COLOR_BEST_GAME_BORDER = arcade.color.GREEN  # Couleur pour le contour du meilleur jeu

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
        self.score = 0
        self.time_limit = 20  # Temps limite par partie, en secondes
        self.start_time = time.time()  # Utiliser time.time() au lieu de arcade.get_time()

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
            self.snake_pos[1] += 10
        elif self.direction == 'DOWN':
            self.snake_pos[1] -= 10

    def play_step(self, delta_time, current_generation):
        state_old = self.agent.get_state(self)
        move = self.agent.act(state_old)
        action = [0, 0, 0]
        action[move] = 1

        # Calcul de la distance avant le mouvement
        distance_old = np.linalg.norm(np.array(self.snake_pos) - np.array(self.food_pos))

        self.move(action)
        self.snake_body.insert(0, self.snake_pos[:])

        reward = -0.1  # Pénalité pour chaque mouvement
        done = False

        # Calcul de la distance après le mouvement
        distance_new = np.linalg.norm(np.array(self.snake_pos) - np.array(self.food_pos))

        # Récompense ou punition basée sur le déplacement
        delta_distance = distance_old - distance_new
        reward += delta_distance * 0.1

        # Gestion du chronomètre
        elapsed_time = time.time() - self.start_time  # Utiliser time.time() ici aussi
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
        else:
            pass

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

# --- Classe principale de l'application ---
class SnakeAIApp(arcade.Window):
    def __init__(self):
        super().__init__(screen_width, screen_height, "Snake AI - Arcade Version", update_rate=difficulty)
        arcade.set_background_color(COLOR_BACKGROUND)

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

        # Définir le périphérique (GPU si disponible)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialiser l'agent
        if best_model_state_dict is not None:
            self.agent = DQNAgent(model=None, memory=None, epsilon=best_epsilon, device=self.device)
            self.agent.model.load_state_dict(best_model_state_dict)
            self.agent.update_target_model()
            if best_memory is not None:
                self.agent.memory = deque(best_memory, maxlen=max_memory_size)
        else:
            self.agent = DQNAgent(device=self.device)

        self.generation = 0
        self.best_score = 0  # Meilleur score atteint jusqu'à présent
        self.best_game_index = None  # Index du meilleur jeu de la génération

        # Créer les jeux
        self.games = []
        for row in range(rows):
            for col in range(cols):
                x_offset = col * single_frame_size_x
                y_offset = row * single_frame_size_y
                game = SnakeGameAI(self.agent, x_offset, y_offset)
                self.games.append(game)

        self.done_flags = [False] * len(self.games)

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

            # Enregistrer les récompenses moyennes et maximales dans TensorBoard
            writer.add_scalar('Reward/avg', avg_reward, self.generation)
            writer.add_scalar('Reward/max', max_reward, self.generation)
            writer.add_scalar('Score/max', max_score, self.generation)

            # Mettre à jour les paramètres de l'agent
            best_model_state_dict = self.agent.model.state_dict()
            best_memory = self.agent.memory
            best_epsilon = max(min_epsilon, self.agent.epsilon * epsilon_decay)
            self.agent.epsilon = best_epsilon

            # Sauvegarder l'agent partagé
            with open("best_agent.pkl", "wb") as f:
                pickle.dump({
                    'model_state_dict': best_model_state_dict,
                    'memory': best_memory,
                    'epsilon': best_epsilon
                }, f)
            print("Meilleur agent sauvegardé dans best_agent.pkl")

            # Préparer la prochaine génération
            self.generation += 1
            self.done_flags = [False] * len(self.games)
            for game in self.games:
                game.reward_total = 0
                game.last_reward = 0
                game.reset()

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
                'epsilon': best_epsilon
            }, f)
        print("Meilleur agent sauvegardé. Programme terminé.")
        writer.close()
        super().on_close()

# --- Fonction principale ---
def main():
    app = SnakeAIApp()
    arcade.run()

if __name__ == '__main__':
    main()
