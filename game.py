"""
Snake Eater
Made with PyGame
"""

import pygame, sys, time, random, os, csv, json

# Difficulty settings
difficulty = 25

# Window size
frame_size_x = 720
frame_size_y = 480

# Checks for errors encountered
check_errors = pygame.init()
if check_errors[1] > 0:
    print(f'[!] Had {check_errors[1]} errors when initialising game, exiting...')
    sys.exit(-1)
else:
    print('[+] Game successfully initialised')

# Initialise game window
pygame.display.set_caption('Snake Eater')
game_window = pygame.display.set_mode((frame_size_x, frame_size_y))

# Colors (R, G, B)
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)

# FPS (frames per second) controller
fps_controller = pygame.time.Clock()

# Game variables
snake_pos = [100, 50]
snake_body = [[100, 50], [90, 50], [80, 50]]

food_pos = [random.randrange(1, (frame_size_x // 10)) * 10, random.randrange(1, (frame_size_y // 10)) * 10]
food_spawn = True

direction = 'RIGHT'
change_to = direction

score = 0

# Tracking variables for data
generation_num = 1  # Numéro de génération
start_time = time.time()  # Temps de début de la partie
positions = []  # Historique des positions du serpent
food_positions = []  # Historique des positions des bonbons

# File management setup
base_path = "data/csv/temporary/"
generation_path = os.path.join(base_path, f"generation-{generation_num}")
os.makedirs(generation_path, exist_ok=True)

def get_next_game_number(directory):
    """Trouve le prochain numéro de partie basé sur les colonnes `Partie` des fichiers CSV."""
    max_game_number = 0  # Initialiser le numéro maximum à 0
    # Parcourt tous les fichiers CSV existants
    for file in os.listdir(directory):
        if file.startswith("tableau-") and file.endswith(".csv"):
            file_path = os.path.join(directory, file)
            with open(file_path, mode="r", newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                # Parcourt toutes les lignes du fichier pour trouver le plus grand numéro de partie
                for row in reader:
                    game_number = int(row["Partie"])  # Lire la valeur dans la colonne `Partie`
                    if game_number > max_game_number:
                        max_game_number = game_number
    return max_game_number + 1  # Retourne le prochain numéro

# Trouver le prochain numéro de partie
game_num = get_next_game_number(generation_path)

# Définir le chemin du fichier CSV pour la nouvelle partie
game_file_path = os.path.join(generation_path, f"tableau-{game_num}.csv")

# Créer le fichier et écrire les en-têtes
with open(game_file_path, mode="w", newline="") as csvfile:
    fieldnames = ["Partie", "Score", "Positionnement", "Bonbon", "Temps"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

# Game Over
def game_over():
    global positions, food_positions, start_time

    my_font = pygame.font.SysFont('times new roman', 90)
    game_over_surface = my_font.render('YOU DIED', True, red)
    game_over_rect = game_over_surface.get_rect()
    game_over_rect.midtop = (frame_size_x / 2, frame_size_y / 4)
    game_window.fill(black)
    game_window.blit(game_over_surface, game_over_rect)
    show_score(0, red, 'times', 20)
    pygame.display.flip()

    # Log final data
    with open(game_file_path, mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Partie", "Score", "Positionnement", "Bonbon", "Temps"])
        writer.writerow({
            "Partie": game_num,  # Utiliser le numéro de partie calculé
            "Score": score,
            "Positionnement": json.dumps(positions),
            "Bonbon": json.dumps(food_positions),
            "Temps": round(time.time() - start_time, 2)
        })

    time.sleep(3)
    pygame.quit()
    sys.exit()

# Score
def show_score(choice, color, font, size):
    score_font = pygame.font.SysFont(font, size)
    score_surface = score_font.render('Score : ' + str(score), True, color)
    score_rect = score_surface.get_rect()
    if choice == 1:
        score_rect.midtop = (frame_size_x / 10, 15)
    else:
        score_rect.midtop = (frame_size_x / 2, frame_size_y / 1.25)
    game_window.blit(score_surface, score_rect)

# Main logic
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP or event.key == ord('z'):
                change_to = 'UP'
            if event.key == pygame.K_DOWN or event.key == ord('s'):
                change_to = 'DOWN'
            if event.key == pygame.K_LEFT or event.key == ord('q'):
                change_to = 'LEFT'
            if event.key == pygame.K_RIGHT or event.key == ord('d'):
                change_to = 'RIGHT'
            if event.key == pygame.K_ESCAPE:
                pygame.event.post(pygame.event.Event(pygame.QUIT))

    if change_to == 'UP' and direction != 'DOWN':
        direction = 'UP'
    if change_to == 'DOWN' and direction != 'UP':
        direction = 'DOWN'
    if change_to == 'LEFT' and direction != 'RIGHT':
        direction = 'LEFT'
    if change_to == 'RIGHT' and direction != 'LEFT':
        direction = 'RIGHT'

    if direction == 'UP':
        snake_pos[1] -= 10
    if direction == 'DOWN':
        snake_pos[1] += 10
    if direction == 'LEFT':
        snake_pos[0] -= 10
    if direction == 'RIGHT':
        snake_pos[0] += 10

    # Tracking positions
    positions.append({"x": snake_pos[0], "y": snake_pos[1]})
    food_positions.append({"x": food_pos[0], "y": food_pos[1]})

    snake_body.insert(0, list(snake_pos))
    if snake_pos[0] == food_pos[0] and snake_pos[1] == food_pos[1]:
        score += 1
        food_spawn = False
    else:
        snake_body.pop()

    if not food_spawn:
        food_pos = [random.randrange(1, (frame_size_x // 10)) * 10, random.randrange(1, (frame_size_y // 10)) * 10]
        food_spawn = True

    game_window.fill(black)
    for pos in snake_body:
        pygame.draw.rect(game_window, green, pygame.Rect(pos[0], pos[1], 10, 10))
    pygame.draw.rect(game_window, white, pygame.Rect(food_pos[0], food_pos[1], 10, 10))

    if snake_pos[0] < 0 or snake_pos[0] > frame_size_x - 10:
        game_over()
    if snake_pos[1] < 0 or snake_pos[1] > frame_size_y - 10:
        game_over()
    for block in snake_body[1:]:
        if snake_pos[0] == block[0] and snake_pos[1] == block[1]:
            game_over()

    show_score(1, white, 'consolas', 20)
    pygame.display.update()
    fps_controller.tick(difficulty)
