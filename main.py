import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import math

grid = [[0]*10 for _ in range(10)]

start_y = 0
start_x = 9

obs_start_y = random.randint(1, 6)
obs_start_x = random.randint(1, 6)

# Place obstacle 
for y in range(obs_start_y, obs_start_y + 3):
    for x in range(obs_start_x, obs_start_x + 3):
        grid[y][x] = 3

# Place goal inside the obstacle 
goal_y = random.randint(obs_start_y, obs_start_y + 2)
goal_x = random.randint(obs_start_x, obs_start_x + 2)

grid[start_y][start_x] = 1
grid[goal_y][goal_x] = 2

def distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance

def monte_carlo_path_with_bias(grid, start, goal, max_steps=200, trials=5000):
    rows, cols = len(grid), len(grid[0])
    best_path = None
    
    directions = [(0,1), (1,0), (0,-1), (-1,0)]  # right, down, left, up

    for _ in range(trials):
        path = [start]
        current = start
        visited = set([start])
        for _ in range(max_steps):
            neighbors = []
            for dy, dx in directions:
                ny, nx = current[0] + dy, current[1] + dx
                if 0 <= ny < rows and 0 <= nx < cols:
                    if grid[ny][nx] != 3 and (ny, nx) not in visited:
                        neighbors.append((ny, nx))
            
            if not neighbors:
                break
            
            neighbors.sort(key=lambda pos: distance(pos, goal))
            
            # Pick one of the closest neighbors (top 2 closest), add randomness
            best_choices = neighbors[:2] if len(neighbors) > 1 else neighbors
            current = random.choice(best_choices)
            
            path.append(current)
            visited.add(current)
            
            if current == goal:
                if best_path is None or len(path) < len(best_path):
                    best_path = path
                break
    return best_path

start = (start_y, start_x)
goal = (goal_y, goal_x)

path = monte_carlo_path_with_bias(grid, start, goal)

cmap = mcolors.ListedColormap(['white', 'blue', 'green', 'red', 'yellow'])
bounds = [0, 1, 2, 3, 4, 5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

if path:
    for y, x in path:
        if grid[y][x] == 0:
            grid[y][x] = 4 

plt.imshow(grid, cmap=cmap, norm=norm)
plt.grid(False)
plt.title("Monte Carlo Pathfinding with Goal Blocked by Obstacle")
plt.show()
