import tensorflow as tf
import math
import random
import sys
import os
import pygame
import numpy as np
import tensorflow
import time
import matplotlib.pyplot as plt
import cProfile as profile

"""**Class NetStructure**"""

class NetStructure:
    def __init__(self, input_dim, output_dim):
        self.input_dim  = input_dim
        self.n_hidden   = 0
        self.hidden_dim = []
        self.output_dim = output_dim
        self.activation = []

    def add_hidden(self, hidden_dim, activation = 'linear'):
        self.n_hidden += 1
        self.hidden_dim.append(hidden_dim)
        self.activation.append(activation)

    def get_input_dim(self):
        return self.input_dim

    def get_output_dim(self):
        return self.output_dim

    def get_num_hidden(self):
        return self.n_hidden

    def get_hidden_dim(self, index):
        return self.hidden_dim[index]

    def get_activation(self, index):
        return self.activation[index]

    def print(self):
        print("----------------------")
        print("    Input dim:", self.input_dim)
        for i in range(self.n_hidden):
            print(" Hidden", i+1, "dim:", self.hidden_dim[i], "- activation:", self.activation[i])
        print("   Output dim:", self.output_dim)
        print("----------------------")

"""Class Car"""

# Constants
# WIDTH = 1600
# HEIGHT = 880

WIDTH = 1920
HEIGHT = 1080

CAR_SIZE_X = 60
CAR_SIZE_Y = 60

BORDER_COLOR = (255, 255, 255, 255) # Color To Crash on Hit

current_generation = 0 # Generation counter

class Car:

    def __init__(self):
        # Load Car Sprite and Rotate
        self.sprite = pygame.image.load('car2.png').convert() # Convert Speeds Up A Lot
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite

        # self.position = [690, 740] # Starting Position
        self.position = [830, 920] # Starting Position
        self.angle = 0
        self.speed = 0

        self.speed_set = False # Flag For Default Speed Later on

        self.center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2] # Calculate Center

        self.radars = [] # List For Sensors / Radars
        self.drawing_radars = [] # Radars To Be Drawn

        self.alive = True # Boolean To Check If Car is Crashed

        self.distance = 0 # Distance Driven
        self.time = 0 # Time Passed

    def draw(self, screen):
        screen.blit(self.rotated_sprite, self.position) # Draw Sprite
        self.draw_radar(screen) #OPTIONAL FOR SENSORS

    def draw_radar(self, screen):
        # Optionally Draw All Sensors / Radars
        for radar in self.radars:
            position = radar[0]
            pygame.draw.line(screen, (0, 255, 0), self.center, position, 1)
            pygame.draw.circle(screen, (0, 255, 0), position, 5)

    def check_collision(self, game_map):
        self.alive = True
        for point in self.corners:
            # If Any Corner Touches Border Color -> Crash
            # Assumes Rectangle
            if game_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR:
                self.alive = False
                break

    def check_radar(self, degree, game_map):
        length = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        # While We Don't Hit BORDER_COLOR AND length < 300 (just a max) -> go further and further
        while not game_map.get_at((x, y)) == BORDER_COLOR and length < 300:
            length = length + 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        # Calculate Distance To Border And Append To Radars List
        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])

    def update(self, game_map):
        # Set The Speed To 20 For The First Time
        # Only When Having 4 Output Nodes With Speed Up and Down
        if not self.speed_set:
            self.speed = 20
            self.speed_set = True

        # Get Rotated Sprite And Move Into The Right X-Direction
        # Don't Let The Car Go Closer Than 20px To The Edge
        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[0] = max(self.position[0], 20)
        self.position[0] = min(self.position[0], WIDTH - 120)

        # Increase Distance and Time
        self.distance += self.speed
        self.time += 1

        # Same For Y-Position
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.position[1] = max(self.position[1], 20)
        self.position[1] = min(self.position[1], WIDTH - 120)

        # Calculate New Center
        self.center = [int(self.position[0]) + CAR_SIZE_X / 2, int(self.position[1]) + CAR_SIZE_Y / 2]

        # Calculate Four Corners
        # Length Is Half The Side
        length = 0.5 * CAR_SIZE_X
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length]
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length]
        self.corners = [left_top, right_top, left_bottom, right_bottom]

        # Check Collisions And Clear Radars
        self.check_collision(game_map)
        self.radars.clear()

        # From -90 To 120 With Step-Size 45 Check Radar
        for d in range(-90, 120, 45):
            self.check_radar(d, game_map)

    def get_data(self):
        # Get Distances To Border
        radars = self.radars
        return_values = [0, 0, 0, 0, 0]
        for i, radar in enumerate(radars):
            return_values[i] = int(radar[1] / 30)

        return return_values

    def is_alive(self):
        # Basic Alive Function
        return self.alive

    def get_reward(self):
        # Calculate Reward (Maybe Change?)
        # return self.distance / 50.0
        return self.distance / 50.0

    def rotate_center(self, image, angle):
        # Rotate The Rectangle
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image
    

class Meta:
    def __init__(self, net_structure):
        self.net_structure  = net_structure

        # definisco il modello sulla base della struttura che gli ho passato
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(net_structure.get_hidden_dim(0), activation=net_structure.get_activation(0), input_dim=net_structure.get_input_dim()))
        for i in range(1, net_structure.get_num_hidden()):
            self.model.add(tf.keras.layers.Dense(net_structure.get_hidden_dim(i), activation=net_structure.get_activation(i)))
        self.model.add(tf.keras.layers.Dense(net_structure.get_output_dim()))

        # salvo il numero di parametri del modello
        self.num_parameters = self.model.count_params()

        # dominio di default per i latent points
        self.domain = [-1, 1]

    def get_model(self):
        return self.model

    def set_num_iterations(self, num_iterations):
        self.num_iterations = num_iterations

    def set_population_size(self, population_size):
        self.population_size = population_size

    def set_latent_points_domain(self, domain):
        self.domain = domain

    def is_in_domain(self, x):
        if (x < self.domain[0] or x > self.domain[1]):
            return False
        return True

    def generate_latent_points(self, n_samples):
        self.latent_points = np.random.uniform(self.domain[0], self.domain[1], (n_samples, self.net_structure.get_input_dim()))

    def update_model_with_parameters(self, opt_par):
        nl = len(self.model.layers)
        wbindex = 0
        for p in range(0, nl):
          W = opt_par[wbindex:(wbindex + self.model.layers[p].input.shape[1] * self.model.layers[p].output.shape[1])]
          b = opt_par[(wbindex + self.model.layers[p].input.shape[1] * self.model.layers[p].output.shape[1]):(wbindex + self.model.layers[p].count_params())]
          self.model.layers[p].set_weights([W.reshape(self.model.layers[p].input.shape[1], self.model.layers[p].output.shape[1]), b])
          wbindex = (wbindex + self.model.layers[p].count_params())

    #def objective_function(self, x = None):
       # if (x == None):
       # return sum


    # objective_function = run_simulation which returns car.get_reward()
    def run_simulation(self, particles):
        # Empty Collections For Nets and Cars

        nets = []
        cars = []
        rewards = []

        #cambiare
        len_population = self.population_size  # Number Of Agents In Population

        #initialization of 70 cars

        for i in range(len_population):
            my_car = Car()
            cars.append(my_car)



        # Clock Settings
        # Font Settings & Loading Map
        clock = pygame.time.Clock()
        generation_font = pygame.font.SysFont("Arial", 30)
        alive_font = pygame.font.SysFont("Arial", 20)
        game_map = pygame.image.load('map2.png').convert() # Convert Speeds Up A Lot

        global current_generation
        current_generation += 1

        # Simple Counter To Roughly Limit Time (Not Good Practice)
        counter = 0

        while True:
            # Exit On Quit Event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit(0)

            for i, car in enumerate(cars):
                # i changed variable name from genome to chromosome (because we are selecting only a row of population which represent a single chromose)
                particle = particles[i,]
                best_index = 0

                wbindex = 0
                nl = len(self.model.layers)

                for p in range(0, nl): #Generazione della rete
                    W = particle[wbindex:(wbindex + self.model.layers[p].input.shape[1] * self.model.layers[p].output.shape[1])]
                    b = particle[(wbindex + self.model.layers[p].input.shape[1] * self.model.layers[p].output.shape[1]):(wbindex + self.model.layers[p].count_params())]
                    self.model.layers[p].set_weights([W.reshape(self.model.layers[p].input.shape[1], self.model.layers[p].output.shape[1]), b])
                    wbindex = (wbindex + self.model.layers[p].count_params())



                # evaluate each ANN and perform the "predicted" action
                # "choice" represent index node with highest output
                # reshape input data
                output = self.model.predict(np.resize(car.get_data(), (1, self.net_structure.get_input_dim())), verbose = 0)
                choice = np.array(output).argmax()
                # print(choice)
                if choice == 0:
                    car.angle += 10 # Left
                elif choice == 1:
                    car.angle -= 10 # Right
                elif choice == 2:
                    if(car.speed - 2 >= 12):
                        car.speed -= 2 # Slow Down
                else:
                    car.speed += 2 # Speed Up

            # Check If Car Is Still Alive
            # Increase Fitness If Yes And Break Loop If Not (fitness += car.get_reward()!! NB not a cost!)
            still_alive = 0
            for i, car in enumerate(cars):
                if car.is_alive():
                    still_alive += 1
                    car.update(game_map)

            if still_alive == 0:
                break

            counter += 1
            if counter == 30 * 40: # Stop After About 20 Seconds
                break

            # Draw Map And All Cars That Are Alive
            screen.blit(game_map, (0, 0))
            for car in cars:
                if car.is_alive():
                    car.draw(screen)

            # Display Info
            text = generation_font.render("Generation: " + str(current_generation), True, (0,0,0))
            text_rect = text.get_rect()
            text_rect.center = (900, 450)
            screen.blit(text, text_rect)

            text = alive_font.render("Still Alive: " + str(still_alive), True, (0, 0, 0))
            text_rect = text.get_rect()
            text_rect.center = (900, 490)
            screen.blit(text, text_rect)

            pygame.display.flip()
            clock.tick(60) # 60 FPS
            
            
        for car in (cars):
            rewards.append(car.get_reward())

        print(rewards)
        return rewards  
    
    
    def test_newmap(self, optimized_parameters):
        
        rewards=[]
        
        car_1 = Car()
        
        # Clock Settings
        # Font Settings & Loading Map
        clock = pygame.time.Clock()
        generation_font = pygame.font.SysFont("Arial", 30)
        alive_font = pygame.font.SysFont("Arial", 20)
        game_map = pygame.image.load('map4.png').convert() # Convert Speeds Up A Lot
        
        global current_generation
        current_generation += 1
        
        counter = 0
        
        while True:
            # Exit On Quit Event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit(0)
                
  
                # i changed variable name from genome to chromosome (because we are selecting only a row of population which represent a single chromose)
                particle = optimized_parameters
                best_index = 0

                wbindex = 0
                nl = len(self.model.layers)

                for p in range(0, nl): #Generazione della rete
                    W = particle[wbindex:(wbindex + self.model.layers[p].input.shape[1] * self.model.layers[p].output.shape[1])]
                    b = particle[(wbindex + self.model.layers[p].input.shape[1] * self.model.layers[p].output.shape[1]):(wbindex + self.model.layers[p].count_params())]
                    self.model.layers[p].set_weights([W.reshape(self.model.layers[p].input.shape[1], self.model.layers[p].output.shape[1]), b])
                    wbindex = (wbindex + self.model.layers[p].count_params())



                # evaluate each ANN and perform the "predicted" action
                # "choice" represent index node with highest output
                # reshape input data
                output = self.model.predict(np.resize(car_1.get_data(), (1, self.net_structure.get_input_dim())), verbose = 0)
                choice = np.array(output).argmax()
                # print(choice)
                if choice == 0:
                    car_1.angle += 10 # Left
                elif choice == 1:
                    car_1.angle -= 10 # Right
                elif choice == 2:
                    if(car_1.speed - 2 >= 12):
                        car_1.speed -= 2 # Slow Down
                else:
                    car_1.speed += 2 # Speed Up

            # Check If Car Is Still Alive
            # Increase Fitness If Yes And Break Loop If Not (fitness += car.get_reward()!! NB not a cost!)
            still_alive = 0
        
            if car_1.is_alive():
                    still_alive = 1
                    car_1.update(game_map)

            if still_alive == 0:
                break

            counter += 1
            if counter == 30 * 40: # Stop After About 20 Seconds
                #break

            # Draw Map And All Cars That Are Alive
                screen.blit(game_map, (0, 0))

            if car_1.is_alive():
                    car_1.draw(screen)

            # Display Info
            text = generation_font.render("Generation: " + str(current_generation), True, (0,0,0))
            text_rect = text.get_rect()
            text_rect.center = (900, 450)
            screen.blit(text, text_rect)

            text = alive_font.render("Still Alive: " + str(still_alive), True, (0, 0, 0))
            text_rect = text.get_rect()
            text_rect.center = (900, 490)
            screen.blit(text, text_rect)

            pygame.display.flip()
            clock.tick(60) # 60 FPS

            
            rewards.append(car_1.get_reward())
        
        return rewards

        
        


    
class PSO(Meta):
    def __init__(self, net_structure):
        super().__init__(net_structure)
        self.w  = 0.3 # inertia_param
        self.c1 = 1.5 # cognitive_param
        self.c2 = 1.5 # social_param

    def set_options(self, inertia_param = 0.3,
                    cognitive_param = 1.5,
                    social_param = 1.5):
        self.w  = inertia_param
        self.c1 = cognitive_param
        self.c2 = social_param

    def set_max_v(self, max_v):
        self.max_v = max_v

    def set_max_x(self, max_x):
        self.max_x = max_x

    def update_velocity(self, position, velocity, best_position, global_best_position):
        inertia = self.w * velocity
        cognitive_component = self.c1 * np.random.rand(1, len(position)) * (best_position - position)
        social_component = self.c2 * np.random.rand(1, len(position)) * (global_best_position - position)
        new_velocity = inertia + cognitive_component + social_component
        return new_velocity

    def optimize(self):
        # ATTENZIONE: è necessario aver generato i latent points in precedenza
        particles  = np.random.uniform(low=-self.max_x, high=self.max_x, size=(self.population_size, self.num_parameters))
        velocities = np.random.uniform(low=-self.max_v, high=self.max_v, size=(self.population_size, self.num_parameters))
        best_positions = np.copy(particles)
        best_scores = np.array([self.num_parameters] * self.population_size)
        global_best_position = None
        global_best_score = 0
        nl = len(self.model.layers)

        for iteration in range(self.num_iterations):
            tic_global = time.perf_counter()

            fitness = self.run_simulation(particles)
    
            
            for i in range(self.population_size):
                #print(fitness[i])

                # print(fitness)

                if  fitness[i] > best_scores[i]:
                    best_scores[i] = fitness[i]
                    best_positions[i] = np.copy(particles[i])

                if  fitness[i] > global_best_score:
                    global_best_score = fitness[i]
                    global_best_position = np.copy(particles[i])

                velocities[i] = self.update_velocity(particles[i], velocities[i], best_positions[i], global_best_position)
                particles[i] += velocities[i]

            # mi assicuro che le velocità e posizioni siano nei range
            velocities = np.minimum(velocities,  self.max_v)
            velocities = np.maximum(velocities, -self.max_v)
            particles  = np.minimum(particles,  self.max_x)
            particles  = np.maximum(particles, -self.max_x)

            toc_global = time.perf_counter()
            print("Iteration #%d - Objective function value: %5.2f - time: %0.3f" % (iteration, global_best_score, toc_global - tic_global))

            # se l'errore fa a zero mi fermo
            if (global_best_score == 0):
                break

        return global_best_position

    def predict(self, x = None):
        if (x == None):
            x = self.latent_points
        return self.model.predict(x)
    

"""**MAIN**"""

if __name__ == "__main__":

    # Here create a "population" of agents (GA, PSO, etc) and run simulation
    #...

    # Initializion of ANN structure by methods of class net_structure

    net = NetStructure(input_dim = 5, output_dim = 4)
    net.add_hidden(hidden_dim = 10, activation = 'softmax')

    # Initialization of GA parameters
    meta = PSO(net)
    meta.set_num_iterations(20)
    meta.set_population_size(15)
    meta.set_max_x(1.5)
    meta.set_max_v(0.3)

    # Initialize PyGame And The Display
    pygame.init()
    screen = pygame.display.set_mode((WIDTH/2, HEIGHT/2), pygame.RESIZABLE)

    # Optimization
    tic_global = time.perf_counter()
    optimized_parameters = meta.optimize()
    toc_global = time.perf_counter()
    
    test = meta.test_newmap(optimized_parameters)

    print("Time: %0.3f" % (toc_global - tic_global))
    
    #self.test_newmap(optimized_parameters)

    # Setting "optimized_parameters" in ANN
    meta.update_model_with_parameters(optimized_parameters)

    #Da fare introdurre il tempo!