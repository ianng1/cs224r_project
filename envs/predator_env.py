import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import PIL.Image as Image
import gym
import random
import time

from gym import Env, spaces
import time

HEIGHT = 20
WIDTH = 20


class Entity(object):
    def __init__(self, name, speed, color):
        self.x = np.random.randint(0, HEIGHT)
        self.y = np.random.randint(0, WIDTH)
        self.name = name
        self.speed = speed
        self.color = color

    def move(self, direction):
        x, y = self.x, self.y
        if (direction == 0):
            x -= 1
        if (direction == 1):
            x -= 1
            y += 1
        if (direction == 2):
            y += 1
        if (direction == 3):
            x += 1
            y += 1
        if (direction == 4):
            x += 1
        if (direction == 5):
            x += 1
            y -= 1
        if (direction == 6):
            y -= 1
        if (direction == 7):
            x -= 1
            y -= 1
        x = max(min(x, HEIGHT), 0)
        y = max(min(y, WIDTH), 0)
        return (x, y)


class PredatorPreyEnv(Env):
    def __init__(self):
        super(PredatorPreyEnv, self).__init__()
        self.observation_shape = (HEIGHT, WIDTH, 3)
        self.observation_space = spaces.Box(low = np.zeros(self.observation_shape), 
                                    high = np.ones(self.observation_shape),
                                    dtype = np.float16) 
        self.action_space = spaces.Discrete(8,)

        self.canvas = np.ones(self.observation_shape) * 1
        self.elements = []

        self.x_min = 0
        self.y_min = 0
        
        self.x_max = self.observation_shape[0]
        self.y_max = self.observation_shape[1]

    def draw_elements_on_canvas(self):
        # Init the canvas 
        self.canvas = np.ones(self.observation_shape) * 1

        # Draw the heliopter on canvas
        for elem in self.elements:
            x, y = elem.x, elem.y
            self.canvas[x, y] = elem.color

    def reset(self):
        # Reset the reward
        self.ep_return  = 0
        self.num_steps = 0
        # Intialise the chopper
        self.predator = Entity("predator", 1, np.array([255, 0, 0]))
        self.prey = Entity("prey", 2, np.array([0, 0, 255]))

        # Intialise the elements 
        self.elements = [self.predator, self.prey]

        # Reset the Canvas 
        self.canvas = np.ones(self.observation_shape) * 1

        # Draw elements on the canvas
        self.draw_elements_on_canvas()


        # return the observation
        return self.canvas 
    
    def render(self, mode = "human"):
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        if mode == "human":
            a = plt.imshow(self.canvas)
            plt.show()
            time.sleep(0.1)
            plt.close()
        elif mode == "rgb_array":
            return self.canvas

    def step(self, action):
        # Flag that marks the termination of an episode
        done = (self.num_steps >= 300)
        
        # Assert that it is a valid action 
        assert self.action_space.contains(action), "Invalid Action"
  
        # apply the action to the chopper
        self.predator.x, self.predator.y = self.predator.move(action)

        if (self.predator.x == self.prey.x and self.predator.y == self.prey.y):
            reward = 100
            done = True
        else:
            reward = -np.linalg.norm(np.array([self.predator.x - self.prey.x, self.predator.y - self.prey.y]))

        for movement in range(self.prey.speed):
            new_locations = [np.array(self.prey.move(dir)) for dir in range(8)]

            distances = [np.linalg.norm(x - np.array([self.predator.x, self.predator.y])) for x in new_locations]
            for x in range(8):
                distances[x] += min(new_locations[x][0], HEIGHT -  new_locations[x][0]) + min(new_locations[x][1], WIDTH -  new_locations[x][1])
            best = np.argmax(np.array(distances))
            print(distances)
            print(best)
            self.prey.x, self.prey.y = self.prey.move(best)

        self.draw_elements_on_canvas()
        self.num_steps += 1
        return self.canvas, reward, done, []
        
env = PredatorPreyEnv()
obs = env.reset()

while True:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
    if done == True:
        break
