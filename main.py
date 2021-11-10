import multiprocessing
import random
import time

import numpy as np
from scipy import ndimage
from PIL import Image as im
from functools import partial

height = 1080
width = 1920
number_of_agents = 20000
step_size = 2
simulation_length = 20
simulation_report_step_size = 10
mutation_probability = 0.6
multiprocessing_pool_size = 8


class Agent:
    def __init__(self, x, y):
        self.pos = np.array([x, y])
        self.dir = np.array([0, 0])
        self.X = 0
        self.Y = 1

        self.moves = np.array(
            [[-1, -1],
             [-1, 0],
             [-1, 1],
             [0, -1],
             [0, 1],
             [1, -1],
             [1, 0],
             [1, 1]]
        )

    def legalize_position(self, pos):
        if pos[self.X] < 0:
            pos[self.X] += height

        if pos[self.X] >= height:
            pos[self.X] -= height

        if pos[self.Y] >= width:
            pos[self.Y] -= width

        if pos[self.Y] < 0:
            pos[self.Y] += width

        return pos

    def get_best_path(self, image):
        best = 0
        move_id_best = 0
        changed = False
        for move_id in range(self.moves.shape[0]):
            sense_point = self.legalize_position(np.array(self.pos + self.moves[move_id]))
            if best < image[sense_point[self.X]][sense_point[self.Y]]:
                best = image[sense_point[self.X]][sense_point[self.Y]]
                move_id_best = move_id

        if best > 0:
            changed = True

        return move_id_best, changed

    def mutate_move(self, move_id, p_mutate):
        if p_mutate > random.random():
            move_id = random.randrange(0, 8)

        return move_id

    def move(self, image):
        result = self.get_best_path(image)
        if result[1]:
            self.pos = self.pos + self.moves[self.mutate_move(result[0], mutation_probability)]
        else:
            self.pos = self.pos + self.moves[self.mutate_move(result[0], 1.1)]

        self.pos = self.legalize_position(self.pos)


def draw_agent(agent, image):
    image[agent.pos[agent.X]][agent.pos[agent.Y]] = 255
    return image


def move_agent_helper(image, agent):
    agent.move(image)
    return agent


def main():
    agents = []
    for agent_number in range(number_of_agents):
        agents.append(Agent(random.randrange(0, height - 1), random.randrange(0, width - 1)))
    raw_image_array = np.full((height, width), 0, np.uint8)

    average_time_processing = 0
    average_time_drawing = 0
    for i in range(simulation_length):
        start_time = time.time()
        mp_pool_partial = partial(move_agent_helper, raw_image_array)
        mp_pool = multiprocessing.Pool(multiprocessing_pool_size)
        result = mp_pool.map(mp_pool_partial, agents)
        mp_pool.close()
        mp_pool.join()
        agents = result
        end_time = time.time()
        average_time_processing += (end_time-start_time)

        start_time = time.time()
        for agent in agents:
            raw_image_array = draw_agent(agent, raw_image_array)

        #raw_image_array = ndimage.gaussian_filter(raw_image_array, sigma=0.25) # slower but fancy
        raw_image_array = raw_image_array // 2 # faster but less fancy and more dangerous
        end_time = time.time()
        average_time_drawing += (end_time - start_time)
        
        if (i + 1) % simulation_report_step_size == 0:
            print("Simulation Progress (length):", i)
            print("Simulation Prozessing Average (seconds):", average_time_processing / simulation_report_step_size)
            print("Simulation Drawing Average (seconds):", average_time_drawing / simulation_report_step_size)
            print()
            average_time_processing = 0
            average_time_drawing = 0

    # creating image object of above array
    data = im.fromarray(raw_image_array)

    # saving the final output as a PNG file
    data.save('gfg_dummy_pic.png')


# driver code
if __name__ == "__main__":
    # function call
    main()
