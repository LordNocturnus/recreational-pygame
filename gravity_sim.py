import numpy as np
import math
import pygame as pg
import numba as nb
#from numba import cuda


simSize = (1920, 1080)
dt = 1 / 60
agents = 250
size = 10
dis = 240
attraction = 1
spring = 1
viscosity = 0.0001

angle = np.random.uniform(0, 2 * np.pi, agents)
xpos = dis * np.sin(angle)
ypos = dis * np.cos(angle)
vx = np.zeros_like(xpos)
vy = np.zeros_like(ypos)
fx = np.zeros_like(xpos)
fy = np.zeros_like(ypos)


def gravity(x, y, fx, fy):
    distance = np.clip(np.sqrt(x ** 2 + y ** 2), size / 1000, np.inf)
    fx += attraction / (distance ** 2) * x * -1
    fy += attraction / (distance ** 2) * y * -1
    return fx, fy


def visc(vx, vy, fx, fy):
    fx += viscosity * np.sqrt(vx ** 2 + vy ** 2) * vx * -1
    fy += viscosity * np.sqrt(vx ** 2 + vy ** 2) * vy * -1
    return fx, fy


def pressure(x, y, fx, fy):
    for i in range(agents):
        for j in range(i + 1, agents):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if abs(dx) < size and abs(dy) < size:
                distance = np.sqrt(dx ** 2 + dy ** 2)
                if distance < size:
                    forcex = spring / (distance ** 2) * dx
                    forcey = spring / (distance ** 2) * dy
                    fx[i] += forcex
                    fx[j] -= forcex
                    fy[i] += forcey
                    fy[j] -= forcey
    return fx, fy


def integrate(x, y, vx, vy, fx, fy, dt):
    x += vx * dt + 1 / 2 * fx * dt ** 2
    y += vy * dt + 1 / 2 * fy * dt ** 2

    vx += fx * dt
    vy += fy * dt
    return x, y, vx, vy


if __name__ == "__main__":

    pg.init()

    display_width = simSize[0]
    display_height = simSize[1]

    gameDisplay = pg.display.set_mode((display_width, display_height))
    surf = pg.surfarray.pixels3d(gameDisplay)
    pg.display.set_caption('gravity_sim')

    black = np.asarray((1, 1, 1))
    red = np.asarray((255, 1, 1))
    white = np.asarray((255, 255, 255))

    clock = pg.time.Clock()
    ended = False

    # -- Game loop --

    while not ended:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                ended = True

        fx[:] = 0.0
        fy[:] = 0.0

        fx, fy = gravity(xpos, ypos, fx, fy)
        fx, fy = visc(vx, vy, fx, fy)
        fx, fy = pressure(xpos, ypos, fx, fy)
        xpos, ypos, vx, vy = integrate(xpos, ypos, vx, vy, fx, fy, dt)

        gameDisplay.fill(black)
        xaverage = np.average(xpos)
        yaverage = np.average(ypos)
        limrange = max(np.ptp(xpos) * 1.1 / simSize[0], np.ptp(ypos) * 1.1 / simSize[1])
        print(limrange)
        #pg.draw.circle(gameDisplay, red, (simSize[0] / 2, simSize[1] / 2), dis, 4)
        for i in range(agents):
            pg.draw.circle(gameDisplay, white, ((xpos[i] - xaverage) / limrange + simSize[0] / 2,
                                                (ypos[i] - yaverage) / limrange + simSize[1] / 2), size / limrange)

        pg.display.update()
        clock.tick(600)
    pg.quit()