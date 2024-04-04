import numpy as np
import math
import pygame as pg
import numba as nb
from numba import cuda


@cuda.jit
def cudasurface(surface, ret, color, prop):
    xpos, ypos, zpos = cuda.grid(3)

    if xpos < surface.shape[0] and ypos < surface.shape[1] and zpos < surface.shape[2]:
        ret[xpos, ypos, zpos] = 0
        ret[xpos, ypos, zpos] += surface[max(xpos - 1, 0), max(ypos - 1, 0), zpos] / 9
        ret[xpos, ypos, zpos] += surface[max(xpos - 1, 0), ypos, zpos] / 9
        ret[xpos, ypos, zpos] += surface[max(xpos - 1, 0), min(ypos + 1, surface.shape[1]), zpos] / 9

        ret[xpos, ypos, zpos] += surface[xpos, max(ypos - 1, 0), zpos] / 9
        ret[xpos, ypos, zpos] += surface[xpos, ypos, zpos] / 9
        ret[xpos, ypos, zpos] += surface[xpos, min(ypos + 1, surface.shape[1]), zpos] / 9

        ret[xpos, ypos, zpos] += surface[min(xpos + 1, surface.shape[0]), max(ypos - 1, 0), zpos] / 9
        ret[xpos, ypos, zpos] += surface[min(xpos + 1, surface.shape[0]), ypos, zpos] / 9
        ret[xpos, ypos, zpos] += surface[min(xpos + 1, surface.shape[0]), min(ypos + 1, surface.shape[1]), zpos] / 9

        ret[xpos, ypos, zpos] = max(ret[xpos, ypos, zpos] - prop[0] * color[zpos], 0)


@cuda.jit
def cudamemflip(surface, ret):
    xpos, ypos, zpos = cuda.grid(3)

    if xpos < surface.shape[0] and ypos < surface.shape[1] and zpos < surface.shape[2]:
        ret[xpos, ypos, zpos] = surface[xpos, ypos, zpos]


@cuda.jit
def cudamoveagents(surface, apos, color, seed, prop):
    a, xpos, ypos = cuda.grid(3)

    if a < apos.shape[0]:
        #for _ in range(0, apos[a, 3]):
            surface[min(max(int(apos[a, 0] - prop[1] / 2) + xpos, 0), surface.shape[0] - 1),
            min(max(int(apos[a, 1] - prop[1] / 2) + ypos, 0), surface.shape[1] - 1), 0] = color[0]
            surface[min(max(int(apos[a, 0] - prop[1] / 2) + xpos, 0), surface.shape[0] - 1),
            min(max(int(apos[a, 1] - prop[1] / 2) + ypos, 0), surface.shape[1] - 1), 1] = color[1]
            surface[min(max(int(apos[a, 0] - prop[1] / 2) + xpos, 0), surface.shape[0] - 1),
            min(max(int(apos[a, 1] - prop[1] / 2) + ypos, 0), surface.shape[1] - 1), 2] = color[2]
            apos[a, 0] += math.sin(apos[a, 2]) * apos[a, 3]
            apos[a, 1] += math.cos(apos[a, 2]) * apos[a, 3]

            if apos[a, 0] < 0:
                #apos[a, 0] = surface.shape[0] / 2
                #apos[a, 1] = surface.shape[1] / 2
                #apos[a, 0] = max(apos[a, 0], 0)
                #apos[a, 2] *= -1
                apos[a, 0] += simSize[0]
            elif apos[a, 0] > surface.shape[0]:
                #apos[a, 0] = surface.shape[0] / 2
                #apos[a, 1] = surface.shape[1] / 2
                #apos[a, 0] = min(apos[a, 0], surface.shape[0])
                #apos[a, 2] *= -1
                apos[a, 0] -= simSize[0]
            elif apos[a, 1] < 0:
                #apos[a, 0] = surface.shape[0] / 2
                #apos[a, 1] = surface.shape[1] / 2
                #apos[a, 1] = max(apos[a, 1], 0)
                #apos[a, 2] *= -1
                apos[a, 1] += simSize[1]
            elif apos[a, 1] > surface.shape[1]:
                #apos[a, 0] = surface.shape[0] / 2
                #apos[a, 1] = surface.shape[1] / 2
                #apos[a, 1] = min(apos[a, 1], surface.shape[1])
                #apos[a, 2] *= -1
                apos[a, 1] -= simSize[1]


@cuda.jit
def cudaupdateagents(surface, apos, seed, color, prop):
    a, _, _ = cuda.grid(3)

    if a < apos.shape[0]:

        left = surface[min(max(int(apos[a, 0] + math.sin(apos[a, 2] - prop[3]) * prop[4]), 0), surface.shape[0] - 1),
                       min(max(int(apos[a, 1] + math.cos(apos[a, 2] - prop[3]) * prop[4]), 0),
                           surface.shape[1] - 1), 0] / color[0] / 3
        left += surface[min(max(int(apos[a, 0] + math.sin(apos[a, 2] - prop[3]) * prop[4]), 0), surface.shape[0] - 1),
                        min(max(int(apos[a, 1] + math.cos(apos[a, 2] - prop[3]) * prop[4]), 0),
                            surface.shape[1] - 1), 1] / color[1] / 3
        left += surface[min(max(int(apos[a, 0] + math.sin(apos[a, 2] - prop[3]) * prop[4]), 0), surface.shape[0] - 1),
                        min(max(int(apos[a, 1] + math.cos(apos[a, 2] - prop[3]) * prop[4]), 0),
                            surface.shape[1] - 1), 2] / color[2] / 3
        center = surface[min(max(int(apos[a, 0] + math.sin(apos[a, 2]) * prop[4]), 0), surface.shape[0] - 1),
                         min(max(int(apos[a, 1] + math.cos(apos[a, 2]) * prop[4]), 0), surface.shape[1] - 1), 0] / color[0] / 3
        center += surface[min(max(int(apos[a, 0] + math.sin(apos[a, 2]) * prop[4]), 0), surface.shape[0] - 1),
                          min(max(int(apos[a, 1] + math.cos(apos[a, 2]) * prop[4]), 0), surface.shape[1] - 1), 1] / color[1] / 3
        center += surface[min(max(int(apos[a, 0] + math.sin(apos[a, 2]) * prop[4]), 0), surface.shape[0] - 1),
                          min(max(int(apos[a, 1] + math.cos(apos[a, 2]) * prop[4]), 0), surface.shape[1] - 1), 2] / color[2] / 3
        right = surface[min(max(int(apos[a, 0] + math.sin(apos[a, 2] + prop[3]) * prop[4]), 0), surface.shape[0] - 1),
                        min(max(int(apos[a, 1] + math.cos(apos[a, 2] + prop[3]) * prop[4]), 0),
                            surface.shape[1] - 1), 0] / color[0] / 3
        right += surface[min(max(int(apos[a, 0] + math.sin(apos[a, 2] + prop[3]) * prop[4]), 0), surface.shape[0] - 1),
                         min(max(int(apos[a, 1] + math.cos(apos[a, 2] + prop[3]) * prop[4]), 0),
                             surface.shape[1] - 1), 1] / color[1] / 3
        right += surface[min(max(int(apos[a, 0] + math.sin(apos[a, 2] + prop[3]) * prop[4]), 0), surface.shape[0] - 1),
                         min(max(int(apos[a, 1] + math.cos(apos[a, 2] + prop[3]) * prop[4]), 0),
                             surface.shape[1] - 1), 2] / color[2] / 3

        if center >= left and center >= right:
            apos[a, 2] = apos[a, 2] + (seed[a] - 0.5) * prop[6]
        elif left > right:
            apos[a, 2] = apos[a, 2] - prop[2]
        elif right > left:
            apos[a, 2] = apos[a, 2] + prop[2]
        #else:
        #    apos[a, 2] = apos[a, 2] + (seed[a] - 0.5) * prop[6]


# -- sim setup --

simSize = (1920, 1080)
agentCount = 3
decayRate = 0.000002
turnRate = 45 / 180 * np.pi
eyeAngle = 45 / 180 * np.pi
eyeDistance = 1
moveSpeed = 1
moveSpread = 0
agentSize = 1
randomTurnRate = 0# + turnRate / 4

cudalayerOne = cuda.to_device(np.zeros((simSize[0], simSize[1], 3)))
cudalayerTwo = cuda.to_device(np.zeros((simSize[0], simSize[1], 3)))
cudaProperties = cuda.to_device(np.asarray([decayRate, agentSize, turnRate, eyeAngle, eyeDistance, moveSpeed,
                                            randomTurnRate]))

imagethreadsperblock = (16, 16, 3)
imageblockspergrid_x = int(np.ceil(simSize[0] / imagethreadsperblock[0]))
imageblockspergrid_y = int(np.ceil(simSize[1] / imagethreadsperblock[1]))
imageblockspergrid_z = int(np.ceil(3 / imagethreadsperblock[2]))
imageblockspergrid = (imageblockspergrid_x, imageblockspergrid_y, imageblockspergrid_z)

agentonethreadsperblock = (16, agentSize, agentSize)
agentoneblockspergrid_x = int(np.ceil(agentCount / agentonethreadsperblock[0]))
agentoneblockspergrid_y = int(1)
agentoneblockspergrid_z = int(1)
agentoneblockspergrid = (agentoneblockspergrid_x, agentoneblockspergrid_y, agentoneblockspergrid_z)

agenttwothreadsperblock = (16, 1, 1)
agenttwoblockspergrid_x = int(np.ceil(agentCount / agenttwothreadsperblock[0]))
agenttwoblockspergrid_y = int(1)
agenttwoblockspergrid_z = int(1)
agenttwoblockspergrid = (agenttwoblockspergrid_x, agenttwoblockspergrid_y, agenttwoblockspergrid_z)

# -- agent setup --
"""
agentPosition = np.transpose(np.asarray([np.random.uniform(0, simSize[0], agentCount).astype(np.float32),
                                         np.random.uniform(0, simSize[1], agentCount).astype(np.float32),
                                         np.random.uniform(0, 2 * np.pi, agentCount).astype(np.float32)]))  # """

"""
agentPosition = np.transpose(np.asarray([np.full(agentCount, simSize[0] / 2).astype(np.float32),
                                         np.full(agentCount, simSize[1] / 2).astype(np.float32),
                                         np.random.uniform(0, 2 * np.pi, agentCount).astype(np.float32),
                                         np.random.normal(moveSpeed, moveSpread, agentCount)]))  # """
#"""
agentPosition = np.transpose(np.asarray([np.full(agentCount, simSize[0] / 2).astype(np.float32) +
                                         np.sin(np.linspace(0, 2 * np.pi, num=agentCount)) * simSize[1] / 8,
                                         np.full(agentCount, simSize[1] / 2).astype(np.float32) +
                                         np.cos(np.linspace(0, 2 * np.pi, num=agentCount)) * simSize[1] / 8,
                                         np.linspace(0, 2 * np.pi, num=agentCount).astype(np.float32) + np.pi,
                                         np.random.normal(moveSpeed, moveSpread, agentCount)]))  # """

cudaAgentPos = cuda.to_device(agentPosition)
# -- Pygame setup --

pg.init()

display_width = simSize[0]
display_height = simSize[1]

gameDisplay = pg.display.set_mode((display_width, display_height))
surf = pg.surfarray.pixels3d(gameDisplay)
pg.display.set_caption('slimesim')

black = np.asarray((1, 1, 1))
red = np.asarray((255, 1, 1))
cudared = cuda.to_device(red)
white = np.asarray((255, 255, 255))

clock = pg.time.Clock()
ended = False

# -- Game loop --

while not ended:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            ended = True
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_RETURN:
                cudaAgentPos = cuda.to_device(agentPosition)

    cudaRandom = cuda.to_device(np.random.uniform(0, 1, agentCount))
    cudamoveagents[agentoneblockspergrid, agentonethreadsperblock](cudalayerOne, cudaAgentPos, cudared, cudaRandom,
                                                                   cudaProperties)
    cudasurface[imageblockspergrid, imagethreadsperblock](cudalayerOne, cudalayerTwo, cudared, cudaProperties)
    cudamemflip[imageblockspergrid, imagethreadsperblock](cudalayerTwo, cudalayerOne)
    cudaupdateagents[agenttwoblockspergrid, agenttwothreadsperblock](cudalayerOne, cudaAgentPos, cudaRandom, cudared,
                                                                     cudaProperties)
    surf[:, :, :] = cudalayerOne.copy_to_host()

    pg.display.update()
    clock.tick(600)
pg.quit()
# """
