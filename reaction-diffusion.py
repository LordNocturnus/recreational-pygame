import numpy as np
import math
import pygame as pg
import numba as nb
from numba import cuda


@cuda.jit
def mem_flip(i, o):
    x, y, _ = cuda.grid(3)
    o[x, y] = i[x, y]


@cuda.jit
def mem_reset(i):
    x, y, _ = cuda.grid(3)
    i[x, y] = 0


"""@cuda.jit
def weight_shift(i, o, w, dx, dy, tx, ty):
    x, y, _ = cuda.grid(3)
    tx[x, y] = x + dx[x, y]
    ty[x, y] = y + dy[x, y]
    if not 0 > tx[x, y] >= i.shape[0] and not 0 > ty[x, y] >= i.shape[1]:
        o[x, y] += i[int(tx[x, y]), int(ty[x, y])] * w[x, y]#"""


@cuda.jit
def update_a(a1, a2, a3, da, b1, f, dt):
    x, y, _ = cuda.grid(3)
    a3[x, y] = 0.05 * (a1[max(x - 1, 0), max(y - 1, 0)] +
                       a1[min(x + 1, simSize[1]), max(y - 1, 0)] +
                       a1[max(x - 1, 0), min(y + 1, simSize[0])] +
                       a1[min(x + 1, simSize[1]), min(y + 1, simSize[0])]) + \
               0.2 * (a1[max(x - 1, 0), y] +
                      a1[x, max(y - 1, 0)] +
                      a1[min(x + 1, simSize[1]), y] +
                      a1[x, min(y + 1, simSize[0])]) - a1[x, y]
    a2[x, y] = a1[x, y] + (da[x, y] * a3[x, y] * a1[x, y] - a1[x, y] * b1[x, y] * b1[x, y] + f[x, y] * (1 - a1[x, y])) * dt[x, y]


@cuda.jit
def update_b(b1, b2, b3, db, a1, f, k, dt):
    x, y, _ = cuda.grid(3)
    b3[x, y] = 0.05 * (b1[max(x - 1, 0), max(y - 1, 0)] +
                       b1[min(x + 1, simSize[1]), max(y - 1, 0)] +
                       b1[max(x - 1, 0), min(y + 1, simSize[0])] +
                       b1[min(x + 1, simSize[1]), min(y + 1, simSize[0])]) + \
               0.2 * (b1[max(x - 1, 0), y] +
                      b1[x, max(y - 1, 0)] +
                      b1[min(x + 1, simSize[1]), y] +
                      b1[x, min(y + 1, simSize[0])]) - b1[x, y]
    b2[x, y] = b1[x, y] + (db[x, y] * b3[x, y] * b1[x, y] + a1[x, y] * b1[x, y] * b1[x, y] - (k[x, y] + f[x, y]) * b1[x, y]) * dt[x, y]


@cuda.jit
def color_update(c, a1, b1, a3):
    x, y, _ = cuda.grid(3)
    c[x, y, 0] = b1[x, y]
    c[x, y, 1] = b1[x, y]
    c[x, y, 2] = b1[x, y]


# -- sim setup --

simSize = (1920, 1080)
diffusion_a = 1
diffusion_b = 0.125
feed_rate = 0.055
kill_rate = 0.062
timestep = 1
w = 25
#corner_weight = 0.05
#edge_weight = 0.2
#middle_weight = -1

b = np.zeros(simSize, dtype=float)
b[int(simSize[0] / 2)-w:int(simSize[0] / 2)+w, int(simSize[1] / 2)-w:int(simSize[1] / 2)+w] = 1

cuda_color = cuda.to_device(np.ones((simSize[0], simSize[1], 3), dtype=float))

cuda_a1 = cuda.to_device(np.ones(simSize, dtype=float))
cuda_a2 = cuda.to_device(np.ones(simSize, dtype=float))
cuda_a3 = cuda.to_device(np.ones(simSize, dtype=float))

cuda_b1 = cuda.to_device(b)
cuda_b2 = cuda.to_device(np.zeros(simSize, dtype=float))
cuda_b3 = cuda.to_device(np.zeros(simSize, dtype=float))

"""cuda_wc = cuda.to_device(np.full(simSize, corner_weight, dtype=float))
cuda_we = cuda.to_device(np.full(simSize, edge_weight, dtype=float))
cuda_wm = cuda.to_device(np.full(simSize, middle_weight, dtype=float))#"""

cuda_da = cuda.to_device(np.full(simSize, diffusion_a, dtype=float))
cuda_db = cuda.to_device(np.full(simSize, diffusion_b, dtype=float))
cuda_f = cuda.to_device(np.full(simSize, feed_rate, dtype=float))
cuda_k = cuda.to_device(np.full(simSize, kill_rate, dtype=float))
cuda_dt = cuda.to_device(np.full(simSize, timestep, dtype=float))

"""cuda_dx1 = cuda.to_device(np.ones(simSize, dtype=float) * -1)
cuda_dx2 = cuda.to_device(np.zeros(simSize, dtype=float))
cuda_dx3 = cuda.to_device(np.ones(simSize, dtype=float))

cuda_dy1 = cuda.to_device(np.ones(simSize, dtype=float) * -1)
cuda_dy2 = cuda.to_device(np.zeros(simSize, dtype=float))
cuda_dy3 = cuda.to_device(np.ones(simSize, dtype=float))

cuda_tx = cuda.to_device(np.zeros(simSize, dtype=float))
cuda_ty = cuda.to_device(np.zeros(simSize, dtype=float))#"""

threadsperblock = (16, 16, 1)
blockspergrid_x = int(np.ceil(simSize[0] / threadsperblock[0]))
blockspergrid_y = int(np.ceil(simSize[1] / threadsperblock[1]))
blockspergrid_z = int(np.ceil(3 / threadsperblock[2]))
blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

# -- Pygame setup --

pg.init()

display_width = simSize[0]
display_height = simSize[1]

gameDisplay = pg.display.set_mode((display_width, display_height))
surf = pg.surfarray.pixels3d(gameDisplay)
pg.display.set_caption('slimesim')

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

    #mem_reset[blockspergrid, threadsperblock](cuda_a3)
    #mem_reset[blockspergrid, threadsperblock](cuda_b3)

    """weight_shift[blockspergrid, threadsperblock](cuda_a1, cuda_a3, cuda_wc, cuda_dx1, cuda_dy1, cuda_tx, cuda_ty)
    weight_shift[blockspergrid, threadsperblock](cuda_a1, cuda_a3, cuda_we, cuda_dx2, cuda_dy1, cuda_tx, cuda_ty)
    weight_shift[blockspergrid, threadsperblock](cuda_a1, cuda_a3, cuda_wc, cuda_dx3, cuda_dy1, cuda_tx, cuda_ty)

    weight_shift[blockspergrid, threadsperblock](cuda_a1, cuda_a3, cuda_we, cuda_dx1, cuda_dy2, cuda_tx, cuda_ty)
    weight_shift[blockspergrid, threadsperblock](cuda_a1, cuda_a3, cuda_wm, cuda_dx2, cuda_dy2, cuda_tx, cuda_ty)
    weight_shift[blockspergrid, threadsperblock](cuda_a1, cuda_a3, cuda_we, cuda_dx3, cuda_dy2, cuda_tx, cuda_ty)

    weight_shift[blockspergrid, threadsperblock](cuda_a1, cuda_a3, cuda_wc, cuda_dx1, cuda_dy3, cuda_tx, cuda_ty)
    weight_shift[blockspergrid, threadsperblock](cuda_a1, cuda_a3, cuda_we, cuda_dx2, cuda_dy3, cuda_tx, cuda_ty)
    weight_shift[blockspergrid, threadsperblock](cuda_a1, cuda_a3, cuda_wc, cuda_dx3, cuda_dy3, cuda_tx, cuda_ty)


    weight_shift[blockspergrid, threadsperblock](cuda_b1, cuda_b3, cuda_wc, cuda_dx1, cuda_dy1, cuda_tx, cuda_ty)
    weight_shift[blockspergrid, threadsperblock](cuda_b1, cuda_b3, cuda_we, cuda_dx2, cuda_dy1, cuda_tx, cuda_ty)
    weight_shift[blockspergrid, threadsperblock](cuda_b1, cuda_b3, cuda_wc, cuda_dx3, cuda_dy1, cuda_tx, cuda_ty)

    weight_shift[blockspergrid, threadsperblock](cuda_b1, cuda_b3, cuda_we, cuda_dx1, cuda_dy2, cuda_tx, cuda_ty)
    weight_shift[blockspergrid, threadsperblock](cuda_b1, cuda_b3, cuda_wm, cuda_dx2, cuda_dy2, cuda_tx, cuda_ty)
    weight_shift[blockspergrid, threadsperblock](cuda_b1, cuda_b3, cuda_we, cuda_dx3, cuda_dy2, cuda_tx, cuda_ty)

    weight_shift[blockspergrid, threadsperblock](cuda_b1, cuda_b3, cuda_wc, cuda_dx1, cuda_dy3, cuda_tx, cuda_ty)
    weight_shift[blockspergrid, threadsperblock](cuda_b1, cuda_b3, cuda_we, cuda_dx2, cuda_dy3, cuda_tx, cuda_ty)
    weight_shift[blockspergrid, threadsperblock](cuda_b1, cuda_b3, cuda_wc, cuda_dx3, cuda_dy3, cuda_tx, cuda_ty)#"""

    update_a[blockspergrid, threadsperblock](cuda_a1, cuda_a2, cuda_a3, cuda_da, cuda_b1, cuda_f, cuda_dt)
    update_b[blockspergrid, threadsperblock](cuda_b1, cuda_b2, cuda_b3, cuda_db, cuda_a1, cuda_f, cuda_k, cuda_dt)

    mem_flip[blockspergrid, threadsperblock](cuda_a2, cuda_a1)
    mem_flip[blockspergrid, threadsperblock](cuda_b2, cuda_b1)

    color_update[blockspergrid, threadsperblock](cuda_color, cuda_a1, cuda_b1, cuda_a3)
    surf[:, :, :] = cuda_color.copy_to_host() * 255

    pg.display.update()
    clock.tick(600)
pg.quit()
# """
