import numpy as np


def square(x, y):
    np_x = np.array(x)
    np_y = np.array(y)
    return np.sum((np_x - np_y) ** 2, axis=1)


def logarithmic(x, y):
    np_x = np.array(x)
    np_y = np.array(y)
    return np.sum((1 - np_y) * np.log((1 - np_y) / (1 - np_x)) + np_y * np.log(np_y / np_x), axis=1)


def hellinger(x, y):
    np_x = np.array(x)
    np_y = np.array(y)
    return np.sum(((np.sqrt(1 - np_y) - np.sqrt(1 - np_x)) ** 2 + (np.sqrt(np_y) - np.sqrt(np_x)) ** 2) / 2, axis=1)


def absolute(x, y):
    np_x = np.array(x)
    np_y = np.array(y)
    return np.sum(np.abs(np_y - np_x), axis=1)


def dot(x, y):
    np_x = np.array(x)
    np_y = np.array(y)
    return np.dot(np_x, np_y.transpose())


def mix(x, y):
    np_x = np.array(x)
    np_y = np.array(y)
    return -np.log(np.sum(np_x * np.exp(-np_y), axis=1))


def regret(player_loss, experts_loss):
    np_player_loss = np.array(player_loss)
    np_experts_loss = np.array(experts_loss)
    best_expert = np.argmin(np.sum(np_experts_loss), axis=0)
    regrets = [np.sum(np_player_loss[:i + 1])
               - np.sum(np_experts_loss[:i + 1, best_expert])
               for i in range(len(np_player_loss))]
    return regrets
