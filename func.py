import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statistics as stats
import math


def guess_centroid(list1, list2):
    plt.scatter(list1, list2)
    plt.show()
    guess = input('Input Centroid Locations: ')
    guess = eval(guess)  # if written as coordinate points gives tuple or tuple of tuples
    # print(type(guess[0][1]))
    return guess


def distance(l1, l2, centroids=None):
    if centroids is None:
        centroids = guess_centroid(l1, l2)
    # creates dictionary
    new_dict = {}
    for val in centroids:
        new_dict[val] = []
    # print(new_dict)
    # assign to centroids
    for i in range(len(l1)):
        dist_l = []
        for c in centroids:
            dist_l.append(((c[0]-l1[i])**2) + ((c[1]-l2[i])**2))
        new_dict[centroids[dist_l.index(min(dist_l))]].append(i)
    # plots the new centroid and reassigned points
    for v in centroids:
        x = [l1[k] for k in new_dict[v]]
        y = [l2[k] for k in new_dict[v]]
        plt.scatter(x, y)
        plt.plot([v[0]], [v[1]], 'k', marker='D')
    plt.show()
    return new_dict


def new_centroid(l1, l2, dict):
    c = []
    for v in dict:
        cx = stats.mean([l1[k] for k in dict[v]])
        cy = stats.mean([l2[k] for k in dict[v]])
        c.append((cx, cy))
    return c


def main(l1, l2, centroids=None, rep=4):
    for n in range(rep):
        d1 = distance(l1, l2, centroids)
        centroids = new_centroid(l1, l2, d1)
    return d1, centroids


def mean(data):
    total = sum(data)
    m = total / len(data)
    return m


def median(data):
    data.sort()
    if len(data) % 2 == 0:
        m = (data[len(data) // 2] + data[len(data) // 2 - 1]) / 2
    else:
        m = data[len(data) // 2]
    return m


def variance(data):
    new_list = [(val - mean(data)) ** 2 for val in data]
    v = mean(new_list)
    return v


def stand_dev(data):
    v = variance(data)
    s = math.sqrt(v)
    return s


def elem_stats(data):
    new_dict = {
        'mean': mean(data),
        'median': median(data),
        'variance': variance(data),
        'std': stand_dev(data),
        'min': min(data),
        'max': max(data)
    }
    return new_dict


def corcoeff(xd, yd):
    sigma1 = sigma_xy(xd, yd) * len(xd)
    sigma2 = sum(xd) * sum(yd)
    sigma3 = len(xd) * sum([val ** 2 for val in xd])
    sigma4 = sum(xd) ** 2
    sigma5 = len(yd) * sum([val ** 2 for val in yd])
    sigma6 = sum(yd) ** 2
    top = (sigma1 - sigma2)
    bottom = (math.sqrt(sigma3 - sigma4)) * (math.sqrt(sigma5 - sigma6))
    return top / bottom


def sigma_xy(xd, yd):
    nlist = []
    for i in range(len(xd)):
        nlist.append((xd[i] * yd[i]))
    return sum(nlist)


def least_sqrs(xd, yd):
    matrix1 = [[sum(val ** 2 for val in xd), sum(xd)], [sum(xd), len(xd)]]
    matrix2 = [sigma_xy(xd, yd), sum(yd)]
    array1 = np.array(matrix1)
    array2 = np.array(matrix2)
    invarray1 = np.linalg.inv(array1)
    solution = np.dot(invarray1, array2)
    return solution




