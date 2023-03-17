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


# REGRESSION EQUATION FOR LINEAR AND BILINEAR
def regression_eqn(ind_array, dep_array, linear=True):
    # input as two arrays or 2 columns of a DF
    x_4 = (ind_array**4).sum()
    x_3 = (ind_array**3).sum()
    x_2 = (ind_array**2).sum()
    x_1 = (ind_array).sum()
    n = len(ind_array)
    xy_2 = ((ind_array**2 * dep_array)).sum()
    xy = (ind_array * dep_array).sum()
    if linear:
        matrix1 = [[x_2, ind_array.sum()], [ind_array.sum(), n]]
        matrix2 = [[xy], [dep_array.sum()]]
        invarray1 = np.linalg.inv(matrix1)
        solution = np.dot(invarray1, matrix2)
        return solution
    else:
        matrix1 = [[x_4, x_3, x_2], [x_3, x_2, x_1], [x_2, x_1, n]]
        matrix2 = [[xy_2], [xy], [dep_array.sum()]]
        invarray1 = np.linalg.inv(matrix1)
        solution = np.dot(invarray1, matrix2)
        return solution[0][0], solution[1][0], solution[2][0]


def get_random(df, per):
    num_per = int(per/100 * len(df))
    training = df.sample(num_per)
    training = training.sort_index()
    test = df.drop(training.index)
    return training, test


def residuals(xd, yd, n=2):
    xdl = xd.tolist()
    ydl = yd.tolist()
    coeff2, coeff1, y_int = regression_eqn(xd, yd, linear=False)
    ys = [(coeff2 * (val**2)) + (coeff1*val) + y_int for val in xdl]
    r = [yd[n]-ys[n] for n in range(len(ydl))]
    mr = mean(r)
    stdr = stand_dev(r)
    return r, mr, stdr


def scatter_plot_er_2(data1, data2, coeff2, coeff1, y_int, std, title='Graph', xt='X', yt='Y', n=2):
    data1 = data1.tolist()
    data2 = data2.tolist()
    y_vals = []
    e1 = []
    e2 = []
    x_data = [min(data1), max(data1)]
    for val in range(len(data1)):
        ans = (coeff2 * (data1[val]**2)) + (coeff1*data1[val]) + y_int
        y_vals.append(ans)
    for val in range(len(data1)):
        ans = (coeff2 * (data1[val]**2)) + (coeff1*data1[val]) + y_int +(n*std)
        e1.append(ans)
    for val in range(len(data1)):
        ans = (coeff2 * (data1[val]**2)) + (coeff1*data1[val]) + y_int -(n*std)
        e2.append(ans)
    plt.plot(data1, y_vals, '-r')
    plt.plot(data1, e1, '--r')
    plt.plot(data1, e2, '--r')
    plt.scatter(data1, data2)
    plt.title(title)
    plt.xlabel(xt)
    plt.ylabel(yt)
    plt.text(x_data[1], y_vals[1], f'Y={round(coeff2, 5)}*X^2+{round(coeff1, 2)}X+{round(y_int, 2)}', color='g')
    plt.show()


def scatter_plot(data1, data2, slope, y_int, xt='X', yt='Y', title='Graph'):
    y_vals = []
    x_data = [min(data1), max(data1)]
    for val in range(2):
        ans = (slope * x_data[val]) + y_int
        y_vals.append(ans)
    plt.plot(x_data, y_vals, '-r')
    plt.scatter(data1, data2)
    plt.title(title)
    plt.xlabel(xt)
    plt.ylabel(yt)
    plt.text(x_data[1], y_vals[1], f'Y={round(slope, 4)}*X+{round(y_int, 4)}', color='g')
    plt.show()


def scatter_plot_bilin(data1, data2, coeff2, coeff1, y_int, title='Graph', xt='X', yt='Y'):
    data1 = data1.tolist()
    data1_s = sorted(data1)
    data2 = data2.tolist()
    y_vals = []
    e1 = []
    e2 = []
    x_data = [min(data1), max(data1)]
    for val in range(len(data1)):
        ans = (coeff2 * (data1_s[val]**2)) + (coeff1*data1_s[val]) + y_int
        y_vals.append(ans)
    plt.plot(data1_s, y_vals, '-r')
    plt.scatter(data1, data2)
    plt.title(title)
    plt.xlabel(xt)
    plt.ylabel(yt)
    plt.text(x_data[1], y_vals[1], f'Y={round(coeff2, 5)}*X^2+{round(coeff1, 2)}X+{round(y_int, 2)}', color='g')
    plt.show()
    
    
