from utils import *
from scipy.linalg import sqrtm

import matplotlib.pyplot as plt

#nn stuff
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch
import pickle

import matrix_factorization as mfac

def bootstrap(data,s):
    vals = np.array(range(len(data['is_correct'])))
    indexes = np.random.choice(vals, size=s)
    return indexes

##########################
# sgd stuff              #
##########################
sgd_model_path = "./models/sgd_boot"
val_data = load_valid_csv("../data")
def evaluate_sgd(factorized_matrix, test_data):
    cur_score = 0
    for n in range(len(test_data['user_id'])):
        q = test_data['question_id'][n]
        u = test_data['user_id'][n]
        if (int(factorized_matrix.item(u, q)>= 0.5) == test_data['is_correct'][n]):
            cur_score += 1
    stats = cur_score/len(test_data['user_id'])
    return stats


def squared_error_loss(data, u, z, bu, bz, mu, lmd):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    dot = np.dot(u,z.T)
    for n in range(len(data['user_id'])):
        q = data['question_id'][n]
        j = data['user_id'][n]
        #print(bu[u],bz[q],mu,factorized_matrix.item(u, q))
        loss += (data['is_correct'][n] - (dot.item(j,q) + bu[j] + bz[q] + mu))**2
        
        loss += lmd*(np.dot(u[j].T,u[j]) + np.dot(z[q].T,z[q]) + bu[j]**2 + bz[q]**2)
    return 1/2*loss

def update_u_z_b(train_data, lr, u, z,bu , bz, mu, lmd, bootstrapped, should_bootstrap):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    if should_bootstrap == 1:
        j = np.random.choice(len(bootstrapped), 1)[0]
        i = bootstrapped[j]
    else:
        i = np.random.choice(len(train_data["is_correct"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]

    cur_u = u[ n]
    cur_z = z[ q]
    dot = np.dot( cur_u.T ,cur_z)
    #u[n] = cur_u - lr * (-(c - np.dot( cur_u.T ,cur_z)) * cur_z )
    #z[q] = cur_z - lr * (-(c - np.dot( cur_u.T ,cur_z)) * cur_u )
    cur_bu = bu[n]
    cur_bz = bz[q]
    u[n] = cur_u - lr * (-(c - mu - cur_bu - cur_bz - dot ) * cur_z + lmd*cur_u)
    z[q] = cur_z - lr * (-(c - mu - cur_bu - cur_bz - dot) * cur_u + lmd*cur_z)
    bu[n] = cur_bu - lr * (-(c - mu - cur_bu - cur_bz - dot)  + lmd*cur_bu)
    bz[q] = cur_bz - lr * (-(c - mu - cur_bu - cur_bz - dot)  + lmd*cur_bz)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z, bu,bz
def sgd_save(matrix, path):
    path_1 = path + "_mat"
    np.save(path_1, matrix)   

def sgd_load(path):
    path_1 = path + "_mat"+ '.npy'

    mat = np.load(path_1 )
    return mat



def als(train_data, k, lr, num_iteration, lmd, bootstrapped, should_bootstrap):
    """ Performs ALS algorithm. Return reconstructed matrix.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix.
    """
    
    # calculating
    num_u = len(set(train_data["user_id"]))
    num_q = len(set(train_data["question_id"]))
    mu = 0
    tot_obs_data = 0
    total = num_u*num_q
    amt_of_data = len(train_data["user_id"])

    bu = np.zeros((num_u,1))
    bz = np.zeros((num_q,1))

    print("sparsity", amt_of_data/total)
    
    # amt_user = [0] * num_u
    # amt_ques = [0] * num_q
    print(num_u, num_q)
    for i in range(len(train_data["user_id"])):
        bu[train_data["user_id"][i]] +=1
        bz[train_data["question_id"][i]] +=1
        tot_obs_data += 1
        mu += train_data["is_correct"][i]
    bu = bu/tot_obs_data
    bz = bz/tot_obs_data

    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(num_u, k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(num_q, k))
    mu = mu/tot_obs_data

    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # itterations = [None] * int(num_iteration//250)
    # results = [None] * int(num_iteration//250)
    plot_y = []
    plot_x = []
    should_bootstrap = 0
    best_loss = 800
    best_matrix = None
    for i in range(num_iteration):
        if i >= 500000 and i % 1000 == 0:
          cur_mat = np.add(np.add(np.dot(u, z.T),bu), bz.T)+ mu
          loss = squared_error_loss(val_data,u,z,bu,bz,mu,lmd)[0]
          if i % 100000 == 0:
            print( loss, i, i/num_iteration)
          if(best_loss > loss):
            best_loss = loss
            best_matrix = cur_mat
            sgd_save(cur_mat, "./models/sgd_boot_0_k"+str(k))
          plot_x.append(i)
          plot_y.append(loss)
        u,z,bu,bz = update_u_z_b(train_data, lr, u,z,bu,bz,mu, lmd, bootstrapped, should_bootstrap)
    print(squared_error_loss(val_data,u,z,bu,bz,mu,lmd))
    # plt.plot(plot_x,plot_y)
    # plt.savefig("../figs/sgd_k"+str(k))
    mat = np.add(np.add(np.dot(u, z.T),bu), bz.T)+ mu
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    #return mat, best_matrix
    return mat

def conf_weight(x):
    if x < 0.2:
        return -1/(1+((x-0.7+1)/(1-x-1))**(-3))
    elif x > 0.8:
        return 1/(1+((x-0.7)/(1-x))**(-3))        
    return 0

def final_guess_func(sgd_guess):
    ## make functions go from -.5 to .5
    
    sgd_new = (sgd_guess - .5)
    return sgd_new + .5 

def new_eval(sgd_matrix, test_data):
    

    total = 0
    correct = 0
    sm = [None]*12
    for i in range(12):
      sm[i] = sgd_load(sgd_model_path+str(i))

    for i in range(len(test_data["user_id"])):
        
        x = test_data['user_id'][i]
        y = test_data['question_id'][i]
        sgd_guess = 0
        
        for j in range(12):
          sgd_guess += sm[j].item(x,y)
        
        sgd_guess = sgd_guess/12
        
        total+=1
        
        final_guess = sgd_guess
        """
        if (sgd_guess >= .5) == test_data["is_correct"][i] and (final_guess >= .5) != test_data["is_correct"][i]:
            print("uo")
            print(sgd_guess, conf_weight(nn_guess)/10, final_guess) 
        if (sgd_guess >= .5) != test_data["is_correct"][i] and (final_guess >= .5) == test_data["is_correct"][i]:
            print("ao")
            print(sgd_guess, conf_weight(nn_guess)/10, final_guess) 
        """

        if (final_guess >= .5) == test_data["is_correct"][i]:
            correct += 1
    return correct/float(total)

def eval_private(sgd_matrix, private_data):
    

    
    new_data = private_data.copy()
    #print(new_data["question_id"])
    #print(new_data["is_correct"])
    total = 0
    sgd_correct = 0
    correct = 0
    sm = [None]*12
    for i in range(12):
      sm[i] = sgd_load(sgd_model_path+str(i))
    
    new_data["is_correct"] = []
    for i in range(len(private_data["user_id"])):
        
        x = private_data['user_id'][i]
        y = private_data['question_id'][i]
        sgd_guess=0
        for j in range(12):
          sgd_guess += sm[j].item(x,y)
        
        sgd_guess = sgd_guess/12
        
        total+=1
        final_guess = sgd_guess
        #print(sgd_guess)
        new_data['is_correct'].append((final_guess >= .5))
        if((final_guess >= .5)!= True and (final_guess >= .5)!= False):
          print(final_guess)
    #print(len(new_data["is_correct"]))
    save_private_test_csv(new_data)



def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    test_data = load_public_test_csv("../data")
    private_data = load_private_test_csv("../data")
    #####################################################################
    # TODO:                                                             #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################


    run_sgd = 1
    should_bootstrap = 0
    
    shoud_sgd_load = 1
    generate_priv = 1  
    
    sgd_matrix = None
    tot_obs_data = 0
    mu = 0
    for i in range(len(train_data["user_id"])):
        tot_obs_data += 1
        mu += train_data["is_correct"][i]

    mu = mu/tot_obs_data


    irt_theta = None
    irt_beta = None
    if should_bootstrap == 1:
      for i in range(12):
        k_value = 60
        bootstrap_index = bootstrap(train_data, 2*int(len(train_data["is_correct"])))
        
        sgd_matrix = als(train_data,k_value,0.01, 3000000, 0.065, bootstrap_index, 0)
        sgd_save(sgd_matrix, sgd_model_path+str(i))

    if shoud_sgd_load == 1:
        sgd_matrix = sgd_load(sgd_model_path+"1")
    elif run_sgd == 1:
        k_value = 60
       
        bootstrap_index = bootstrap(train_data, 2*int(len(train_data["is_correct"])))
        
        sgd_matrix = als(train_data,k_value,0.01, 3000000, 0.065, bootstrap_index, 0)
        sgd_save(sgd_matrix, sgd_model_path)
        
        cur_score = 0
        print("done training sgd")
    #print(evaluate_sgd(sgd_matrix, val_data))
               
    
    print("val:",new_eval(sgd_matrix, val_data))
    print("test:",new_eval(sgd_matrix, test_data))
    if generate_priv == 1: 
        eval_private(sgd_matrix, private_data)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
    
