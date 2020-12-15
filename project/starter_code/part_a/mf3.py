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

####################
# neural net stuff #
####################
val_data = load_valid_csv("../data")

def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1|| + ||W^2||.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2)
        h_w_norm = torch.norm(self.h.weight, 2)
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        sig = nn.Sigmoid()
        out = sig(self.h(sig(self.g(inputs))))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function. 
    
    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.)
            loss.backward()
            train_loss += loss.item()

            train_loss += lamb/2*model.get_weight_norm()

            optimizer.step()
        print("epach: ", epoch)
        #valid_acc = evaluate(model, zero_train_data, valid_data)
        #print("Epoch: {} \tTraining Cost: {:.6f}\t "
        #      "Valid Acc: {}".format(epoch, train_loss, valid_acc))
    return model
    #valid_acc = evaluate(model, zero_train_data, valid_data)
    #print("Epoch: {} \tTraining Cost: {:.6f}\t "
    #         "Valid Acc: {}".format(epoch, train_loss, valid_acc))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate_nn(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)

##########################
# sgd stuff              #
##########################

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
    # for i in range(num_u):
    #     amt_user[i] = amt_user[i]/num_u
    # for i in range(num_u):
    #     amt_ques[i] = amt_ques[i]/num_q
    #print(amt_user)

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
            sgd_save(cur_mat, "./models/sgd_k"+str(k))
        #elif i >= 900000 and i%(100) == 0:
        #    loss = squared_error_loss(val_data,u,z,bu,bz,mu,lmd)[0]
        #    evaluation = None
        #    if loss < best_loss:
        #        cur_mat = np.add(np.add(np.dot(u, z.T),bu), bz.T)+ mu
        #        evaluation = evaluate_sgd(cur_mat, val_data)
        #        print("wow!!!!!", evaluation, loss, i)
        #        sgd_save(cur_mat, "./models/sgd_final")
        #        best_loss = loss
          plot_x.append(i)
          plot_y.append(loss)
        u,z,bu,bz = update_u_z_b(train_data, lr, u,z,bu,bz,mu, lmd, bootstrapped, should_bootstrap)
    print(squared_error_loss(val_data,u,z,bu,bz,mu,lmd))
    plt.plot(plot_x,plot_y)
    plt.savefig("../figs/sgd_k"+str(k))
    mat = np.add(np.add(np.dot(u, z.T),bu), bz.T)+ mu
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat, best_matrix

def conf_weight(x):
    if x < 0.2:
        return -1/(1+((x-0.7+1)/(1-x-1))**(-3))
    elif x > 0.8:
        return 1/(1+((x-0.7)/(1-x))**(-3))        
    return 0

def final_guess_func(nn_guess,sgd_guess):
    ## make functions go from -.5 to .5
    nn_new = (nn_guess-.5)/8
    sgd_new = (sgd_guess - .5)
    return nn_new + sgd_new + .5 

def new_eval(nn_model, sgd_matrix, train_data, test_data):
    nn_model.eval()

    total = 0
    nn_correct = 0
    sgd_correct = 0
    correct = 0
 
    s2 = sgd_load("./models/sgd_k40")
    s3 = sgd_load("./models/sgd_k50")
    s4 = sgd_load("./models/sgd_k60")
    s5 = sgd_load("./models/sgd_k70")
    s6 = sgd_load("./models/sgd_k80")
    s7 = sgd_load("./models/sgd_k90")


    for i, u in enumerate(test_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = nn_model(inputs)

        x = test_data['user_id'][i]
        y = test_data['question_id'][i]

        nn_guess = conf_weight(output[0][test_data["question_id"][i]].item()/3 + sgd_matrix.item(x, y)/3 + sgd_matrix.item(x, y)/3)
        sgd_guess = sgd_matrix.item(x, y)
        sgd_guess = (s2.item(x, y) + s3.item(x, y) + s4.item(x, y) + s5.item(x, y) + s6.item(x, y) + s7.item(x, y))/6
        total+=1
        #print(nn_guess, sgd_guess)
        #final_guess = final_guess_func(0*nn_guess,sgd_guess)
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
    print("eval:" + str(correct/float(total)))

def eval_private(nn_model, sgd_matrix,train_data, private_data):
    nn_model.eval()

    total = 0
    nn_correct = 0
    sgd_correct = 0
    correct = 0
    new_data = private_data.copy()
    #print(new_data["question_id"])
    #print(new_data["is_correct"])
    s2 = sgd_load("./models/sgd_k40")
    s3 = sgd_load("./models/sgd_k50")
    s4 = sgd_load("./models/sgd_k60")
    s5 = sgd_load("./models/sgd_k70")
    s6 = sgd_load("./models/sgd_k80")
    s7 = sgd_load("./models/sgd_k90")
    new_data["is_correct"] = [None] * len(private_data["user_id"])
    for i, u in enumerate(private_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = nn_model(inputs)

        nn_guess = output[0][private_data["question_id"][i]].item()
        
        x = private_data['user_id'][i]
        y = private_data['question_id'][i]
        sgd_guess = sgd_matrix.item(x, y)
        sgd_guess = (s2.item(x, y) + s3.item(x, y) + s4.item(x, y) + s5.item(x, y) + s6.item(x, y) + s7.item(x, y))/6
        total+=1
        #print(nn_guess, sgd_guess)
        final_guess = final_guess_func(0*nn_guess,sgd_guess)
        new_data['is_correct'][i] = final_guess >= .5

        
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
    run_nn = 1
    nn_load = 1
    shoud_sgd_load = 1
    generate_priv = 1
    nn_model_path = "./models/nn"
    sgd_model_path = "./models/sgd_k90"
    #np.save(sgd_model_path, sgd_matrix)   
    nn_model = None
    sgd_matrix = None
    tot_obs_data = 0
    mu = 0
    for i in range(len(train_data["user_id"])):
        tot_obs_data += 1
        mu += train_data["is_correct"][i]

    mu = mu/tot_obs_data


    irt_theta = None
    irt_beta = None

    

    if shoud_sgd_load == 1:
        sgd_matrix = sgd_load(sgd_model_path)
    elif run_sgd == 1:
        k_value = 100
        # prev 2000000
        # prev 1250000
        #sgd_matrix = als(bootstrap(train_data, int(len(train_data["is_correct"])*3/4)),k_value,0.01, 1000000, 0.065)
        bootstrap_index = bootstrap(train_data, int(len(train_data["is_correct"])*3/4))
        sgd_matrix,best_matrix = als(train_data,k_value,0.01, 4000000, 0.065, bootstrap_index, 0)
        sgd_save(sgd_matrix, sgd_model_path)
        print("best matrix:", evaluate_sgd(best_matrix, val_data))
        cur_score = 0
        print("done training sgd")
    print(evaluate_sgd(sgd_matrix, val_data))
               
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    if nn_load == 1:
        nn_model = AutoEncoder(train_matrix.shape[1], 10)
        nn_model.load_state_dict(torch.load(nn_model_path))
        nn_model.eval()
    elif run_nn == 1:
        device = torch.device("cuda")
        model = AutoEncoder(train_matrix.shape[1], 10)
        lr = 0.05
        num_epoch = 10
        lamb = 0
        nn_model = train(model, 0.05, 0.1, train_matrix, zero_train_matrix, valid_data, 11)    
        torch.save(nn_model.state_dict(), nn_model_path)
        print("done training nn")
    print(evaluate_nn(nn_model, zero_train_matrix, val_data))
    new_eval(nn_model, sgd_matrix, zero_train_matrix, test_data)
    if generate_priv == 1: 
        eval_private(nn_model, sgd_matrix, zero_train_matrix, private_data)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    #main()
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    test_data = load_public_test_csv("../data")
    private_data = load_private_test_csv("../data")

    k_vals = [40, 50, 60, 70, 80, 90]
    k = 40
    num_iteration = 4000000
    lr = 0.05
    lmd = 0.065
    iter_step = 1000

    tot_obs = 0
    mu = 0
    num_u = len(set(train_data["user_id"]))
    num_q = len(set(train_data["question_id"]))
    amt_of_data = len(train_data["user_id"])
    bu = np.zeros((num_u, 1))
    bz= np.zeros((num_q, 1))
    for i in range(amt_of_data):
        bu[train_data["user_id"][i]] += 1
        bz[train_data["question_id"][i]] += 1
        tot_obs += 1
        mu += train_data["is_correct"][i]
    bu /= tot_obs
    bz /= tot_obs
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(num_u, k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(num_q, k))
    u2 = u.copy()
    z2 = z.copy()
    mu = mu/tot_obs

    old_losses = []
    new_losses = []
    for i in range(num_iteration):
        if i % iter_step == 0:
            print(i)
        if i >= 500000 and i % iter_step == 0:
            old_losses.append(mfac.squared_error_loss(train_data, u2, z2))
            #new_losses.append(squared_error_loss(val_data, u, z, bu, bz, mu, lmd)[0])
        #u, z, bu, bz = update_u_z_b(train_data, lr, u, z, bu, bz, mu, lmd, [], False)
        u2, z2 = mfac.update_u_z(train_data, lr, u2, z2)
    plt.plot(range(500000, num_iteration, iter_step), old_losses, color="blue")
    print(evaluate_sgd(np.add(np.add(np.dot(u, z.T),bu), bz.T)+ mu, test_data))
    #plt.plot(range(500000, num_iteration, iter_step), new_losses, color="red")
    plt.show()

