#!/usr/bin/env python
# coding: utf-8

# ## Ques 1 Consider the entire Yale Faces datase

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from random import sample
import shutil
from zipfile import ZipFile
import warnings
warnings.filterwarnings("ignore") 


# In[4]:


my_dir = os.getcwd() 
zip_folder = os.path.join(my_dir,"yalefaces.zip")
print("Path to the zipped folder is {}".format(zip_folder))
with ZipFile(zip_folder, 'r') as zip: 
    zip.extractall()


# In[5]:


data_folder = os.path.join(my_dir,"yalefaces")
file_list = os.listdir(data_folder)
print(file_list)
sample_images = sample(file_list,10) 
for img_name in sample_images:
    image_path = os.path.join(data_folder,img_name)
    image = mpimg.imread(image_path)
    plt.imshow(image,cmap="gray")
    plt.title(img_name)
    plt.show()


# In[6]:


train_folder_path = os.path.join(my_dir,"Faces_Train")
test_folder_path = os.path.join(my_dir,"Faces_Test")

## Delete the folders if they exist already
if os.path.exists(train_folder_path):
    shutil.rmtree(train_folder_path)

if os.path.exists(test_folder_path):
    shutil.rmtree(test_folder_path)

os.mkdir(train_folder_path)  ## Creates a new directory
os.mkdir(test_folder_path) ## Creates a new directory

idx_list = [str(i).zfill(2) for i in range(1,16,1)] 
print(idx_list)

file_name_list = [[] for i in range(15)]

for i in range(len(idx_list)):
    for fname in file_list:
        if fname.startswith("subject"+idx_list[i]):
            file_name_list[i].append(os.path.join(data_folder,fname))

print(file_name_list)

for i in range(len(idx_list)):
    ls = file_name_list[i]  
    test_idx = np.random.choice(11)  
    test_file = ls[test_idx]
    shutil.copy(test_file,test_folder_path)
    
    ls.remove(ls[test_idx])
    
    for train_file in ls:
        shutil.copy(train_file,train_folder_path)


# In[11]:


train_files_list = os.listdir(train_folder_path)
#print(train_files_list)
train_ls = []
for file in train_files_list:
    img_file = os.path.join(train_folder_path,file)
    arr = mpimg.imread(img_file)
    #print(arr.shape)
    arr = arr.reshape(1,arr.shape[0]*arr.shape[1]) ## Convert to a 1D matrix
    train_ls.append(np.ravel(arr)) ## Before appending, convert the 1D martix to a 1d array using np.ravel 
train_mat = np.matrix(train_ls)
print(train_mat.shape)
mean_img = np.mean(train_mat,axis=0)
print(mean_img.shape)
cov = np.cov(train_mat)
print(cov.shape)
eig_val,eig_vec = np.linalg.eig(cov)
#print(eig_vec)


# In[14]:


eigen_vec_ls = []
#eig1 = train_mat.T@eig_vec[:,0]
#print(eig1.shape)
for i in range(eig_vec.shape[1]):
    eig1 = train_mat.T@eig_vec[:,i]
    eig1 = eig1/eig_val[i]
    eigen_vec_ls.append(np.ravel(eig1))
#print(eigen_vec_ls)


# In[15]:


sort_idx = np.argsort(eig_val) ## indices for eigenvalues which are in ascending order
sort_idx = sort_idx[::-1]

eig_val_sum = np.sum(eig_val)
temp_sum = 0
principal_eig_vec = []
principal_eig_val = []
i=0
while(temp_sum<0.95*eig_val_sum):
    principal_eig_vec.append(eigen_vec_ls[sort_idx[i]])
    principal_eig_val.append(eig_val[sort_idx[i]])
    temp_sum += eig_val[sort_idx[i]]
    i += 1
print("Number of components is {}".format(i))


# ## Ques 2 . Obtain the transformation matrix Q and mean vector μ by performing Principal Component analysis on the dataset

# In[20]:


Q_hat = np.matrix(principal_eig_vec)
mu=mean_img


# In[22]:


print(Q_hat)
print(50*"*")
print(mu)


# ## Ques 3 Obtain the feature vector for every training set by using the transformation v = QT (x − μ).

# In[18]:


test_files = os.listdir(test_folder_path)
feat_vec_ls = []
for file in test_files:
    img_file = os.path.join(test_folder_path,file)
    test_img = mpimg.imread(img_file)
    test_img = test_img.reshape(arr.shape[0]*arr.shape[1],1)
    test_img = test_img - mean_img.T
    #print(np.linalg.pinv(Q_hat).shape,test_img.shape)
    feat_vec = np.linalg.pinv(Q_hat).T@test_img
    feat_vec_ls.append(np.ravel(feat_vec))


# In[26]:


vec_len=len(feat_vec_ls[0])


# ## Ques 4 Generate around 50 random vectors of dimension n_dim, where n_dim is the number of dimensions in v.

# In[23]:


def genRandomHashVectors(m, length):  # Generate random unit vectors for Hashing
    hash_vector = []
    for i in range(m):
        v = np.random.uniform(-1,1,length)
        vcap = v / np.linalg.norm(v)
        hash_vector.append(vcap)
    return hash_vector   


# In[49]:


hash_vec=genRandomHashVectors(50, vec_len)
hash_vec


# ## Ques 5 Generate 50-bit hash representation of each of the feature vectors.

# In[24]:


def localSensitiveHashing(hash_vector ,data): 
    hash_code = []
    for i in range(len(hash_vector)):
        if np.dot(data,hash_vector[i]) > 0:
            hash_code.append('1')
        else:
            hash_code.append('0')
    return hash_code 


# In[51]:


localSensitiveHashing(hash_vec ,feat_vec_ls[1])


# ## Ques 6 Calculate the L1-norm distance between the hash representation of a par- ticular feature vector with the hash representation of other feature vectors and sort the vectors based on the distance values.

# In[31]:


n= np.random.randint(1,vec_len)
list_1=[]
n


# In[44]:


for i in range (len(feat_vec_ls)):
    list_1.append(np.linalg.norm((feat_vec_ls[i]- feat_vec_ls[n]),ord=1))


# In[45]:


list_1.sort()


# In[46]:


print(list_1)


# In[ ]:





# In[ ]:




