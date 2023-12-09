
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import time
import recommenders0001 as recommender
# %matplotlib inline

"""## Loading the Song Dataset"""


user_song_metadata = "song_data.csv"
triplets_file = "10000.txt"

meta_file_cols = ['user_id','song_id','listen_count']
song_meta_df = pd.read_table(triplets_file, header=None,sep='\t')
song_meta_df.columns = meta_file_cols

user_song_data = pd.read_csv(user_song_metadata)
#Mergin the Two Dataframes to form User ID and Song Co-occurance
song_df = pd.merge(song_meta_df,user_song_data.drop_duplicates(['song_id']), on="song_id",how="left")

song_df.head(3)

print(f"Length Of Total Dataset: {len(song_df)}")

"""## Creating a Subset Of the Data(10,000)"""

song_df = song_df.head(10000)

#Merging the Song Title and Artist Name Column
song_df['song'] = song_df['title'] + " - " + song_df['artist_name']
song_df.head(1)

"""## Showing the Popular Songs in Dataset"""

#Creating the Count For Each Unique Song for User ID's(Listen Count)
song_counter = song_df.groupby(['song']).agg({'listen_count':'count'}).reset_index()
total_song_listen = song_counter['listen_count'].sum()
#Creating the Song Listen % (Describes it's Worth as a Whole)
song_counter['percentage'] = (song_counter['listen_count'].div(total_song_listen))*100
song_counter.head(3)

#The Maximum Rated Song on the Subset:
song_counter[song_counter.percentage == song_counter['percentage'].max()]

"""###  The Uniqueness from the Dataset"""

print(f"No of Unique Users in Dataset: {len(song_df['user_id'].unique())}")
print(f"No Of Unique Songs in Dataset: {len(song_df['song'].unique())}")

"""## Getting Into Recommendation"""

#First Let's Seperate the Dataset Into Testing and Training Set:
training_data, testing_data = train_test_split(song_df,test_size=0.20,random_state=42)
training_data.head(1)

"""## Let't Try Using Popularity Based Recommender
Popularity Based Recommendor is a Type of Collaborative Filtering Technique. It is a Type of Recommender System which Basically <b>Works with Trend(Current).</b> It Produces Same Resuly for Each Unique Users.
<br><br>
<b>But Collabrative Filtering Have Two Main CONS:<br><b>
<tab><b><i>1.Data Sparsity <br> 2.Computationally Expensive</b></i><br>
<img src="colabfil.png" />
<br>
    <center>
    Source from <i><b>GroupLens Research Group/Army HPC Research Center <br></i></b>
        </center>
    <center>
        Appears in WWW10, May 1-5, 2001, Hong Kong.
    </center>
"""

#We have Coded the Model for Popularity Based Recommender in Seperate Python File
#Creating an Instance of the Pop-Recommender Model
popularity_based_recommendor = recommender.popularity_recommender_py()
popularity_based_recommendor.create(training_data,'user_id','song')

#Dataframe for Unique Users:
Users = song_df['user_id'].unique()
print(f'User DF is {type(Users)}')

"""## Using Popularity Based Model for Predicting the Recommendations for Users"""

#Getting the Top 10 Recommended Songs for the User based on the Popularity
user_id1 = Users[45]
popularity_based_recommendor.recommend(user_id1)

"""# Item Based Similarity Recommender
## -->Personalized Systems Indented for Individuals
### Item Based Similarity is a Type of Content-Based Filtering. Instred of Looking for people who are Similar to You(User Based Collaborative Filtering), It looks for Things You Like and Recommend Things that are Similar to those Stuff
<img src="item_based_sim.png" />
<br>
    <center>
    Source from <i><b>GroupLens Research Group/Army HPC Research Center <br></i></b>
        </center>
    <center>
        Appears in WWW10, May 1-5, 2001, Hong Kong.
    </center>

### Here We are creating User Item Co-Occurence Matrix, Which is an Array Of the Users and Their Ratings to the Songs.Consider this Link for Knowing the Term Co-Occurence Matrix:
https://learnforeverlearn.com/cooccurrence/
<hr>
"""

#Recommenders.item_similarity_recommender_py(Method for Item Based Recommendation)

per_r_model = recommender.item_similarity_recommender_py()
per_r_model.create(training_data, 'user_id', "song")

"""## Generating Song Recommendation Using Personalized Model"""

user_id = Users[4]
user_songs = per_r_model.get_user_items(user_id)

print("------------------------------------------------------------------------------------")
print(f"Training data songs for the user userid: {user_id}")
print("------------------------------------------------------------------------------------")

#Printing the User's Unique Songs:
for songs in user_songs:
    print(songs)

print("----------------------------------------------------------------------")
print("Recommendation process going on:")
print("----------------------------------------------------------------------")
per_r_model.recommend(user_id)

"""## Getting Similar Songs for A Song:"""

per_r_model.get_similar_items(['U Smile - Justin Bieber'])

"""## Now It's Time to Calculate the Precision and Recall
### F1 Score is the Harmonic Mean of the Precision and Recall
<img src="f1.png" />
<br>
==> Precision is the percentage of the Your Results which are correct<br><br>
==> Recall refers to the Total Relevent results correctly Classified by Your Algorithm
"""

import Evaluation
start = time.time()

#Define what percentage of users to use for precision recall calculation
user_sample = 0.05

#Instantiate the precision_recall_calculator class
pr = Evaluation.precision_recall_calculator(testing_data, training_data, popularity_based_recommendor, per_r_model)

#Call method to calculate precision and recall values
(pm_avg_precision_list, pm_avg_recall_list, ism_avg_precision_list, ism_avg_recall_list) = pr.calculate_measures(user_sample)

end = time.time()
print(end - start)

import pylab as pl

#Method to generate precision and recall curve
def plot_precision_recall(m1_precision_list, m1_recall_list, m1_label, m2_precision_list, m2_recall_list, m2_label):
    pl.clf()
    pl.plot(m1_recall_list, m1_precision_list, label=m1_label)
    pl.plot(m2_recall_list, m2_precision_list, label=m2_label)
    pl.xlabel('Recall')
    pl.ylabel('Precision')
    pl.ylim([0.0, 0.20])
    pl.xlim([0.0, 0.20])
    pl.title('Precision-Recall curve')
    #pl.legend(loc="upper right")
    pl.legend(loc=9, bbox_to_anchor=(0.5, -0.2))
    pl.show()

print("Plotting precision recall curves.")

plot_precision_recall(pm_avg_precision_list, pm_avg_recall_list, "popularity_model",
                      ism_avg_precision_list, ism_avg_recall_list, "item_similarity_model")
import random
import pickle

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error

import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from model import vae
from data_loader import DataLoader


class VaeDataset(torch.utils.data.Dataset):

    def __init__(self, df):
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        user_id = np.array(self.df.iloc[idx]['userID'])
        item_id = np.array(self.df.iloc[idx]['itemID'])
        label = np.array(self.df.iloc[idx]['label'], dtype=np.float32)
        return user_id, item_id, label

def prepare_arguments(arguments=None):

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
    parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use, sum, concat, or neighbor')
    parser.add_argument('--mixer', type=str, default='attention', help='which mixer to use, attention or transe?')
    parser.add_argument('--n_epochs', type=int, default=20, help='the number of epochs')
    parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
    parser.add_argument('--dim', type=int, default=16, help='dimension of user and entity embeddings')
    parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--ratio', type=float, default=0.8, help='size of training dataset')
    parser.add_argument('--random_seed', type=int, default=1, help='random seed')
    arguments = arguments.split() if arguments else None
    return parser.parse_args(arguments)

def experiment(arguments=None):

    # Initialization
    args = prepare_arguments(arguments)

    random.seed(args.random_seed)

    torch.manual_seed(args.random_seed)

    # Data Preparation: build dataset and knowledge graph
    data_loader = DataLoader(args.dataset)
    kg = data_loader.load_kg()
    df_dataset = data_loader.load_dataset()
    print(df_dataset)

    # Dataset Splitting
    x_train, x_test, y_train, y_test = train_test_split(
    df_dataset, df_dataset['label'],  # x are triplets; a triplet is (user, item, label)
    test_size=1 - args.ratio, shuffle=False, random_state=999)
    train_dataset = VaeDataset(x_train)
    test_dataset = VaeDataset(x_test)


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

    # Model Preparation: network, loss function, optimizer

    num_user, num_entity, num_relation = data_loader.get_num()
    user_encoder, entity_encoder, relation_encoder = data_loader.get_encoders()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = vae(num_user, num_entity, num_relation, kg, args, device).to(device)
    criterion = torch.nn.BCELoss()  # binary cross entropy loss
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.l2_weight)
    print(f'[Info] Using {device}.')

    # Results in List
    loss_list = []
    test_loss_list = []
    auc_score_list = []
    acc_list = []
    f1_sc_list = []
    mean_list = []

    # Training
    for epoch in range(args.n_epochs):
        running_loss = 0.0
        for i, (user_ids, item_ids, labels) in enumerate(train_loader):
            user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
            optimizer.zero_grad()  # Reset the gradients of all the parameters in the optimizer to zero
            outputs = net(user_ids, item_ids)
            loss = criterion(outputs, labels)
            loss.backward()  # Calculate the gradients of the parameters
            optimizer.step()  # Update the parameters using the gradients
            running_loss += loss.item()

        # Print train loss
        print(f'[Epoch {epoch+1:>3}] Train_loss--> {running_loss / len(train_loader):2.6f} ', end='')
        loss_list.append(running_loss / len(train_loader))

        # Evaluation
        with torch.no_grad():  # Disabling gradient calculation
            test_loss = 0
            total_roc = 0
            total_acc = 0.0
            total_f1 = 0.0
            total_mean_error = 0.0
            for user_ids, item_ids, labels in test_loader:  # For each row in the data
                user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
                outputs = net(user_ids, item_ids)
                predict = outputs.cpu().detach().numpy()

                test_loss += criterion(outputs, labels).item()
                total_roc += roc_auc_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
                total_acc += accuracy_score(labels.cpu().detach().numpy(), predict > 0.5)
                total_f1 += f1_score(labels.cpu().detach().numpy(), predict > 0.5)
                total_mean_error += mean_squared_error(labels.cpu().detach().numpy(), predict > 0.5)

            print(f'Test_loss--> {test_loss / len(test_loader):2.6f}' , f'AUC--> {total_roc / len(test_loader):2.6f}'
                  ,f'Accuracy--> {total_acc / len(test_loader):2.6f}',f'F1-Score--> {total_f1 / len(test_loader):2.6f}',
                  f'Mean Squared Error-->{total_mean_error / len(test_loader):2.6f}')

            test_loss_list.append(test_loss / len(test_loader))
            auc_score_list.append(total_roc / len(test_loader))
            acc_list.append(total_acc / len(test_loader))
            f1_sc_list.append(total_f1 / len(test_loader))
            mean_list.append(total_mean_error / len(test_loader))

    return loss_list, test_loss_list, auc_score_list , acc_list , f1_sc_list , mean_list

if __name__ == '__main__':
    arguments = '--dataset music --mixer transe --n_epochs 5 --batch_size=16 --l2_weight 1e-4'
    loss_list, test_loss_list, auc_score_list , acc_list , f1_sc_list , mean_list = experiment(arguments)

    # plot losses / scores
    fig, (ax1,ax2) = plt.subplots(2,1, figsize=(6,6))
    ax1.plot(loss_list , label='loss_list')
    ax1.plot(test_loss_list, label='test_loss_list')
    ax2.plot(auc_score_list, label='auc_score_list')
    plt.tight_layout()
    plt.show()

    # store loss_list, test_loss_list, auc_score_list
    result = [loss_list, test_loss_list, auc_score_list]
    with open('result.pkl', 'wb') as f:  # open a text file
        pickle.dump(result, f) # serialize the result
    with open('result.pkl', 'rb') as f:
        data = pickle.load(f)
        print(data)