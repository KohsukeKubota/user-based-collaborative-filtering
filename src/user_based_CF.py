import numpy as np
import pandas as pd
from tqdm import tqdm


def cosine_similarity(v1, v2, mean_adjustment=False):
    if mean_adjustment:
        v1 = v1 - np.mean(v1)
        v2 = v2 - np.mean(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

if __name__ == '__main__':
    # read MovieLens data
    train = pd.read_csv('../data/input/ml-100k/ua.base', names=["user_id", "item_id", "rating", "timestamp"], sep='\t')
    test = pd.read_csv('../data/input/ml-100k/ua.test', names=['user_id', 'item_id', 'rating', 'timestamp'], sep='\t')

    # create table of index=user_id, columns=item_id
    train_rating_mat = pd.pivot_table(train, index='user_id', columns='item_id', values='rating')
    train_rating_mat.fillna(0,  inplace=True)

    rating_arr = train_rating_mat.values.T
    train.set_index('item_id', inplace=True)
    precision_list = []

    # use five people for evaluation of recommend system
    for user1_id in tqdm([1, 100, 233, 666, 888]):
        cos_sim_list = []
        for user2_index in range(rating_arr.shape[1]):
            user1 = rating_arr[:, user1_id-1][:, np.newaxis]
            user2 = rating_arr[:, user2_index][:, np.newaxis]
            two_users_mat = np.concatenate((user1, user2), axis=1)
            two_users_mat = two_users_mat[~np.isnan(two_users_mat).any(axis=1), :]
            # calucalate cosine similarity between user1 and user2
            cos_sim = cosine_similarity(two_users_mat[:, 0], two_users_mat[:, 1], mean_adjustment=True)
            cos_sim_list.append(cos_sim)
        cos_sim_mat = pd.Series(cos_sim_list, index=[i for i in range(1, rating_arr.shape[1] + 1)])

        # use top 10 users of cosine similarity
        top_n = 10
        top_n_sim = cos_sim_mat.sort_values(ascending=False)[1:top_n+1]
        top_n_users = top_n_sim.index

        # test data of user1
        test_user1 = test[test['user_id'] == user1_id].sort_values(by='rating', ascending=False)

        # calculate the prediction of user1
        user1_not_rating = train_rating_mat.iloc[user1_id-1, :]
        user1_not_rating = pd.Series(np.logical_not(user1_not_rating), index=user1_not_rating.index)
        mean_r = train_rating_mat.replace(0, np.nan).mean(axis=1).drop(labels=[user1_id])
        mean_r = mean_r[mean_r.index.isin(top_n_users)]

        not_user1_rating_item = train[train.index.isin(user1_not_rating[user1_not_rating == 1].index)]
        not_user1_rating_item = not_user1_rating_item[not_user1_rating_item['user_id'].isin(top_n_users)]
        not_user1_rating_item.reset_index(inplace=True)

        ra = train_rating_mat.replace(0, np.nan).iloc[0, :].mean()
        bottom_value = np.sum(top_n_sim)
        item_id_list = []
        pred_list = []
        hits = 0

        # recommend top 10 item
        for item_id in not_user1_rating_item['item_id'].unique():
            rating_by_item = not_user1_rating_item[not_user1_rating_item['item_id'] == item_id]
            top_value = np.sum([top_n_sim[uid] * (rating_by_item[rating_by_item['user_id'] == uid]['rating'].values - mean_r[uid]) for uid in rating_by_item['user_id'].values])
            pred = ra + top_value / bottom_value
            item_id_list.append(item_id)
            pred_list.append(pred)

        # check the precision of recommend list
        pred_dict = {'item_id': item_id_list, 'pred': pred_list}
        pred_df = pd.DataFrame.from_dict(pred_dict).sort_values(by='pred', ascending=False).reset_index(drop=True)
        recommend_list = pred_df[:10]['item_id'].values
        purchase_list = test_user1['item_id'].values
        for item_id in recommend_list:
            if item_id in purchase_list:
                hits += 1
        precision_ = hits / 10.0
        precision_list.append(precision_)
    print('precision: {:.2f}'.format(sum(precision_list) / len(precision_list)))
