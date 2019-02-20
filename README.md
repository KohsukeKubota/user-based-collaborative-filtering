amazonやNetflixで使われているレコメンドシステムに突然興味を持ったので、推薦システムの勉強をしたのでpythonで実装してみました。

勉強のアウトプットなので、間違っていたらご指摘いただけると幸いです。


# 推薦システムとは
amazonでモノを買ったりすると、「この商品をチェックした人はこんな商品もチェックしています」みたいな推薦をしてくれますよね。
まさにあれが推薦システムで、特定のユーザーに対してどのモノを推薦すべきかを決定するシステムのことを**推薦システム**といいます。
その中でも、すべてのユーザーの嗜好に応じて異なる推薦のリストを提示するシステムが個人化推薦といいます。

種類としては、

+ 協調型推薦
+ 内容ベース型推薦
+ 知識型ベース型
+ 複数のアプローチを組み合わせたハイブリッド型

があります。

今回は、協調型推薦の中のユーザーに基づいた推薦が対象です。

# 協調型推薦
協調型推薦は、
*過去に同じ興味を共有したユーザーは将来的にも同じような興味をもつだろう*
仮定しています。

なので、ユーザーAとユーザーBの購入履歴が似ていて、ユーザーBがまだ知らないモノをユーザーAが最近購入したとすると、
これをユーザーBに提案することは合理的だといえます。

ユーザーがお互いに示し合わせたわけではなく暗黙的に協調して、膨大なモノの集合から最も有望なモノをフィルタリングするため、
この技術を**協調フィルタリング**と呼んでいます。

協調型推薦を考える上で、以下の問いに答える必要がでてきます。

+ 類似したユーザーというけれど類似したユーザーを膨大なユーザーの中から探し出すのか？
+ そもそも類似ってなに？
+ 類似ってどうやって測る？
+ 新規のユーザーでまだシステムの利用履歴がないけどどう対処する？
+ ユーザーに関して利用できるデータが少ない場合はどうする？
+ 類似ユーザーを探す以外に特定のユーザーがあるモノを好きかどうかを予測できる手法は存在する？

## ユーザーベース推薦
ユーザーベース推薦のシステムのアイディアは非常にシンプルで、

+ 対象ユーザーの評価データ入力して、過去の嗜好と似ている他のユーザー(＝ピアユーザー)を特定する
+ 対象ユーザーがまだ見ていないモノに対する評価をピアユーザーの評価を用いて予測する

ユーザーベース推薦は以下の２つの仮説に基づいています。

+ ユーザーが過去に似た嗜好をもっているなら、その嗜好は将来においても似ている
+ ユーザーの好みは長い間一貫している

実際この辺りの仮定は少し無理があるので、amazonとかは別の手法を用いているみたいです。この辺りはまだ知識不足なので、よくわかってません笑

このあたりを基礎知識として、ユーザーベース推薦を実装してみました。

# 使用するデータセット
今回利用するのはMovieLensデータセットです。
[https://grouplens.org/datasets/movielens/embed]

MovieLensデータセットは推薦システムの開発やベンチマークとしてミネソタ大学の研究グループが公開してくれています。ありがたいですね。

今回はMovieLens 100K Datasetを使用します。

# ベンチマーク（参考コード）
推薦システムの実装で使われていることが多いので、他の人のコードを眺めながらできるのは非常にありがたいです。
ちなみに以下のgithubのコードをベンチマークとして実装しています。
[https://github.com/fuxuemingzhu/MovieLens-Recommender:embed]

今回の実装はMovieLens100kに対するユーザーベース協調フィルタリングになるので、
precisionで19.69%を目標とすることになります。

# 実装
## データの読み込み
データは手元にダウンロードして、使用しています。
ua.baseがtrain dataで、ua.testがtest dataとなっているみたいです。

```python
import numpy as np
import pandas as pd

train = pd.read_csv('../data/input/ml-100k/ua.base', names=["user_id", "item_id", "rating", "timestamp"], sep='\t')
test = pd.read_csv('../data/input/ml-100k/ua.test', names=['user_id', 'item_id', 'rating', 'timestamp'], sep='\t')
```

データとしては以下のような内容です。
データの全容としては、943ユーザーによる1682アイテムに対して1~5の評価をしているレビューサイトのデータという感じです。

amazonをイメージしてもらえると理解の助けになると思うのですが、全員がすべてのアイテムのレビューをしているわけではないので、非常に疎なデータセットになっています。

||user_id|item_id|rating|timestamp|
|---|---|---|---|---|
|0|186|302|3|891717742|
|1|22|377|1|878887116|
|2|244|51|2|880606923|
|3|166|346|1|886397596|
|4|298|474|4|884182806|

+ 類似したユーザーというけれど類似したユーザーを膨大なユーザーの中から探し出すのか？

→今回はすべてのユーザーと総当たりで類以度を計算して、上位何人かを類似したユーザーと定義します。

+ そもそも類似ってなに？

→類以度という概念を導入してユーザー間の類似を表現します。

+ 類似ってどうやって測る？

→類似度にも様々な表現方法があるのですが、今回はコサイン類以度を採用します。

+ 新規のユーザーでまだシステムの利用履歴がないけどどう対処する？
+ ユーザーに関して利用できるデータが少ない場合はどうする？
+ 類似ユーザーを探す以外に特定のユーザーがあるモノを好きかどうかを予測できる手法は存在する？

→上の３つに関しては今回は範囲外としてまた別のタイミングで扱うことにします。

とりあえず今回の記事で扱う話を整理したので、それぞれに注目して行きたいと思います。

## コサイン類以度
今回類以度として採用したコサイン類以度は2つのベクトルを用いて以下のように定式化できます。
[tex: \displaystyle
  sim(\boldsymbol{a}, \boldsymbol{b}) = \frac{\boldsymbol{a} \cdot \boldsymbol{b}}{|\boldsymbol{a} |\times |\boldsymbol{b}|}
]

コサイン類以度は、0から1の値を取りベクトル同士の成す角度の近さを表現しています。

推薦システムの文脈でコサイン類以度を考える場合には、ユーザーが異なることを気をつける必要があります。例えば、ユーザーAは甘めに評価する一方で、ユーザーBは辛めに評価する傾向にあるといったところを考慮する必要があるということです。
これは、ユーザーの評価値の平均をそのユーザーの評価から引くことで解決でき、これを調整コサイン類以度といいます。

実装としては、以下のように実装しています。下のmean_adjustment=Trueとすると調整コサイン類以度になります。
```python
def cosine_similarity(v1, v2, mean_adjustment=False):
    if mean_adjustment:
        v1 = v1 - np.mean(v1)
        v2 = v2 - np.mean(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
```

## 下準備
ここではindexがuser_id、columnsがitem_idになるような評価値行列を作成しています。
ユーザーが今回評価をつけていないところは0としました。調べてみると、Rの推薦システムのライブラリrecommenderlabも欠損は0としているみたいなので、とりあえずこれで進めていきます。

```python
train_rating_mat = pd.pivot_table(train, index='user_id', columns='item_id', values='rating')
train_rating_mat.fillna(0,  inplace=True)
```

## メインの実装部
以下がメインの実装部になっています。
user1には参考にしたコードでテスト用に5人選抜していたので、そのまま流用しています。

手順としては、

+ user1に対して全ユーザーのコサイン類以度を算出
+ 算出されたコサイン類以度の上位10人を選抜
+ 上位10人の近接性とuser1の平均評価値を使ってuser1のアイテムに対する評価値を予測
+ 予測結果からトップ10を用いてレコメンドリストを作成
+ 実際にuser1が購入したリストと比較してprecisionで評価

これを実行するとprecisionが22%になり、少し高いような気がしますが参考にしたコード付近のprecisionになりました。

```python
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
```
