import pandas as pd
import numpy as np


def preprocess_yow(path):
    df = pd.read_csv(path)

    # remove NaN values
    clds = df[df['classes'].notna()]

    # get 'people' topics
    txds = clds[clds['classes'].str.contains("people")]

    # select two most popular RSS IDs
    top2_RSS_ID = txds.RSS_ID.value_counts().nlargest(2).to_dict()

    top1 = list(top2_RSS_ID.keys())[0]
    top2 = list(top2_RSS_ID.keys())[1]

    top1_df = txds.loc[txds['RSS_ID'] == top1]
    top2_df = txds.loc[txds['RSS_ID'] == top2]

    final_df = pd.concat([top1_df, top2_df])

    # normalize relevance to between 0 and 1, then add Gaussian noise,
    final_df['relevant'] = (final_df['relevant'] / 5) + np.random.normal(0, 0.05, final_df['relevant'].count())

    # then clip
    final_df.loc[final_df['relevant'] < 0] = 0
    final_df.loc[final_df['relevant'] > 1] = 1

    return final_df

