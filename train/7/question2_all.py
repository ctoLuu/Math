import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import itertools
from tqdm import tqdm
from collections import namedtuple

from sklearn.preprocessing import MinMaxScaler, LabelEncoder


# Define feature classes and NCF model components
class SparseFeat:
    def __init__(self, name, vocabulary_size, embedding_dim):
        self.name = name
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim


def build_input_layers(feature_columns):
    dense_input_dict, sparse_input_dict = {}, {}
    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            sparse_input_dict[fc.name] = Input(shape=(1,), name=fc.name)
    return dense_input_dict, sparse_input_dict


def build_embedding_layers(feature_columns, input_layers_dict, is_linear, prefix=''):
    embedding_layers_dict = {}
    sparse_feature_columns = [x for x in feature_columns if isinstance(x, SparseFeat)]
    if is_linear:
        for fc in sparse_feature_columns:
            embedding_layers_dict[fc.name] = Embedding(fc.vocabulary_size + 1, 1, name=prefix + '1d_emb_' + fc.name)
    else:
        for fc in sparse_feature_columns:
            embedding_layers_dict[fc.name] = Embedding(fc.vocabulary_size + 1, fc.embedding_dim,
                                                       name=prefix + 'kd_emb_' + fc.name)
    return embedding_layers_dict


def get_dnn_out(dnn_inputs, units=(32, 16)):
    dnn_out = dnn_inputs
    for out_dim in units:
        dnn_out = Dense(out_dim)(dnn_out)
    return dnn_out


def NCF(dnn_feature_columns):
    _, sparse_input_dict = build_input_layers(dnn_feature_columns)
    input_layers = list(sparse_input_dict.values())
    GML_embedding_dict = build_embedding_layers(dnn_feature_columns, sparse_input_dict, is_linear=False, prefix='GML')
    MLP_embedding_dict = build_embedding_layers(dnn_feature_columns, sparse_input_dict, is_linear=False, prefix='MLP')
    GML_user_emb = Flatten()(GML_embedding_dict['user_id'](sparse_input_dict['user_id']))
    GML_item_emb = Flatten()(GML_embedding_dict['item_id'](sparse_input_dict['item_id']))

    # Use Keras Multiply layer instead of tf.multiply
    GML_out = Multiply()([GML_user_emb, GML_item_emb])

    MLP_user_emb = Flatten()(MLP_embedding_dict['user_id'](sparse_input_dict['user_id']))
    MLP_item_emb = Flatten()(MLP_embedding_dict['item_id'](sparse_input_dict['item_id']))
    MLP_dnn_input = Concatenate(axis=1)([MLP_user_emb, MLP_item_emb])
    MLP_dnn_out = get_dnn_out(MLP_dnn_input, (32, 16))
    concat_out = Concatenate(axis=1)([GML_out, MLP_dnn_out])
    output_layer = Dense(1)(concat_out)
    model = Model(input_layers, output_layer)
    return model


def prepare_data(file_path):
    data = pd.read_excel(file_path)
    data['StockCode'] = data['StockCode'].astype(str)
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    data = data.dropna(subset=['CustomerID'])  # Drop rows where CustomerID is NaN
    data['CustomerID'] = data['CustomerID'].astype(int)

    # Create a time index to separate data by months
    data.set_index('InvoiceDate', inplace=True)
    data.sort_index(inplace=True)

    # Extract months
    months = data.index.to_period('M').unique()
    data['InvoiceDate'] = pd.to_datetime(data.index)
    data['Month'] = data['InvoiceDate'].dt.to_period('M')
    weekly_purchase_frequency = data.groupby(['CustomerID', 'StockCode', 'Month']).size().reset_index(
        name='rating')

    # weekly_purchase_frequency['InvoiceDate'] = data['InvoiceDate']
    # weekly_purchase_frequency.set_index('InvoiceDate', inplace=True)
    # weekly_purchase_frequency.sort_index(inplace=True)
    data = weekly_purchase_frequency
    data['monthly_rating'] = data.groupby(['StockCode', 'Month'])['StockCode'].transform('count')
    data['rating'] = data['monthly_rating']
    # Split data into training (first 5 months) and test (6th month)
    # Encoding
    lbe_user = LabelEncoder()
    lbe_item = LabelEncoder()
    data['user_id'] = lbe_user.fit_transform(data['CustomerID'])
    data['item_id'] = lbe_item.fit_transform(data['StockCode'])
    item_id_to_stockcode = dict(zip(lbe_item.transform(lbe_item.classes_), lbe_item.classes_))
    train_weeks = months[:]

    random_values = np.random.choice(data['user_id'].values, size=1000, replace=False)
    repeated_values = np.repeat(random_values, 3665)
    test_data = pd.DataFrame()
    test_data['user_id'] = repeated_values
    test_data['item_id'] = np.tile(np.arange(3665), len(random_values))

    train_data = data.loc[data['Month'].isin(train_weeks)]


    return train_data, test_data, item_id_to_stockcode, data


def main():
    file_path = './handled_data2.xlsx'
    train_data, test_data, item_id_to_stockcode, data = prepare_data(file_path)

    # Print the mapping between item_id and StockCode before training the model
    print("Mapping from item_id to StockCode:")
    for item_id, stockcode in item_id_to_stockcode.items():
        print(f"item_id: {item_id} -> StockCode: {stockcode}")

    # Define feature columns for the model
    dnn_feature_columns = [
        SparseFeat('user_id', data['user_id'].nunique(), 8),
        SparseFeat('item_id', data['item_id'].nunique(), 8)
    ]

    # Build and compile the model
    model = NCF(dnn_feature_columns)
    model.summary()
    model.compile(optimizer="adam", loss="mse", metrics=['mae'])

    # Prepare training data
    train_model_input = {name: train_data[name] for name in ['user_id', 'item_id']}
    train_labels = train_data['rating']

    # Train the model
    model.fit(train_model_input, train_labels, batch_size=32, epochs=50, validation_split=0.2)

    # Predict on test data
    test_model_input = {name: test_data[name] for name in ['user_id', 'item_id']}
    test_predictions = model.predict(test_model_input)

    # Add predictions to test_data
    test_data['predicted_rating'] = test_predictions

    # Find the top 5 items with the highest predicted rating
    top_items = test_data.groupby('item_id').agg({'predicted_rating': 'mean'}).reset_index()
    top_items = top_items.sort_values(by='predicted_rating', ascending=False).head(5)

    # Map back to original StockCode
    top_items['StockCode'] = top_items['item_id'].map(item_id_to_stockcode)

    # Find the customer who spent the most
    customer_spending = train_data.groupby('user_id').agg({'rating': 'sum'}).reset_index()
    top_customer = customer_spending.sort_values(by='rating', ascending=False).iloc[0]['user_id']

    # Recommend top 5 items for the top customer
    top_customer_data = train_data[train_data['user_id'] == top_customer]
    top_customer_items = top_customer_data.groupby('item_id').agg({'rating': 'sum'}).reset_index()
    top_customer_items = top_customer_items.sort_values(by='rating', ascending=False).head(5)
    top_customer_items['StockCode'] = top_customer_items['item_id'].map(item_id_to_stockcode)

    print("Top 5 items to recommend:")
    print(top_items[['StockCode', 'predicted_rating']])

    print("Top 5 items bought by top customer:")
    print(top_customer_items[['StockCode', 'rating']])


if __name__ == "__main__":
    main()