import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding,Conv1D,MaxPooling1D
from keras.layers import Dense,Dropout ,Flatten
from keras.layers import LSTM

from keras.preprocessing import sequence
from keras.constraints import max_norm

from sklearn.model_selection import train_test_split
from sklearn import utils


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def main():

    #read in training dataset
    train_validation_dataset = pd.read_csv('train.tsv', sep='\t', header=0)
    x_train = np.array(list(train_validation_dataset['Phrase']))
    y_train = np.array(list(train_validation_dataset['Sentiment']))

    #read in testing dataset
    testing_dataset = pd.read_csv('test.tsv', sep='\t', header=0)
    x_test = np.array(list(testing_dataset['Phrase']))
    x_test_id = np.array(list(testing_dataset['PhraseId']))

    #build Tokenizer and allow unknown token in test dataset
    tokenizer = keras.preprocessing.text.Tokenizer(oov_token = True)
    tokenizer.fit_on_texts(x_train)
    #specific tokenizer vocabulary size
    Tokenizer_vocab_size = len(tokenizer.word_index) + 1

    #split training and validation set
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1)

    #max word count limit and max dictionary limit
    max_word_count = 60
    max_dic_size = Tokenizer_vocab_size

    #changing the text to the same size
    encoded_train = tokenizer.texts_to_sequences(x_train)
    encoded_val = tokenizer.texts_to_sequences(x_val)
    encoded_test = tokenizer.texts_to_sequences(x_test)
    x_train_words = sequence.pad_sequences(encoded_train, maxlen=max_word_count)
    x_val_words = sequence.pad_sequences(encoded_val, maxlen=max_word_count)
    X_test_words = sequence.pad_sequences(encoded_test, maxlen=max_word_count)

    # One Hot Encoding
    y_train = keras.utils.to_categorical(y_train, 5)
    y_val   = keras.utils.to_categorical(y_val, 5)

    #training set needs to be shuffled for a balanced training model
    x_train_words, y_train = utils.shuffle(x_train_words, y_train)

    
    #model design
    model = Sequential()

    model.add(Embedding(max_dic_size, 32, input_length=max_word_count))
    model.add(Conv1D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(Conv1D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    #add a LSTM layer
    model.add(LSTM(64,return_sequences=True))
    model.add(Flatten())
    model.add(Dense(64, activation='relu',kernel_constraint=max_norm(1)))

    model.add(Dropout(0.5))
    #output layer
    model.add(Dense(5, activation='softmax'))

    model.summary()

    #training using optimizers = Nadam
    epochs = 40
    batch_size = 32
    Nadam = keras.optimizers.Nadam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=Nadam, metrics=['accuracy'])


    training_history  = model.fit(
        x_train_words, 
        y_train, 
        epochs = epochs, 
        batch_size=batch_size, 
        validation_data=(x_val_words, y_val)
        )

    model.save('sentiment_analysis.keras')
    


    # plot accuracy over epoch graph
    plt.plot(training_history.history['accuracy'])
    plt.plot(training_history.history['val_accuracy'])
    plt.xlabel('epoch [times]')
    plt.ylabel('accuracy [rate]')
    plt.legend(['training','validation'], loc='upper left')
    plt.savefig('accuracy.png')
    

    # model =  tf.keras.models.load_model('sentiment_analysis.keras')
    # test model over test dataset
    predictions = model.predict(X_test_words, batch_size=batch_size)
    # write in to a separate csv file
    test_result_file = 'test_result.csv'
    Sentiment_list = []
    PhraseId_list = []

    sentiment = ['0', '1', '2', '3', '4']
    print(predictions.shape)
    for i in range(0, predictions.shape[0]):
        sentiment_category = predictions[i, :].argmax(axis=-1)
        Sentiment_list += [sentiment[sentiment_category]]
        PhraseId_list += [x_test_id[i]]

    testing_result = pd.DataFrame()
    testing_result['PhraseId'] = PhraseId_list
    testing_result['Sentiment'] = Sentiment_list

    testing_result.to_csv(test_result_file, index=False)

if __name__ == "__main__":
    main()