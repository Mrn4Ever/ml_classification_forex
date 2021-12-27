import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



def main ():
    symbol          = 'EURUSD'
    timeframe       = 16385
    fileName        = f'./data/{symbol}-{str(timeframe)}.csv'
    
    features        = ['open','high','low','close','volume']
    iHistorical     = 24
    iRound          = 2
    testSize        = 0.3
    bUseScaledData  = True
    epochs          = 100

    print('.oO(Machine Learning - Classification Forex)')
    print(f'1 - Load Data {symbol} {timeframe}...')
    df_prices = pd.read_csv(fileName)
    df_prices.set_index('time',inplace=True)
    print(df_prices.tail())

    print(f'2 - Clean, Feed and Transform Data...')

    #Caclulate Candle Classes
    df_prices['size'] = df_prices['close'] - df_prices['open']
    df_prices['cat']  = df_prices['size'].apply(setCategory)
    print(df_prices.tail())

    print('3- Candle Classes Repartition:')
    print (df_prices['cat'].value_counts())

    print('4- Windowing Data...')
    data     = df_prices [features].values
    X        = []
    X_scaled = []
    y        = []
    icount = 1
    for i in df_prices.index:
        if (icount > iHistorical):
            y.append(df_prices.loc[i,'cat'])
            windowData =  data[icount-iHistorical-1: icount-1][:]
            X.append(windowData)

            #Here We will scale on each column to get relevant scaling data
            windowDataScaled = windowData.copy()
            for c in range(windowData.shape[1]):
                col = windowDataScaled[:,c]
                windowDataScaled[:,c] = np.interp(col, (col.min(), col.max()), (0, 1))
            X_scaled.append(np.around(windowDataScaled,decimals=iRound))
        icount += 1

    X           = np.array(X)
    X_scaled    = np.array(X_scaled)
    y           = np.array(y)
    lastindex = y.shape[0] -1

    print (f'X  Shape {str(X.shape)}')
    print (f'X_scaled  Shape {str(X_scaled.shape)}')
    print (f'y  Shape {str(y.shape)}')

    print('5- Checking Consistancy...')
    if (X.shape[0] == y.shape[0] and X_scaled.shape[0] == y.shape[0]):
        print ('[OK]')
    else:
        print ('[NOK]')
        return

    """
    lastIndex = y.shape[0] - 1
    print (f'y category is {y[lastIndex]}')
    print ('X :')
    print (X[lastIndex])
    print ('X_scaled :')
    print (X_scaled[lastIndex])
    """

    print('6- Splitting data...')
    if bUseScaledData:
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=testSize)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize)

    print('7- Defining Model...')
    keras.backend.clear_session()

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(X_train.shape[1], X_train.shape[2])),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(X_train.shape[1] * X_train.shape[2], activation=tf.nn.relu),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(50, activation=tf.nn.relu),
        keras.layers.Dense(25, activation=tf.nn.relu),
        keras.layers.Dense(5, activation=tf.nn.relu),
        keras.layers.Dense(3, activation=tf.nn.softmax)
    ])

    print('8- Defining Optimizer...')
    model.compile(optimizer='SGD',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    print('9- Traning Data...')
    model.fit(X_train, y_train, epochs=epochs)

    print('9- Evaluate Model...')
    test_loss, test_acc =  model.evaluate(X_test, y_test)


def setCategory(x):
    #Significant Candle absolute size is grater than 5Pips (50 Points)
    val = 0.0004     
    if x > val: return  0
    if x < -val: return 1
    return 2


if __name__ =='__main__':
    main()