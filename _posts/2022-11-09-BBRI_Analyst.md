---
layout: post
title: "Predicting Closing Price BBRI using LSTM"
subtitle: "Predicting Closing Price of BBRI's Stock"
background: '/img/posts/bbri/bg-bbri.jpg'
---

# Predicting Closing Price of BBRI's Stock

BBRI is the stock name of Bank Rakyat Indonesia. PT Bank Rakyat Indonesia (Persero) Tbk (People's Bank of Indonesia, commonly known as BRI or Bank BRI) is one of the largest banks in Indonesia. It specialises in small scale and microfinance style borrowing from and lending to its approximately 30 million retail clients through its over 4,000 branches, units and rural service posts. It also has a comparatively small, but growing, corporate business. As of 2010 it is the second largest bank in Indonesia by asset.

This dataset can be found using [Pandas DataReader](https://pandas-datareader.readthedocs.io/en/latest/) and will pick from January 2008 to the end of May 2022. It will be more than 14 years. In this session, we will predict the closing price of BBRI with Long Short-Term Memory.

Long short-term memory (LSTM) is an artificial recurrent neural network (RNN) architecture used in the field of deep learning. Unlike standard feedforward neural networks, LSTM has feedback connections. It can not only process single data points (such as images), but also entire sequences of data (such as speech or video). For example, LSTM is applicable to tasks such as unsegmented, connected handwriting recognition, speech recognition and anomaly detection in network traffic or IDS's (intrusion detection systems).


```python
!pip install --upgrade pandas
!pip install --upgrade pandas-datareader
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: pandas in /usr/local/lib/python3.7/dist-packages (1.3.5)
    Requirement already satisfied: pytz>=2017.3 in /usr/local/lib/python3.7/dist-packages (from pandas) (2022.5)
    Requirement already satisfied: numpy>=1.17.3 in /usr/local/lib/python3.7/dist-packages (from pandas) (1.21.6)
    Requirement already satisfied: python-dateutil>=2.7.3 in /usr/local/lib/python3.7/dist-packages (from pandas) (2.8.2)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.7/dist-packages (from python-dateutil>=2.7.3->pandas) (1.15.0)
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: pandas-datareader in /usr/local/lib/python3.7/dist-packages (0.9.0)
    Collecting pandas-datareader
      Downloading pandas_datareader-0.10.0-py3-none-any.whl (109 kB)
    [K     |████████████████████████████████| 109 kB 36.7 MB/s 
    [?25hRequirement already satisfied: lxml in /usr/local/lib/python3.7/dist-packages (from pandas-datareader) (4.9.1)
    Requirement already satisfied: pandas>=0.23 in /usr/local/lib/python3.7/dist-packages (from pandas-datareader) (1.3.5)
    Requirement already satisfied: requests>=2.19.0 in /usr/local/lib/python3.7/dist-packages (from pandas-datareader) (2.23.0)
    Requirement already satisfied: python-dateutil>=2.7.3 in /usr/local/lib/python3.7/dist-packages (from pandas>=0.23->pandas-datareader) (2.8.2)
    Requirement already satisfied: pytz>=2017.3 in /usr/local/lib/python3.7/dist-packages (from pandas>=0.23->pandas-datareader) (2022.5)
    Requirement already satisfied: numpy>=1.17.3 in /usr/local/lib/python3.7/dist-packages (from pandas>=0.23->pandas-datareader) (1.21.6)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.7/dist-packages (from python-dateutil>=2.7.3->pandas>=0.23->pandas-datareader) (1.15.0)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests>=2.19.0->pandas-datareader) (3.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests>=2.19.0->pandas-datareader) (2022.9.24)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests>=2.19.0->pandas-datareader) (2.10)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests>=2.19.0->pandas-datareader) (1.24.3)
    Installing collected packages: pandas-datareader
      Attempting uninstall: pandas-datareader
        Found existing installation: pandas-datareader 0.9.0
        Uninstalling pandas-datareader-0.9.0:
          Successfully uninstalled pandas-datareader-0.9.0
    Successfully installed pandas-datareader-0.10.0
    




```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

## 1) Data Preparation
Using Pandas DataReader, we can get dataset about the stock that we want to use. Of course we're using some libraries to make it happen.


```python
import numpy as np
import pandas as pd
import pandas_datareader as web
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
```


```python
# Catching data BBRI
bbri = web.DataReader('bbri.jk', 'yahoo', '2008-01-01', '2022-05-31')
bbri
```





  <div id="df-4ce132c0-08d6-4fbc-be7f-4df3eecf234d">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Adj Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2008-01-02</th>
      <td>740.0</td>
      <td>730.0</td>
      <td>735.0</td>
      <td>735.0</td>
      <td>44765000.0</td>
      <td>499.284210</td>
    </tr>
    <tr>
      <th>2008-01-03</th>
      <td>735.0</td>
      <td>710.0</td>
      <td>735.0</td>
      <td>720.0</td>
      <td>102940000.0</td>
      <td>489.094513</td>
    </tr>
    <tr>
      <th>2008-01-04</th>
      <td>755.0</td>
      <td>715.0</td>
      <td>720.0</td>
      <td>750.0</td>
      <td>143670000.0</td>
      <td>509.473419</td>
    </tr>
    <tr>
      <th>2008-01-07</th>
      <td>750.0</td>
      <td>730.0</td>
      <td>735.0</td>
      <td>745.0</td>
      <td>63700000.0</td>
      <td>506.077026</td>
    </tr>
    <tr>
      <th>2008-01-08</th>
      <td>755.0</td>
      <td>745.0</td>
      <td>750.0</td>
      <td>750.0</td>
      <td>84225000.0</td>
      <td>509.473419</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2022-05-24</th>
      <td>4500.0</td>
      <td>4320.0</td>
      <td>4320.0</td>
      <td>4460.0</td>
      <td>165758600.0</td>
      <td>4460.000000</td>
    </tr>
    <tr>
      <th>2022-05-25</th>
      <td>4450.0</td>
      <td>4350.0</td>
      <td>4430.0</td>
      <td>4350.0</td>
      <td>138961700.0</td>
      <td>4350.000000</td>
    </tr>
    <tr>
      <th>2022-05-27</th>
      <td>4540.0</td>
      <td>4440.0</td>
      <td>4500.0</td>
      <td>4540.0</td>
      <td>209962100.0</td>
      <td>4540.000000</td>
    </tr>
    <tr>
      <th>2022-05-30</th>
      <td>4540.0</td>
      <td>4390.0</td>
      <td>4540.0</td>
      <td>4430.0</td>
      <td>193724500.0</td>
      <td>4430.000000</td>
    </tr>
    <tr>
      <th>2022-05-31</th>
      <td>4630.0</td>
      <td>4450.0</td>
      <td>4470.0</td>
      <td>4630.0</td>
      <td>379626500.0</td>
      <td>4630.000000</td>
    </tr>
  </tbody>
</table>
<p>3556 rows × 6 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-4ce132c0-08d6-4fbc-be7f-4df3eecf234d')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-4ce132c0-08d6-4fbc-be7f-4df3eecf234d button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-4ce132c0-08d6-4fbc-be7f-4df3eecf234d');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
# Let's visualize the Closing Price

rolling_mean20 = bbri['Close'].rolling(window=20).mean()
rolling_mean50 = bbri['Close'].rolling(window=50).mean()
rolling_mean200 = bbri['Close'].rolling(window=200).mean()

plt.figure(figsize = (16,8))
plt.plot(bbri['Close'], color = 'b', label = 'BBRI Close')
plt.plot(rolling_mean20, color = 'r', linewidth = 2.5, label = 'MA20')
plt.plot(rolling_mean50, color = 'y', linewidth = 2.5, label = 'MA50')
plt.plot(rolling_mean200, color = 'c',linewidth = 2.5, label = 'MA200')

plt.xlabel('Date', fontsize = 15)
plt.ylabel('Closing Price in Rupiah (Rp)', fontsize = 18)
plt.title('Closing Price of BBRI')
plt.legend(loc = 'lower right')
```




    <matplotlib.legend.Legend at 0x7f430d2c2210>




    
![png](/img/posts/bbri/output_6_1.png)
    



Here the visualization of BBRI's closing price since 2008. Based on this graph, it's been a good money when you invest your money into this stock. Just saying. I'm showing you the Moving Average (MA) too withing 20, 50 and 200 days to make it more technical with these indicators.

## 2) Data Pre-processing
Now let's straight to the pre-processing. We have to prepare up like splitting our dataset into Train and Test dataset. We do this by call some libraries from sklearn.


```python
# Don't forget to using 'values' attribute before apply it into Neural Network!

X = bbri.drop('Close', axis = 1).values
y = bbri['Close'].values
```


```python
from sklearn.model_selection import train_test_split
```


```python
# Split the Data into Train and Test dataset with 20% test size.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
```


```python
X_train.shape
```




    (2844, 5)



So, it's already splitted with 80:20. Notice our X_train size is 2844 (80% of the dataset).


```python
# Scale it with MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
```


```python
scaler = MinMaxScaler()
```


```python
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
```

After scaling our X dataset, we should reshape our dataset into proper one. Especially on X dataset, we should turn it into 3-dimensional data before apply it to LSTM NN model or you'll get some advices from the notebook.


```python
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
```


```python
# Reshape it into 3-dimensional before input it into LSTM Model.
# Reshape the data to be 3-dimensional in the form [number of samples, number of time steps, and number of features].

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
```

## 3) Make a LSTM Model
Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies. They were introduced by Hochreiter & Schmidhuber (1997), and were refined and popularized by many people in following work.1 They work tremendously well on a large variety of problems, and are now widely used.

LSTMs are explicitly designed to avoid the long-term dependency problem. Remembering information for long periods of time is practically their default behavior, not something they struggle to learn!

All recurrent neural networks have the form of a chain of repeating modules of neural network. In standard RNNs, this repeating module will have a very simple structure, such as a single tanh layer. 


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
```

We use 5 neurons in input layer because we have 5 features along training our model. We add 1 hidden layer with 5 neurons and also ReLU activation. Finally adding 1 layer for output. We choose Adam optimizer with mean squared error for calculating model's loss.


```python
# just to remember our dataset shape

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
```

    (2844, 5, 1)
    (2844, 1)
    (712, 5, 1)
    (712, 1)
    


```python
model = Sequential()

# input layer
model.add(LSTM(5, return_sequences=True, input_shape=(X_train.shape[1],1)))
model.add(LSTM(5, return_sequences= False))

# hidden layer
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))

# output layer
model.add(Dense(1))

# compiler
model.compile(optimizer='adam', loss = 'mse')
```

I love efficiency, so I add Callbacks feature to the model with EarlyStopping from TensorFlow to avoid overfitting and improve generalization.


```python
from tensorflow.keras.callbacks import EarlyStopping
```


```python
early_stop = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 25)
```

I set epochs to 500 iterations, but at the same time I do believe this model won't iterate more than 500 cause EarlyStopping will save us.


```python
# Let's train our model!

model.fit(x = X_train,
          y = y_train,
          validation_data = (X_test, y_test),
          epochs = 500, 
          callbacks = [early_stop])
```

    Epoch 1/500
    89/89 [==============================] - 7s 21ms/step - loss: 6728794.5000 - val_loss: 7068232.0000
    Epoch 2/500
    89/89 [==============================] - 1s 8ms/step - loss: 6723610.5000 - val_loss: 7057610.0000
    Epoch 3/500
    89/89 [==============================] - 1s 8ms/step - loss: 6707948.0000 - val_loss: 7036904.5000
    Epoch 4/500
    89/89 [==============================] - 1s 9ms/step - loss: 6682745.5000 - val_loss: 7005203.5000
    Epoch 5/500
    89/89 [==============================] - 1s 8ms/step - loss: 6644921.0000 - val_loss: 6959078.0000
    Epoch 6/500
    89/89 [==============================] - 1s 8ms/step - loss: 6591286.5000 - val_loss: 6895181.5000
    Epoch 7/500
    89/89 [==============================] - 1s 8ms/step - loss: 6519325.5000 - val_loss: 6811321.0000
    Epoch 8/500
    89/89 [==============================] - 1s 9ms/step - loss: 6426884.5000 - val_loss: 6705898.5000
    Epoch 9/500
    89/89 [==============================] - 1s 8ms/step - loss: 6312536.0000 - val_loss: 6577160.5000
    Epoch 10/500
    89/89 [==============================] - 1s 8ms/step - loss: 6175521.0000 - val_loss: 6425698.5000
    Epoch 11/500
    89/89 [==============================] - 1s 8ms/step - loss: 6015493.5000 - val_loss: 6250165.0000
    Epoch 12/500
    89/89 [==============================] - 1s 8ms/step - loss: 5833048.0000 - val_loss: 6052452.5000
    Epoch 13/500
    89/89 [==============================] - 1s 8ms/step - loss: 5629385.5000 - val_loss: 5832922.5000
    Epoch 14/500
    89/89 [==============================] - 1s 8ms/step - loss: 5405945.0000 - val_loss: 5595558.0000
    Epoch 15/500
    89/89 [==============================] - 1s 8ms/step - loss: 5165781.5000 - val_loss: 5340338.0000
    Epoch 16/500
    89/89 [==============================] - 1s 8ms/step - loss: 4910532.0000 - val_loss: 5073329.5000
    Epoch 17/500
    89/89 [==============================] - 1s 9ms/step - loss: 4644864.0000 - val_loss: 4794452.5000
    Epoch 18/500
    89/89 [==============================] - 1s 8ms/step - loss: 4371365.0000 - val_loss: 4511833.0000
    Epoch 19/500
    89/89 [==============================] - 1s 8ms/step - loss: 4094801.7500 - val_loss: 4226146.0000
    Epoch 20/500
    89/89 [==============================] - 1s 9ms/step - loss: 3819316.5000 - val_loss: 3943429.7500
    Epoch 21/500
    89/89 [==============================] - 1s 8ms/step - loss: 3548088.2500 - val_loss: 3669539.5000
    Epoch 22/500
    89/89 [==============================] - 1s 9ms/step - loss: 3286434.0000 - val_loss: 3403550.5000
    Epoch 23/500
    89/89 [==============================] - 1s 10ms/step - loss: 3037673.7500 - val_loss: 3152802.2500
    Epoch 24/500
    89/89 [==============================] - 1s 14ms/step - loss: 2804934.2500 - val_loss: 2922020.7500
    Epoch 25/500
    89/89 [==============================] - 1s 15ms/step - loss: 2591608.5000 - val_loss: 2708664.2500
    Epoch 26/500
    89/89 [==============================] - 1s 14ms/step - loss: 2399314.0000 - val_loss: 2518329.2500
    Epoch 27/500
    89/89 [==============================] - 2s 18ms/step - loss: 2230064.0000 - val_loss: 2351074.2500
    Epoch 28/500
    89/89 [==============================] - 2s 17ms/step - loss: 2084102.5000 - val_loss: 2208367.7500
    Epoch 29/500
    89/89 [==============================] - 1s 15ms/step - loss: 1961157.5000 - val_loss: 2089056.8750
    Epoch 30/500
    89/89 [==============================] - 2s 17ms/step - loss: 1860061.6250 - val_loss: 1990759.0000
    Epoch 31/500
    89/89 [==============================] - 1s 14ms/step - loss: 1779148.5000 - val_loss: 1912582.8750
    Epoch 32/500
    89/89 [==============================] - 1s 15ms/step - loss: 1716673.1250 - val_loss: 1850092.3750
    Epoch 33/500
    89/89 [==============================] - 1s 16ms/step - loss: 1669327.7500 - val_loss: 1804724.8750
    Epoch 34/500
    89/89 [==============================] - 1s 15ms/step - loss: 1635036.3750 - val_loss: 1770084.8750
    Epoch 35/500
    89/89 [==============================] - 1s 12ms/step - loss: 1610981.7500 - val_loss: 1746241.6250
    Epoch 36/500
    89/89 [==============================] - 1s 14ms/step - loss: 1594289.1250 - val_loss: 1729051.3750
    Epoch 37/500
    89/89 [==============================] - 1s 16ms/step - loss: 1274190.0000 - val_loss: 1078870.7500
    Epoch 38/500
    89/89 [==============================] - 1s 14ms/step - loss: 876666.5000 - val_loss: 917421.1875
    Epoch 39/500
    89/89 [==============================] - 1s 12ms/step - loss: 738829.5000 - val_loss: 777625.6875
    Epoch 40/500
    89/89 [==============================] - 2s 17ms/step - loss: 620443.3750 - val_loss: 656790.5625
    Epoch 41/500
    89/89 [==============================] - 1s 17ms/step - loss: 518916.4062 - val_loss: 551334.6250
    Epoch 42/500
    89/89 [==============================] - 2s 21ms/step - loss: 432518.0312 - val_loss: 461674.6562
    Epoch 43/500
    89/89 [==============================] - 2s 18ms/step - loss: 359889.3125 - val_loss: 385974.8438
    Epoch 44/500
    89/89 [==============================] - 2s 19ms/step - loss: 299184.0938 - val_loss: 321966.5938
    Epoch 45/500
    89/89 [==============================] - 2s 20ms/step - loss: 248667.4844 - val_loss: 268694.4375
    Epoch 46/500
    89/89 [==============================] - 2s 20ms/step - loss: 205980.6719 - val_loss: 223295.9062
    Epoch 47/500
    89/89 [==============================] - 1s 11ms/step - loss: 170021.3281 - val_loss: 185113.6094
    Epoch 48/500
    89/89 [==============================] - 1s 8ms/step - loss: 139950.8906 - val_loss: 152330.4375
    Epoch 49/500
    89/89 [==============================] - 1s 8ms/step - loss: 114830.4688 - val_loss: 125235.7500
    Epoch 50/500
    89/89 [==============================] - 1s 8ms/step - loss: 94134.4844 - val_loss: 102681.2031
    Epoch 51/500
    89/89 [==============================] - 1s 9ms/step - loss: 77076.0703 - val_loss: 84842.9688
    Epoch 52/500
    89/89 [==============================] - 1s 9ms/step - loss: 63257.7656 - val_loss: 69759.1719
    Epoch 53/500
    89/89 [==============================] - 1s 8ms/step - loss: 51887.4766 - val_loss: 57550.2148
    Epoch 54/500
    89/89 [==============================] - 1s 13ms/step - loss: 42710.1641 - val_loss: 47487.1289
    Epoch 55/500
    89/89 [==============================] - 1s 16ms/step - loss: 35261.4727 - val_loss: 39156.7930
    Epoch 56/500
    89/89 [==============================] - 1s 11ms/step - loss: 29141.7969 - val_loss: 32645.3984
    Epoch 57/500
    89/89 [==============================] - 1s 8ms/step - loss: 24204.2168 - val_loss: 27203.0586
    Epoch 58/500
    89/89 [==============================] - 1s 8ms/step - loss: 20151.5801 - val_loss: 22898.7109
    Epoch 59/500
    89/89 [==============================] - 1s 8ms/step - loss: 16899.8926 - val_loss: 19403.0078
    Epoch 60/500
    89/89 [==============================] - 1s 8ms/step - loss: 14241.0938 - val_loss: 16547.8203
    Epoch 61/500
    89/89 [==============================] - 1s 9ms/step - loss: 12103.2207 - val_loss: 14157.0586
    Epoch 62/500
    89/89 [==============================] - 1s 8ms/step - loss: 10390.1396 - val_loss: 12558.1826
    Epoch 63/500
    89/89 [==============================] - 1s 8ms/step - loss: 8988.3750 - val_loss: 10850.4727
    Epoch 64/500
    89/89 [==============================] - 1s 8ms/step - loss: 7777.8579 - val_loss: 9270.9590
    Epoch 65/500
    89/89 [==============================] - 1s 9ms/step - loss: 6764.6582 - val_loss: 8420.1631
    Epoch 66/500
    89/89 [==============================] - 1s 8ms/step - loss: 5966.5669 - val_loss: 7300.9307
    Epoch 67/500
    89/89 [==============================] - 1s 8ms/step - loss: 5290.8247 - val_loss: 6495.5781
    Epoch 68/500
    89/89 [==============================] - 1s 9ms/step - loss: 4759.8613 - val_loss: 5921.5303
    Epoch 69/500
    89/89 [==============================] - 1s 8ms/step - loss: 4207.7979 - val_loss: 5239.5376
    Epoch 70/500
    89/89 [==============================] - 1s 8ms/step - loss: 3770.0527 - val_loss: 4750.5854
    Epoch 71/500
    89/89 [==============================] - 1s 8ms/step - loss: 3375.0735 - val_loss: 4312.5293
    Epoch 72/500
    89/89 [==============================] - 1s 9ms/step - loss: 3059.5713 - val_loss: 3904.9648
    Epoch 73/500
    89/89 [==============================] - 1s 9ms/step - loss: 2850.0164 - val_loss: 3737.8792
    Epoch 74/500
    89/89 [==============================] - 1s 8ms/step - loss: 2564.4143 - val_loss: 3438.4224
    Epoch 75/500
    89/89 [==============================] - 1s 9ms/step - loss: 2357.3450 - val_loss: 3358.5623
    Epoch 76/500
    89/89 [==============================] - 1s 9ms/step - loss: 2232.6135 - val_loss: 2793.4851
    Epoch 77/500
    89/89 [==============================] - 1s 9ms/step - loss: 1997.9294 - val_loss: 2636.9138
    Epoch 78/500
    89/89 [==============================] - 1s 9ms/step - loss: 1915.8571 - val_loss: 2517.6533
    Epoch 79/500
    89/89 [==============================] - 1s 9ms/step - loss: 1729.1658 - val_loss: 2317.3748
    Epoch 80/500
    89/89 [==============================] - 1s 9ms/step - loss: 1631.3309 - val_loss: 2103.8298
    Epoch 81/500
    89/89 [==============================] - 1s 9ms/step - loss: 1550.3922 - val_loss: 2060.3123
    Epoch 82/500
    89/89 [==============================] - 1s 9ms/step - loss: 1463.8959 - val_loss: 2043.3530
    Epoch 83/500
    89/89 [==============================] - 1s 9ms/step - loss: 1433.6577 - val_loss: 1867.8933
    Epoch 84/500
    89/89 [==============================] - 1s 9ms/step - loss: 1379.9380 - val_loss: 1746.9908
    Epoch 85/500
    89/89 [==============================] - 1s 9ms/step - loss: 1307.4775 - val_loss: 1657.3741
    Epoch 86/500
    89/89 [==============================] - 1s 8ms/step - loss: 1227.0078 - val_loss: 1609.0542
    Epoch 87/500
    89/89 [==============================] - 1s 9ms/step - loss: 1209.1255 - val_loss: 1649.5646
    Epoch 88/500
    89/89 [==============================] - 1s 8ms/step - loss: 1163.1046 - val_loss: 1585.8846
    Epoch 89/500
    89/89 [==============================] - 1s 8ms/step - loss: 1122.1663 - val_loss: 1381.3331
    Epoch 90/500
    89/89 [==============================] - 1s 8ms/step - loss: 1103.4403 - val_loss: 1333.7379
    Epoch 91/500
    89/89 [==============================] - 1s 9ms/step - loss: 1084.5853 - val_loss: 1350.7050
    Epoch 92/500
    89/89 [==============================] - 1s 9ms/step - loss: 1051.3956 - val_loss: 1296.9717
    Epoch 93/500
    89/89 [==============================] - 1s 9ms/step - loss: 1053.0925 - val_loss: 1309.9183
    Epoch 94/500
    89/89 [==============================] - 1s 9ms/step - loss: 1048.1725 - val_loss: 1329.9364
    Epoch 95/500
    89/89 [==============================] - 1s 10ms/step - loss: 995.8461 - val_loss: 1263.0800
    Epoch 96/500
    89/89 [==============================] - 1s 9ms/step - loss: 990.8187 - val_loss: 1250.9036
    Epoch 97/500
    89/89 [==============================] - 1s 9ms/step - loss: 958.6639 - val_loss: 1410.3779
    Epoch 98/500
    89/89 [==============================] - 1s 9ms/step - loss: 997.1367 - val_loss: 1562.8750
    Epoch 99/500
    89/89 [==============================] - 1s 9ms/step - loss: 1018.0589 - val_loss: 1231.3599
    Epoch 100/500
    89/89 [==============================] - 1s 9ms/step - loss: 937.6019 - val_loss: 1076.1615
    Epoch 101/500
    89/89 [==============================] - 1s 8ms/step - loss: 963.8284 - val_loss: 1120.6255
    Epoch 102/500
    89/89 [==============================] - 1s 9ms/step - loss: 946.0831 - val_loss: 1294.1167
    Epoch 103/500
    89/89 [==============================] - 1s 8ms/step - loss: 907.2535 - val_loss: 1006.6993
    Epoch 104/500
    89/89 [==============================] - 1s 9ms/step - loss: 907.2841 - val_loss: 1216.4642
    Epoch 105/500
    89/89 [==============================] - 1s 9ms/step - loss: 916.0087 - val_loss: 1297.0554
    Epoch 106/500
    89/89 [==============================] - 1s 9ms/step - loss: 921.1118 - val_loss: 1310.7106
    Epoch 107/500
    89/89 [==============================] - 1s 10ms/step - loss: 923.8624 - val_loss: 1510.4812
    Epoch 108/500
    89/89 [==============================] - 1s 9ms/step - loss: 954.5092 - val_loss: 1170.1067
    Epoch 109/500
    89/89 [==============================] - 1s 9ms/step - loss: 917.3290 - val_loss: 1012.5935
    Epoch 110/500
    89/89 [==============================] - 1s 9ms/step - loss: 893.7215 - val_loss: 1273.6051
    Epoch 111/500
    89/89 [==============================] - 1s 9ms/step - loss: 895.7330 - val_loss: 1229.0637
    Epoch 112/500
    89/89 [==============================] - 1s 9ms/step - loss: 908.8897 - val_loss: 1031.2739
    Epoch 113/500
    89/89 [==============================] - 1s 9ms/step - loss: 960.9525 - val_loss: 1069.6318
    Epoch 114/500
    89/89 [==============================] - 1s 9ms/step - loss: 872.4641 - val_loss: 1066.6165
    Epoch 115/500
    89/89 [==============================] - 1s 9ms/step - loss: 881.0060 - val_loss: 1067.6473
    Epoch 116/500
    89/89 [==============================] - 1s 9ms/step - loss: 878.1358 - val_loss: 1353.8790
    Epoch 117/500
    89/89 [==============================] - 1s 9ms/step - loss: 895.1171 - val_loss: 1000.1158
    Epoch 118/500
    89/89 [==============================] - 1s 9ms/step - loss: 915.8824 - val_loss: 970.6990
    Epoch 119/500
    89/89 [==============================] - 1s 9ms/step - loss: 904.9184 - val_loss: 1093.6401
    Epoch 120/500
    89/89 [==============================] - 1s 9ms/step - loss: 902.4607 - val_loss: 1033.6764
    Epoch 121/500
    89/89 [==============================] - 1s 9ms/step - loss: 889.4005 - val_loss: 1611.1193
    Epoch 122/500
    89/89 [==============================] - 1s 9ms/step - loss: 892.5366 - val_loss: 984.4285
    Epoch 123/500
    89/89 [==============================] - 1s 9ms/step - loss: 921.8589 - val_loss: 1402.7405
    Epoch 124/500
    89/89 [==============================] - 1s 9ms/step - loss: 875.1534 - val_loss: 1046.4728
    Epoch 125/500
    89/89 [==============================] - 1s 8ms/step - loss: 849.3232 - val_loss: 961.7246
    Epoch 126/500
    89/89 [==============================] - 1s 9ms/step - loss: 886.5376 - val_loss: 1025.4518
    Epoch 127/500
    89/89 [==============================] - 1s 9ms/step - loss: 905.3434 - val_loss: 1285.8262
    Epoch 128/500
    89/89 [==============================] - 1s 9ms/step - loss: 904.4112 - val_loss: 1076.7793
    Epoch 129/500
    89/89 [==============================] - 1s 9ms/step - loss: 893.3470 - val_loss: 1285.9775
    Epoch 130/500
    89/89 [==============================] - 1s 9ms/step - loss: 871.8917 - val_loss: 1979.8333
    Epoch 131/500
    89/89 [==============================] - 1s 9ms/step - loss: 951.7941 - val_loss: 875.3125
    Epoch 132/500
    89/89 [==============================] - 1s 9ms/step - loss: 902.2049 - val_loss: 986.6984
    Epoch 133/500
    89/89 [==============================] - 1s 9ms/step - loss: 887.7725 - val_loss: 887.2075
    Epoch 134/500
    89/89 [==============================] - 1s 8ms/step - loss: 897.8741 - val_loss: 1218.7233
    Epoch 135/500
    89/89 [==============================] - 1s 9ms/step - loss: 910.2280 - val_loss: 873.9462
    Epoch 136/500
    89/89 [==============================] - 1s 9ms/step - loss: 898.3157 - val_loss: 1551.2761
    Epoch 137/500
    89/89 [==============================] - 1s 8ms/step - loss: 902.2271 - val_loss: 1377.0199
    Epoch 138/500
    89/89 [==============================] - 1s 9ms/step - loss: 865.7802 - val_loss: 1155.8024
    Epoch 139/500
    89/89 [==============================] - 1s 9ms/step - loss: 875.7888 - val_loss: 963.3873
    Epoch 140/500
    89/89 [==============================] - 1s 9ms/step - loss: 878.3383 - val_loss: 978.4841
    Epoch 141/500
    89/89 [==============================] - 1s 9ms/step - loss: 857.4650 - val_loss: 988.0334
    Epoch 142/500
    89/89 [==============================] - 1s 8ms/step - loss: 884.1348 - val_loss: 1442.5347
    Epoch 143/500
    89/89 [==============================] - 1s 8ms/step - loss: 852.4373 - val_loss: 1036.8562
    Epoch 144/500
    89/89 [==============================] - 1s 8ms/step - loss: 864.8729 - val_loss: 875.8694
    Epoch 145/500
    89/89 [==============================] - 1s 9ms/step - loss: 885.4371 - val_loss: 1141.1495
    Epoch 146/500
    89/89 [==============================] - 1s 9ms/step - loss: 859.5933 - val_loss: 1095.2089
    Epoch 147/500
    89/89 [==============================] - 1s 9ms/step - loss: 858.5456 - val_loss: 1033.9537
    Epoch 148/500
    89/89 [==============================] - 1s 8ms/step - loss: 887.5352 - val_loss: 1058.0504
    Epoch 149/500
    89/89 [==============================] - 1s 8ms/step - loss: 926.1168 - val_loss: 1259.0009
    Epoch 150/500
    89/89 [==============================] - 1s 9ms/step - loss: 885.6328 - val_loss: 1103.0502
    Epoch 151/500
    89/89 [==============================] - 1s 9ms/step - loss: 831.0637 - val_loss: 1859.7780
    Epoch 152/500
    89/89 [==============================] - 1s 9ms/step - loss: 883.6222 - val_loss: 926.1632
    Epoch 153/500
    89/89 [==============================] - 1s 10ms/step - loss: 865.0244 - val_loss: 883.8705
    Epoch 154/500
    89/89 [==============================] - 1s 10ms/step - loss: 900.2144 - val_loss: 965.1517
    Epoch 155/500
    89/89 [==============================] - 1s 9ms/step - loss: 857.1143 - val_loss: 1058.2616
    Epoch 156/500
    89/89 [==============================] - 1s 9ms/step - loss: 879.2825 - val_loss: 1146.2948
    Epoch 157/500
    89/89 [==============================] - 1s 9ms/step - loss: 852.4259 - val_loss: 1225.2124
    Epoch 158/500
    89/89 [==============================] - 1s 8ms/step - loss: 909.5094 - val_loss: 1384.2289
    Epoch 159/500
    89/89 [==============================] - 1s 9ms/step - loss: 844.4752 - val_loss: 851.6119
    Epoch 160/500
    89/89 [==============================] - 1s 9ms/step - loss: 871.4876 - val_loss: 1137.0043
    Epoch 161/500
    89/89 [==============================] - 1s 9ms/step - loss: 873.1281 - val_loss: 839.7911
    Epoch 162/500
    89/89 [==============================] - 1s 9ms/step - loss: 882.4369 - val_loss: 958.2319
    Epoch 163/500
    89/89 [==============================] - 1s 8ms/step - loss: 897.3118 - val_loss: 976.2214
    Epoch 164/500
    89/89 [==============================] - 1s 9ms/step - loss: 863.2068 - val_loss: 1056.6986
    Epoch 165/500
    89/89 [==============================] - 1s 9ms/step - loss: 900.8514 - val_loss: 1004.8099
    Epoch 166/500
    89/89 [==============================] - 1s 8ms/step - loss: 894.6444 - val_loss: 988.3345
    Epoch 167/500
    89/89 [==============================] - 1s 8ms/step - loss: 889.9005 - val_loss: 1560.9814
    Epoch 168/500
    89/89 [==============================] - 1s 9ms/step - loss: 882.0018 - val_loss: 872.4977
    Epoch 169/500
    89/89 [==============================] - 1s 13ms/step - loss: 887.3175 - val_loss: 1488.1952
    Epoch 170/500
    89/89 [==============================] - 1s 9ms/step - loss: 827.1350 - val_loss: 870.1962
    Epoch 171/500
    89/89 [==============================] - 1s 8ms/step - loss: 857.9397 - val_loss: 934.2194
    Epoch 172/500
    89/89 [==============================] - 1s 9ms/step - loss: 853.0777 - val_loss: 920.4064
    Epoch 173/500
    89/89 [==============================] - 1s 10ms/step - loss: 884.1110 - val_loss: 1187.6647
    Epoch 174/500
    89/89 [==============================] - 1s 13ms/step - loss: 866.8095 - val_loss: 888.5013
    Epoch 175/500
    89/89 [==============================] - 1s 8ms/step - loss: 852.5687 - val_loss: 1314.4453
    Epoch 176/500
    89/89 [==============================] - 1s 9ms/step - loss: 859.4567 - val_loss: 1150.2583
    Epoch 177/500
    89/89 [==============================] - 1s 9ms/step - loss: 848.0142 - val_loss: 1070.6479
    Epoch 178/500
    89/89 [==============================] - 1s 10ms/step - loss: 839.4297 - val_loss: 938.0947
    Epoch 179/500
    89/89 [==============================] - 1s 8ms/step - loss: 865.0428 - val_loss: 1112.5199
    Epoch 180/500
    89/89 [==============================] - 1s 9ms/step - loss: 890.2850 - val_loss: 1593.5288
    Epoch 181/500
    89/89 [==============================] - 1s 9ms/step - loss: 890.6694 - val_loss: 1251.8132
    Epoch 182/500
    89/89 [==============================] - 1s 9ms/step - loss: 865.7590 - val_loss: 1131.9238
    Epoch 183/500
    89/89 [==============================] - 1s 9ms/step - loss: 897.3402 - val_loss: 1227.0776
    Epoch 184/500
    89/89 [==============================] - 1s 9ms/step - loss: 847.9206 - val_loss: 965.4744
    Epoch 185/500
    89/89 [==============================] - 1s 9ms/step - loss: 833.6210 - val_loss: 1284.7535
    Epoch 186/500
    89/89 [==============================] - 1s 8ms/step - loss: 853.7308 - val_loss: 1018.0238
    Epoch 186: early stopping
    




    <keras.callbacks.History at 0x7f42a1213fd0>




```python
# So there are 162 iterations. Let's recap our training loss vs validation loss, and make it to DataFrame.

model_loss = pd.DataFrame(model.history.history)
model_loss
```





  <div id="df-8068862d-7765-4290-ba43-84805aac95a6">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loss</th>
      <th>val_loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6.728794e+06</td>
      <td>7.068232e+06</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6.723610e+06</td>
      <td>7.057610e+06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6.707948e+06</td>
      <td>7.036904e+06</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6.682746e+06</td>
      <td>7.005204e+06</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6.644921e+06</td>
      <td>6.959078e+06</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>181</th>
      <td>8.657590e+02</td>
      <td>1.131924e+03</td>
    </tr>
    <tr>
      <th>182</th>
      <td>8.973402e+02</td>
      <td>1.227078e+03</td>
    </tr>
    <tr>
      <th>183</th>
      <td>8.479206e+02</td>
      <td>9.654744e+02</td>
    </tr>
    <tr>
      <th>184</th>
      <td>8.336210e+02</td>
      <td>1.284754e+03</td>
    </tr>
    <tr>
      <th>185</th>
      <td>8.537308e+02</td>
      <td>1.018024e+03</td>
    </tr>
  </tbody>
</table>
<p>186 rows × 2 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-8068862d-7765-4290-ba43-84805aac95a6')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-8068862d-7765-4290-ba43-84805aac95a6 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-8068862d-7765-4290-ba43-84805aac95a6');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
plt.figure(figsize = (12,6))
model_loss.plot()
plt.xlabel('n of Epochs')
```




    Text(0.5, 0, 'n of Epochs')




    <Figure size 864x432 with 0 Axes>



    
![png](/img/posts/bbri/output_31_2.png)
    


I can't believe this our model get "just right" form along the training. We see that 'training loss' and 'validation loss' walk along the side. It's beautiful graph that you could see.

## 4) Predictions
Training is done, let's get our model to predict and compare it with y_test dataset. We will see the value of RMSE and linear regression graph.


```python
from sklearn.metrics import mean_squared_error, explained_variance_score
```


```python
predictions = model.predict(X_test)
```

    23/23 [==============================] - 1s 3ms/step
    


```python
# MSE
mean_squared_error(y_test, predictions)
```




    1018.0237450375761




```python
# RMSE
np.sqrt(mean_squared_error(y_test, predictions))
```




    31.906484372891605




```python
# Explained variance regression score function

explained_variance_score(y_test, predictions)
# Best possible score is 1.0, lower values are worse (sklearn).
```




    0.9994461431887325



We see that we get the value of RMSE is 31.906. It's very small according to the context of stocks. Look at the 'explained variance score', we get 0.9994. Our model just did a great job. So what if we plot our valid dataset with model's prediction? Could we get some nice line of regression?


```python
plt.figure(figsize=(12,6))
plt.scatter(y_test,predictions)
plt.plot(y_test, y_test, 'r', linewidth = 2.5)
plt.xlabel('y_test')
plt.ylabel('predictions')
plt.title('LSTM Model Prediction Evaluation')
```




    Text(0.5, 1.0, 'LSTM Model Prediction Evaluation')




    
![png](/img/posts/bbri/output_40_1.png)
    


Still can't believe this that our model's predictions just fit with y_test dataset. It shows fit very well with few outliers there. It's okay. Let's try to predict closing price of BBRI on June 2nd.

## 5) Predict with Another Data on June 2nd, 2022
First we get our dataset, then prepare it up into our model's prediction.


```python
# BBRI
bbri2 = web.DataReader('bbri.jk', 'yahoo', '2022-06-01', '2022-06-01')
bbri2
```





  <div id="df-d2389d51-9570-4632-8d9d-3b7d54ab239d">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Adj Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-06-02</th>
      <td>3290</td>
      <td>2950</td>
      <td>2950</td>
      <td>3180</td>
      <td>467460800</td>
      <td>2993.267334</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-d2389d51-9570-4632-8d9d-3b7d54ab239d')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-d2389d51-9570-4632-8d9d-3b7d54ab239d button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-d2389d51-9570-4632-8d9d-3b7d54ab239d');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
bbri2 = bbri2.drop('Close', axis = 1)
```


```python
bbri2 = bbri2.values
```


```python
bbri2 = scaler.transform(bbri2)
```


```python
bbri2 = bbri2.reshape(-1, 5, 1)
```


```python
model.predict(bbri2)
```

    1/1 [==============================] - 0s 21ms/step
    




    array([[3157.858]], dtype=float32)



Our model predicts 3124 of BBRI's closing price on June 2nd, 2022 while the actual close price is 3180. Well, that's not bad though since our model's RMSE is about 31 units and the evaluation just fit between its predictions and y_test dataset.
