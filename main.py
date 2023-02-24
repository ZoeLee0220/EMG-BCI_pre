import numpy as np
import pandas as pd
import keras
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 读取数据集
df = pd.read_csv("data.csv")
X = df.drop("label", axis=1).values
y = df["label"].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 数据预处理（标准化）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# VAE参数
latent_dim = 2
intermediate_dim = 128

# 构建VAE
inputs = Input(shape=(X_train.shape[1],))
h = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(h)
z_log_var = Dense(latent_dim, name='z_log_var')(h)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = np.random.normal(size=z_mean.shape)
    return z_mean + np.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
decoder = Dense(intermediate_dim, activation='relu')
h_decoded = decoder(z)
x_decoded = Dense(X_train.shape[1], activation='sigmoid')(h_decoded)

vae = Model(inputs, x_decoded)

def vae_loss(x, x_decoded):
    xent_loss = keras.losses.binary_crossentropy(x, x_decoded)
    kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))
    return xent_loss + kl_loss

vae.compile(optimizer='adam', loss=vae_loss)

#训练VAE
history = vae.fit(X_train, X_train,
epochs=50,
batch_size=32,
validation_data=(X_test, X_test))

#提取编码器
encoder = Model(inputs, z_mean)

#将训练数据编码为隐变量
X_train_encoded = encoder.predict(X_train)

#将测试数据编码为隐变量
X_test_encoded = encoder.predict(X_test)

#利用隐变量进行分类
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train_encoded, y_train)
y_pred = classifier.predict(X_test_encoded)

#计算分类准确率
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


