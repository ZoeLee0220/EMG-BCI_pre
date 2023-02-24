import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K

# 导入数据
X = np.load("electromyogram_data.npy")
y = np.load("labels.npy")

# 将数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义变分自动编码器模型
inputs = Input(shape=(X.shape[1],))

# 编码器
z_mean = Dense(128, activation='relu')(inputs)
z_log_var = Dense(128, activation='relu')(inputs)

# 重构损失
def vae_loss(x, x_decoded_mean):
    xent_loss = K.sum(K.binary_crossentropy(x, x_decoded_mean), axis=-1)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss

# 解码器
latent_inputs = Input(shape=(128,))
x = Dense(128, activation='relu')(latent_inputs)
outputs = Dense(X.shape[1], activation='sigmoid')(x)

# 构建解码器模型
decoder = Model(latent_inputs, outputs)

# 将编码器与解码器拼接起来
z = Lambda(lambda x: x[0] + x[1] * K.random_normal(shape=K.shape(x[0])))([z_mean, K.exp(0.5 * z_log_var)])
x_decoded_mean = decoder(z)

# 构建整个 VAE 模型
vae = Model(inputs, x_decoded_mean)
vae.compile(optimizer='adam', loss=vae_loss)

# 训练 VAE
history = vae.fit(X_train, X_train,
                  epochs=50,
                  batch_size=32,
                  validation_data=(X_test, X_test))

# 使用编码器提取特征
X_train_encoded = vae.predict(X_train, batch_size=32)
X_test_encoded = vae.predict(X_test, batch_size=32)

# 训练分类器
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train_encoded, y_train)

# 预测结果
y_pred = clf.predict(X_test_encoded)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)