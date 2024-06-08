import re
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def is_uchar(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    if uchar >= u'\u0030' and uchar <= u'\u0039':
        return True
    if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
        return True
    if uchar in ('，', '。', '：', '？', '“', '”', '！', '；', '、', '《', '》', '——'):
        return True
    return False

# 读取和处理文本
with open(r'C:\Users\HP\PycharmProjects\NLP_homework\金庸-语料库\白马啸西风.txt', encoding='gbk', errors='ignore') as f:
    data = f.readlines()

pattern = re.compile(r'\(.*\)')
data = [pattern.sub('', lines) for lines in data]
data = [line.replace('……', '。') for line in data if len(line) > 1]
data = ''.join(data)
data = [char for char in data if is_uchar(char)]
data = ''.join(data)

# 构建词汇表
vocab = list(set(data))
char2id = {c: i for i, c in enumerate(vocab)}
id2char = {i: c for i, c in enumerate(vocab)}
numdata = [char2id[char] for char in data]

# 数据生成器
def data_generator(data, batch_size, time_steps):
    num_batches = len(data) // (batch_size * time_steps)
    data = data[:num_batches * batch_size * time_steps]
    data = np.array(data).reshape((batch_size, -1))
    while True:
        for i in range(0, data.shape[1], time_steps):
            x = data[:, i:i + time_steps]
            y = np.roll(x, -1, axis=1)
            yield x, y

# RNNModel 类定义
class RNNModel(tf.keras.Model):
    def __init__(self, hidden_size, hidden_layers, vocab_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.vocab_size = vocab_size

        self.embedding = tf.keras.layers.Embedding(vocab_size, hidden_size)
        self.lstm_layers = [tf.keras.layers.LSTM(hidden_size, return_sequences=True, return_state=True) for _ in
                            range(hidden_layers)]
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = self.embedding(inputs)
        new_states = []
        for i in range(self.hidden_layers):
            x, state_h, state_c = self.lstm_layers[i](x, initial_state=states[i] if states else None, training=training)
            new_states.append([state_h, state_c])
        x = self.dense(x)
        if return_state:
            return x, new_states
        else:
            return x

hidden_size = 256  # 减少隐藏单元数量
hidden_layers = 3  # 减少隐藏层数量
vocab_size = len(vocab)
batch_size = 32  # 增加批次大小
time_steps = 50  # 减少时间步长
epochs = 100  # 减少训练轮次
learning_rate = 0.01  # 降低学习率

model = RNNModel(hidden_size, hidden_layers, vocab_size)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

train_data = data_generator(numdata, batch_size, time_steps)

# 定义回调函数
class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))

history = LossHistory()

# 训练模型
model.fit(train_data, epochs=epochs, steps_per_epoch=len(numdata) // (batch_size * time_steps), callbacks=[history])

# 绘制loss曲线
plt.plot(history.losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('seq2seq_model Training Loss')
plt.show()

# 文本生成函数
def generate_text(model, start_string, num_generate=100):
    input_eval = [char2id[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []
    states = None

    for i in range(num_generate):
        predictions, states = model(input_eval, states=states, return_state=True)
        predictions = tf.squeeze(predictions, 0)
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(id2char[predicted_id])

    return start_string + ''.join(text_generated)


# 生成文本示例
print(generate_text(model, start_string="约莫过了半个时辰，李文秀突然闻到一阵焦臭，跟著便咳嗽起来。华辉"))
