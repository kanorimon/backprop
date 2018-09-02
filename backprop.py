import sys
import numpy as np
from collections import OrderedDict

global dbg_flg                                  # ログ出力フラグ 0:テスト 1:計算 2:損失
global dbg_term                                 # ログ出力タイミング
global i                                        # 繰り返し回数

# アフィン変換（計算処理）
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
    
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        
        if dbg_flg == 1 and i % dbg_term == 0:
            print('---アフィン変換(forward)---------------')
            print('入力')
            print(self.x)
            print('重み')
            print(self.W)
            print('バイアス')
            print(self.b)
            print('出力')
            print(out)
            print('')
        
        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)

        if self.x.ndim == 1:
            # 1次元の転置
            self.dW = np.dot(self.x.reshape(-1,1), dout).reshape(-1,1)
        else:
            # 多次元の転置
            self.dW = np.dot(self.x.T, dout)

        self.db = np.sum(dout, axis=0)

        if dbg_flg == 1 and i % dbg_term == 0:
            print('---アフィン変換(backward)--------------')
            print('出力からの損失の逆伝播')
            print(dout)
            print('入力への損失の逆伝播')
            print(dx)
            print('重みの勾配')
            print(self.dW)
            print('バイアスの勾配')
            print(self.db)
            print('')

        return dx

# シグモイド関数（活性化関数）
class Sigmoid:
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        
        if dbg_flg == 1 and i % dbg_term == 0:
            print('---シグモイド関数(forward)-------------')
            print('入力')
            print(x)
            print('出力')
            print(out)
            print('')

        return out
    
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        if dbg_flg == 1 and i % dbg_term == 0:
            print('---シグモイド関数(backward)------------')
            print('出力からの損失の逆伝播')
            print(dout)
            print('入力への損失の逆伝播')
            print(dx)
            print('')

        return dx
        
# 恒等関数（出力関数）
class OutWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
    
    def forward(self, x, t):
        self.t = t
        self.y = x
        self.loss = mean_squared_error(self.y, self.t)
        
        if dbg_flg == 1 and i % dbg_term == 0:
            print('---二乗和誤差(forward)-----------------')
            print('入力')
            print(x)
            print('教師データ')
            print(t)
            print('損失')
            print(self.loss)
            print('')

        return self.loss
    
    def backward(self):
        dx = self.y - self.t
        
        if dbg_flg == 1 and i % dbg_term == 0:
            print('---二乗和誤差(backward)----------------')
            print('入力')
            print(self.y)
            print('教師データ')
            print(self.t)
            print('入力への損失の逆伝播')
            print(dx)
            print('')
        
        return dx

# 二乗和誤差（損失関数）
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2) / batch_size

# 2層ネットワーク
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):

        # 重み/バイアスの初期化
        self.params = {}
        
        if dbg_flg == 1 and i % dbg_term == 0:
            print('***重み/バイアスの初期設定************************')

        # Affine1の重み/バイアス
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)

        if dbg_flg == 1 and i % dbg_term == 0:
            print('---1層目--------------------------------')
            print('重み')
            print(self.params['W1'])
            print('バイアス')
            print(self.params['b1'])
            print('')

        # Affine2の重み/バイアス
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
        if dbg_flg == 1 and i % dbg_term == 0:
            print('---2層目--------------------------------')
            print('重み')
            print(self.params['W2'])
            print('バイアス')
            print(self.params['b2'])
            print('')

        # レイヤの生成
        self.layers = OrderedDict()
        
        # Affine1
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        #Sigmoid
        self.layers['Sigmoid1'] = Sigmoid()
        # Affine2
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        # Output
        self.lastLayer = OutWithLoss()

    # アフィン変換,シグモイド関数による順伝播 x:入力データ
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    # 損失関数による損失の算出 x:入力データ t:教師データ
    def loss(self, x, t):
        y = self.predict(x)
        self.loss_mem = self.lastLayer.forward(y, t)
        return self.loss_mem

    # バックプロパゲーションによる勾配の算出 x:入力データ t:教師データ
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = self.lastLayer.backward()
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 勾配の設定
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        return grads

# ------------------------------------------------

# ログ出力の設定
dbg_flg = int(sys.argv[1])                      # ログ出力フラグ 0:テスト 1:計算 2:損失
dbg_term = int(sys.argv[2])                     # ログ出力タイミング

# 訓練データの設定
x_train = np.array([[0,0],[0,1],[1,0],[1,1]])   # 入力データ
t_train = np.array([[0],[1],[1],[0]])           # 教師データ

# 訓練パラメータの設定
iters_num = 3000                                # 繰り返し回数
train_size = x_train.shape[0]                   # 入力データのパターン数
batch_size = 10                                 # バッチ実行数
learning_rate = 0.1                             # 学習係数

# 繰り返し回数の初期設定
i = 0                                           # 繰り返し回数

# ネットワークの作成
network = TwoLayerNet(input_size=2, hidden_size=3, output_size=1)

# バックプロパゲーションの実行
for i in range(iters_num):

    if dbg_flg == 1 and i % dbg_term == 0:
        print('***{0}回目*****************************************'.format(i))

    # ミニバッチ
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # バックプロパゲーションによる勾配の算出
    grad = network.gradient(x_batch, t_batch)

    # パラメータの更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    if dbg_flg == 2 and i % dbg_term == 0:
        print('繰り返し回数={0} 損失={1}'.format(i,network.loss_mem))

    if dbg_flg == 1 and i % dbg_term == 0:
        print('---パラメータの更新------------------------------')
        print('---1層目-------------------------------')
        print('重み')
        print(network.params['W1'])
        print('バイアス')
        print(network.params['b1'])
        print('')

        print('---2層目-------------------------------')
        print('重み')
        print(network.params['W2'])
        print('バイアス')
        print(network.params['b2'])
        print('')

# 検証
x_test = np.array([[0,0],[0,1],[1,0],[1,1]])

if dbg_flg < 2:
    print('***テスト****************************************')
    print('[0,0]={0}'.format(network.predict([0,0])))
    print('[0,1]={0}'.format(network.predict([0,1])))
    print('[1,0]={0}'.format(network.predict([1,0])))
    print('[1,1]={0}'.format(network.predict([1,1])))
    print('')

