# 라이브러리를 불러온다
import sys, os
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)

import numpy as np
try: 
    import cupy as cp
    is_cupy_available = True # 학습속도를 위해 GPU를 사용할 수 있는 환경이면 사용한다.
    print('CuPy is available. Using CuPy for all computations.') 
except:
    is_cupy_available = False # CPU로 학습시킬 경우 매우 오래걸린다
    print('CuPy is not available. Switching to NumPy.') 
    
import pickle as pkl
from tqdm import tqdm
from transformer.modules import Encoder
from transformer.modules import Decoder
from transformer.optimizers import Adam, Nadam, Momentum, RMSProp, SGD, Noam
from transformer.losses import CrossEntropy
from transformer.prepare_data import DataPreparator
import matplotlib.pyplot as plt


# DEFINE 
DATA_TYPE = np.float32  # 데이터 타입을 정의한다 (float32)
BATCH_SIZE = 32         # 배치 사이즈를 정의한다. 학습시 배치사이즈의 평균에 대한 로스로 가중치를 업데이트함

PAD_TOKEN = '<pad>'     # Padding 토큰을 정의한다
SOS_TOKEN = '<sos>'     # Start of Sentence 토큰을 정의한다
EOS_TOKEN = '<eos>'     # End of Sentence 토큰을 정의한다
UNK_TOKEN = '<unk>'     # Unknown 토큰을 정의한다

# 토큰에 대응되는 인덱스
PAD_INDEX = 0           # PAD_TOKEN 의 인덱스를 정의한다
SOS_INDEX = 1           # SOS_TOKEN 의 인덱스를 정의한다
EOS_INDEX = 2           # EOS_TOKEN 의 인덱스를 정의한다
UNK_INDEX = 3           # UNK_INDEX 의 인덱스를 정의한다

tokens  = (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN)   # 토큰 리스트 튜플
indexes = (PAD_INDEX, SOS_INDEX, EOS_INDEX, UNK_INDEX)   # 인덱스 리스트 튜플

data_preparator = DataPreparator(tokens, indexes)        # 데이터를 준비한다. DataPreparator 클래스로부터 인스턴스를 생성

# 데이터를 학습 , 검증, 테스트 셋으로 나눈다. prepare_data 메서드는 학습,테스트,검증 세트에 대한 소스와 타겟쌍의 튜플을 반환한다.
train_data, test_data, val_data = data_preparator.prepare_data( 
                    path = '/home/gyeongseoplee/np-transformer/numpy-transformer/dataset/', 
                    batch_size = BATCH_SIZE, 
                    min_freq = 2)

source, target = train_data 

train_data_vocabs = data_preparator.get_vocabs() # get_vocabs 메서드를 호출하여 전체 단어 사전을 가져온다


# 모델 정의
class Seq2Seq():   # Sequence to Sequence 모델의 클래스를 선언한다.
    def __init__(self, encoder, decoder, pad_idx) -> None: # 초기화.
        self.encoder = encoder                 # 인자로 받은 인코더로 인스턴스의 인코더를 초기화한다
        self.decoder = decoder                 # 인자로 받은 디코더로 인스턴스의 디코더를 초기화 한다
        self.pad_idx = pad_idx                 # 인자로 받은 패딩 인덱스로 인스턴스의 패딩 인덱스를 초기화한다

        self.optimizer = Adam()                # 옵티마이저는 기본값으로 Adam으로 설정한다
        self.loss_function = CrossEntropy()    # 손실으로는 CrossEntropy를 사용한다. 
        ####### 함수명은 CrossEntropy이나, 실제로는 멀티 클래스를 분류하기 위한 Categorical Cross Entropy로스이다. 함수의 정의 부분에는 멀티 클래스 분류에 사용하는 NLL Loss를 그대로 사용하고 있다.

    def set_optimizer(self):                   # 인코더와 디코더의 옵티바이저를 설정하는 메소드  
        encoder.set_optimizer(self.optimizer)  
        decoder.set_optimizer(self.optimizer)   

    def compile(self, optimizer, loss_function):  # 모델을 컴파일 하는 메소드. 옵티마이저와 손실함수를 인자로 받는다.
        self.optimizer = optimizer                
        self.loss_function = loss_function        
        

    def load(self, path):                                  # 사전학습된 모델을 불러오는 메소드              
        pickle_encoder = open(f'{path}/encoder.pkl', 'rb')      
        pickle_decoder = open(f'{path}/decoder.pkl', 'rb')      

        self.encoder = pkl.load(pickle_encoder)                 
        self.decoder = pkl.load(pickle_decoder)                 

        pickle_encoder.close()                                 
        pickle_decoder.close()                                  

        print(f'Loaded from "{path}"') 

    def save(self, path):                                  # 모델을 저장하는 메소드
        if not os.path.exists(path):
            os.makedirs(path)

        pickle_encoder = open(f'{path}/encoder.pkl', 'wb')
        pickle_decoder = open(f'{path}/decoder.pkl', 'wb')

        pkl.dump(self.encoder, pickle_encoder)
        pkl.dump(self.decoder, pickle_decoder)

        pickle_encoder.close()
        pickle_decoder.close()
        
        print(f'Saved to "{path}"')

    def get_pad_mask(self, x):                          # 패딩 마스크를 생성하는 메소드                    
        #x: (batch_size, seq_len)
        return (x != self.pad_idx).astype(int)[:, np.newaxis, :]

    def get_sub_mask(self, x):                          # 서브 시퀀스 마스크를 생성함
        #x: (batch_size, seq_len)
        seq_len = x.shape[1]
        subsequent_mask = np.triu(np.ones((seq_len, seq_len)), k = 1).astype(int)
        subsequent_mask = np.logical_not(subsequent_mask)
        return subsequent_mask

    def forward(self, src, trg, training):                          # 전방 전파 메소드
        src, trg = src.astype(DATA_TYPE), trg.astype(DATA_TYPE)     # 소스와 타겟의 데이터 타입을 변환한다. DATA_TYPE (np.float32 참조)
        #src: (batch_size, source_seq_len)
        #tgt: (batch_size, target_seq_len)

        # src_mask: (batch_size, 1, seq_len)
        # tgt_mask: (batch_size, seq_len, seq_len)
        src_mask = self.get_pad_mask(src)

        trg_mask = self.get_pad_mask(trg) & self.get_sub_mask(trg)

        enc_src = self.encoder.forward(src, src_mask, training)

        out, attention = self.decoder.forward(trg, trg_mask, enc_src, src_mask, training)
        # output: (batch_size, target_seq_len, vocab_size)
        # attn: (batch_size, heads_num, target_seq_len, source_seq_len)
        return out, attention

    def backward(self, error):                                      # 역전파 메소드
        error = self.decoder.backward(error)
        error = self.encoder.backward(self.decoder.encoder_error)

    def update_weights(self):                                       # 가중치를 업데이트 하는 메소드
        self.encoder.update_weights()
        self.decoder.update_weights()

    def _train(self, source, target, epoch, epochs):                # 소스와 타겟 데이터를 받아 학습을 하는 메소드
        loss_history = []                                           # 배치별 로그를 위해 빈 리스트 생성
        
        tqdm_range = tqdm(enumerate(zip(source, target)), total = len(source))      # 소스와 타겟을 zip, 하여 enumerate로 인덱스를 가져옴. 진행상태를 tqdm으로 시각화함
        for batch_num, (source_batch, target_batch) in tqdm_range:                  # 각 배치에서 소스 데이터와 타겟 데이터를 가지고옴
            
            output, attention = self.forward(source_batch, target_batch[:,:-1], training = True)  # forward 메서드를 호출한다. 현재 배치의 추론 결과와 attention을 계산함. EOS 토큰은 제외한다.
            
            _output = output.reshape(output.shape[0] * output.shape[1], output.shape[2])          # 출력 결과를 flatten하여 손실함수에 전달할 수 있도록 함

            loss_history.append(self.loss_function.loss(_output, target_batch[:, 1:].astype(np.int32).flatten()).mean())#[:, np.newaxis] #Loss 히스토리를 기록한다
            error = self.loss_function.derivative(_output, target_batch[:, 1:].astype(np.int32).flatten())#[:, np.newaxis]               # 손실함수의 경사도를 계산, 역전파에 사용할 error를 계산


            self.backward(error.reshape(output.shape)) # 역전파 매소드를 호출하여 가중치와 파라미터의 Delta를 계산함
            self.update_weights()                      # 계산된 Delta에 따라 가중치를 업데이트함

            tqdm_range.set_description(
                    f"training | loss: {loss_history[-1]:.7f} | perplexity: {np.exp(loss_history[-1]):.7f} | epoch {epoch + 1}/{epochs}" #loss: {loss:.4f}
                )   #진행 상태를 표시 (자세히 분석하지 않음)

            if batch_num == (len(source) - 1):
                if is_cupy_available:    # GPU 사용 가능 여부에 따라 로스를 numpy 혹은 cupy로 업데이트
                    epoch_loss = cp.mean(cp.array(loss_history))
                else:
                    epoch_loss = np.mean(loss_history)

                tqdm_range.set_description(
                        f"training | avg loss: {epoch_loss:.7f} | avg perplexity: {np.exp(epoch_loss):.7f} | epoch {epoch + 1}/{epochs}"
                ) #진행 상태를 표시

        return epoch_loss.get() if is_cupy_available else epoch_loss #loss_history의 평균 손실값을 반환함


    def _evaluate(self, source, target):  # 모델을 평가하는 메소드
        loss_history = []                 # 로스 히스토리 로깅을 위한 리스트 생성

        tqdm_range = tqdm(enumerate(zip(source, target)), total = len(source))    # 위 학습 과정과 동일함. 소스와 타겟을 묶어 enumerate, 진행상황에 대한 시각화
        for batch_num, (source_batch, target_batch) in tqdm_range:                # 각 배치 데이터를 가져와 평가
            
            output, attention = self.forward(source_batch, target_batch[:,:-1], training = False) # forward 메서드를 호출해 현재 배치의 추론 결과와 어텐션을 계산
            
            _output = output.reshape(output.shape[0] * output.shape[1], output.shape[2])          # 출력결과를 flatten 

            loss_history.append(self.loss_function.loss(_output, target_batch[:, 1:].astype(np.int32).flatten()).mean()) # 손실함수를 호출하여 현재 배치의 손실값을 계산 및 기록. SOS 토큰을 제외한 데이터
            
            tqdm_range.set_description(
                    f"testing  | loss: {loss_history[-1]:.7f} | perplexity: {np.exp(loss_history[-1]):.7f}"
                ) # 신행 상태 표시

            if batch_num == (len(source) - 1):
                if is_cupy_available:
                    epoch_loss = cp.mean(cp.array(loss_history))
                else:
                    epoch_loss = np.mean(loss_history)

                tqdm_range.set_description(
                        f"testing  | avg loss: {epoch_loss:.7f} | avg perplexity: {np.exp(epoch_loss):.7f}"
                ) # 진행 상태 표시

        return epoch_loss.get() if is_cupy_available else epoch_loss # epoch_loss의 평균 손실값을 반환함


    def fit(self, train_data, val_data, epochs, save_every_epochs, save_path = None, validation_check = False): #전체 학습 과정을 실행하는 메서드
        # 학습데이터, 검증데이터, epochs, 모델 저장 주가, 저장 경로를 인자로 받는다.
        self.set_optimizer() # 옵티마이저를 셋업한다.

        best_val_loss = float('inf') # 최적의 검증 로스를 선언한다. 초기값으로 무한대의 값을 지정한다.
        
        #학습과 검증 로스를 로깅하기 위한 리스트
        train_loss_history = []
        val_loss_history = []
        # 학습과 검증 데이터를 split
        train_source, train_target = train_data
        val_source, val_target = val_data

        for epoch in range(epochs):   # 지정된 epoch 만큼 반복하여 학습 진행

            train_loss_history.append(self._train(train_source, train_target, epoch, epochs)) #_train 메소드 호출. 학습 데이터에 대한 손실을 계산하고 기록함
            val_loss_history.append(self._evaluate(val_source, val_target))                   #_evaluate 메소드를 호출하여 검증 데이터에 대한 손실을 계산하고 기록함


            if (save_path is not None) and ((epoch + 1) % save_every_epochs == 0):      # 모델의 저장 조건을 확인함. 경로가 None이 아니어야 함.
                if validation_check == False:
                    self.save(save_path + f'/{epoch + 1}')
                else:
                    if val_loss_history[-1] < best_val_loss:     # 검증 손실을 기준으로 모델의 성능이 개선되었을 때만 저장함.
                        best_val_loss = val_loss_history[-1]
                        
                        self.save(save_path + f'/{epoch + 1}')
                    else:
                        print(f'Current validation loss is higher than previous; Not saved')
                
        return train_loss_history, val_loss_history # 학습과 검증 로그 리스트 전체를 반환함.




    def predict(self, sentence, vocabs, max_length = 50): # 새로운 입력 문장을 기반으로 감정평가를 하는 메소드.
        # 입력으로는 문장, vocabs는 사전, max_length는 최대 생성 길이

        src_inds = [vocabs[0][word] if word in vocabs[0] else UNK_INDEX for word in sentence] # 입력 문장의 각 단어를 단어 사전에서 인덱스로 변환함
        src_inds = [SOS_INDEX] + src_inds + [EOS_INDEX]                                       # SOS와 EOS 토큰 추가함
        
        src = np.asarray(src_inds).reshape(1, -1)                                             # 모델의 차원과 일치시키기 위해 배치 차원 추가
        src_mask =  self.get_pad_mask(src)                                                    # 입력 데이터에 대한 패딩 마스크 생성

        enc_src = self.encoder.forward(src, src_mask, training = False)                       # 인코더를 통해 입력 데이터의 인코딩 결과 계산

        trg_inds = [SOS_INDEX]                                                                # SOS 토큰으로 초기화된 타겟 문장

        for _ in range(max_length):                                                           #최대 길이까지 반복하면서 타겟 문장 생성
            trg = np.asarray(trg_inds).reshape(1, -1)
            trg_mask = self.get_pad_mask(trg) & self.get_sub_mask(trg)
            # 현재 생성된 타겟 문장에 대해 마스크 생성.

            out, attention = self.decoder.forward(trg, trg_mask, enc_src, src_mask, training = False) # 디코더를호출하여 다음 단어를 예측
            
            trg_indx = out.argmax(axis=-1)[:, -1].item() # 예측된 단어의 인덱스(최대 확률값) 을 선택한다.
            trg_inds.append(trg_indx)                    # 생성된 단어를 타겟 문장에 추가한다.

            if trg_indx == EOS_INDEX or len(trg_inds) >= max_length: # EOS 토큰을 생성하거나 최대길이에 도달하면 생성을 중단함.
                break
        
        reversed_vocab = dict((v,k) for k,v in vocabs[1].items())                           # 디코더의 인덱스를 단어로 변환하기 위해 단어 사전을 역변환
        decoded_sentence = [reversed_vocab[indx] if indx in reversed_vocab else UNK_TOKEN for indx in trg_inds]   # 생성된 문장을 단어로 변환

        return decoded_sentence[1:], attention[0]    # SOS 를 제외한 생성된 문장과 attention을 반환함



# 하이퍼파라미터 정의
INPUT_DIM = len(train_data_vocabs[0])   #인코더의 차원을 설정한다
OUTPUT_DIM = len(train_data_vocabs[1])  # 출력 차원을 설정한다
HID_DIM = 256  #512 in original paper. 은닉차원의 갯수를 정의한다
ENC_LAYERS = 3 #6 in original paper 인코더 레이어 수를 정의한다
DEC_LAYERS = 3 #6 in original paper 디코더 레이어를 정의한다
ENC_HEADS = 8  # 멀티헤드 어텐션에 사용되는 헤드 수
DEC_HEADS = 8  # 
FF_SIZE = 512  #2048 in original paper Feedforward 네트워크의 차원을 정의함
ENC_DROPOUT = 0.1 # 드랍아웃까지 넣어놨네 굳이 왜넣은건지 모르겠지만 과적합을 방지하기 위해 넣은듯하다.
DEC_DROPOUT = 0.1 # 드랍아웃까지 넣어놨네 

MAX_LEN = 5000 # 최대 시퀀스 길이를 설정한다.


encoder = Encoder(INPUT_DIM, ENC_HEADS, ENC_LAYERS, HID_DIM, FF_SIZE, ENC_DROPOUT, MAX_LEN, DATA_TYPE)  #Encoder 클래스로부터 encoder 인스턴스를 생성함
decoder = Decoder(OUTPUT_DIM, DEC_HEADS, DEC_LAYERS, HID_DIM, FF_SIZE, DEC_DROPOUT, MAX_LEN, DATA_TYPE) #Dncoder 클래스로부터 decoder 인스턴스를 생성함

model = Seq2Seq(encoder, decoder, PAD_INDEX)   #정의된 모델로부터 인스턴스를 생성한다. 

try:  # 사전학습된 모델을 불러온다. 감정평가시에만 불러와야 하며, 학습시에는 사전학습된 모델을 불러오지 않는다.
    model.load("/home/gyeongseoplee/np-transformer/numpy-transformer/transformer/saved models/seq2seq_model/10/")
except: # 사전학습된 모델을 불러올 수 없을 경우는 오류 메세지를 출력함
    print("Can't load saved model state")


model.compile( #모델을 컴파일한다. 옵티마이저와 로스 함수를 설정한다.
                optimizer = Noam(
                                Adam(alpha = 1e-4, beta = 0.9, beta2 = 0.98, epsilon = 1e-9), #NOTE: alpha doesn`t matter for Noam scheduler
                                model_dim = HID_DIM,
                                scale_factor = 2,
                                warmup_steps = 4000
                            ) 
                , loss_function = CrossEntropy(ignore_index=PAD_INDEX) 
            ) #옵티마이저로는 ADAM, Loss로는 CategoricalCrossEntropy를 사용한다.

train_loss_history, val_loss_history = None, None
train_loss_history, val_loss_history = model.fit(train_data, val_data, epochs = 10, save_every_epochs = 5, save_path = "saved models/seq2seq_model", validation_check = True)# "saved models/seq2seq_model"
# 현재는 학습을 완료한 후 감정평가를 진행하므로 model.fit을 주석으로 막아놨다. 실제 학습을 할때는 주석을 해제해야 한다.



# 학습과 검증의 히스토리를 시각화 plot 하기 위한 함수.
def plot_loss_history(train_loss_history, val_loss_history):
    plt.plot(train_loss_history)
    plt.plot(val_loss_history)
    plt.title('Loss history')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("loss_history.png")
    print(f"Loss history plot saved as loss_history.png")
    plt.show()
    plt.close(fig)
        
if train_loss_history is not None and val_loss_history is not None:
    print("plot loss history")
    plot_loss_history(train_loss_history, val_loss_history)
else :
    print("loss history is empty")



_, _, val_data = data_preparator.import_multi30k_dataset(path = "/home/gyeongseoplee/np-transformer/numpy-transformer/dataset/") # 검증 데이터셋을 불러온다.
val_data = data_preparator.clear_dataset(val_data)[0] # validation dataset에서 첫번째 요소만 추출하여 정리한다
sentences_num = 30 # 검증할 문장 수를 정의한다.

random_indices = np.random.randint(0, len(val_data), sentences_num) # 검증 데이터셋에서 검증할 문장의 수만큼 랜덤으로 샘플링함.
sentences_selection = [val_data[i] for i in random_indices]

#Translate sentences from validation set
for i, example in enumerate(sentences_selection): # 검증 데이터의 문장을 모델에 입력하고 번역 결과를 출력한다
    print(f"\nExample №{i + 1}")
    print(f"Input sentence: { ' '.join(example['en'])}")
    print(f"Decoded sentence: {' '.join(model.predict(example['en'], train_data_vocabs)[0])}")
    print(f"Target sentence: {' '.join(example['de'])}")


'''
# Attention, 즉 어텐션 맵을 시각화한다.
def plot_attention(sentence, translation, attention, heads_num = 8, rows_num = 2, cols_num = 4):
    
    assert rows_num * cols_num == heads_num
    
    sentence = [SOS_TOKEN] + [word.lower() for word in sentence] + [EOS_TOKEN]

    fig = plt.figure(figsize = (15, 25))
    
    for h in range(heads_num):
        
        ax = fig.add_subplot(rows_num, cols_num, h + 1)
        ax.set_xlabel(f'Head {h + 1}')
        
        if is_cupy_available:
            ax.matshow(cp.asnumpy(attention[h]), cmap = 'inferno')
        else:
            ax.matshow(attention[h], cmap = 'inferno')

        ax.tick_params(labelsize = 7)

        ax.set_xticks(range(len(sentence)))
        ax.set_yticks(range(len(translation)))

        ax.set_xticklabels(sentence, rotation=90)
        ax.set_yticklabels(translation)
        
    # plot을 파일로 저장한다
    plt.savefig("attention_plot.png")
    print("Attention plot saved as 'attention_plot.png'")
    plt.close(fig)
    plt.show()
'''

def plot_attention(sentence, translation, attention, heads_num=8, rows_num=2, cols_num=4):
    """
    어텐션 맵을 시각화하고 저장합니다.
    """
    assert rows_num * cols_num == heads_num

    sentence = [SOS_TOKEN] + [word.lower() for word in sentence] + [EOS_TOKEN]

    fig = plt.figure(figsize=(15, 25))

    for h in range(heads_num):
        ax = fig.add_subplot(rows_num, cols_num, h + 1)
        ax.set_xlabel(f'Head {h + 1}')
        
        if is_cupy_available:
            ax.matshow(cp.asnumpy(attention[h]), cmap='inferno')
        else:
            ax.matshow(attention[h], cmap='inferno')

        ax.tick_params(labelsize=7)

        # 입력 문장과 출력 문장(translation)을 레이블로 설정
        ax.set_xticks(range(len(sentence)))
        ax.set_yticks(range(len(translation)))

        ax.set_xticklabels(sentence, rotation=90)
        ax.set_yticklabels(translation)

    # 플롯 저장
    plt.savefig("attention_plot.png")
    print("Attention plot saved as 'attention_plot.png'")
    plt.show()
    plt.close(fig)

# 감정평가 및 어텐션 맵을 시각화함.
sentence = sentences_selection[0]['en'] #['a', 'trendy', 'girl', 'talking', 'on', 'her', 'cellphone', 'while', 'gliding', 'slowly', 'down', 'the', 'street']
print(f"\nInput sentence: {sentence}")
decoded_sentence, attention =  model.predict(sentence, train_data_vocabs)
print(f"Decoded sentence: {decoded_sentence}")

plot_attention(sentence, decoded_sentence, attention)