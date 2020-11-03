import numpy as np
import librosa
import librosa.display
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from time import time
import matplotlib.pyplot as plt

from bss_cnn import BSS_CNN
from musdb_generator import generator

SR = 22050  # Частота дискретизации, с которой работает сеть
WIN = 1024  # Размер окна дискретного преобразования Фурье
HOP = 256    # Размер шага дискретного преобразования Фурье
TRGT = 'vocals' # Целевой источник, выделению которого будет обучатсья сеть
SMPL = 27   # Длина отрезков спектрограмм, подающихся на вход сети

BS = 64      # Размер мини-батча (пакета)
TIB = 16     # Количество разных музыкльных композиций в батче

EPOCHS = 3   # Количесво эпох обучения
SPE = 6500   # Количество шагов за эпоху (25% тренировочного набора)

print('<----------[INFO] batch generators creation...')
train_gen = generator(SPE, 'train', 'train', TRGT, BS, TIB, SR, WIN, HOP, SMPL)
valid_gen = generator(SPE, 'train', 'valid', TRGT, BS, TIB, SR, WIN, HOP, SMPL)

print('<----------[INFO] creation and compile network...')
model = BSS_CNN.define(freq_bins = int(WIN/2+1), length = SMPL)

print('<----------[INFO] training network...')
t0 = time()
H = model.fit_generator(generator=train_gen,
                        steps_per_epoch = SPE,
                        epochs = EPOCHS,
                        callbacks = [ ModelCheckpoint('output/model.hdf5', monitor='val_loss', save_best_only=True, save_weights_only=False, mode='auto') ],
                        validation_data = valid_gen,
                        validation_steps = SPE,
                        max_queue_size = 20,
                        workers = 4
                        )
t1 = time()
print("<----------[INFO] model was trained in " + str(round((t1-t0)/60, 1)) + " minutes")

print("<----------[INFO] evaluating network...")
# Построение графиков потерь на тренировочном и валидационном наборах
N = np.arange(1, EPOCHS+1)
plt.style.use('ggplot')
plt.figure()
plt.plot(N, H.history['loss'], label='train_loss')
plt.plot(N, H.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.savefig('output/loss.png')

# Сохранение модели на диск
print("<----------[INFO] serializing network...")
model.save('output/cnn.hdf5')
