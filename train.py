import os
from sklearn.model_selection import train_test_split
from dataset import load_dataset
from generator import generator
from model import make_model
from utils import *

xy = load_dataset()
X = xy["x"]
Y = xy["y"]

X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.33, random_state=42)
win_len = 2992
num_leads_signal = 12
train_generator= generator(X=X_train, Y=Y_train, win_len=win_len, batch_size=10, num_leads_signal=num_leads_signal)
test_set = next(generator(X=X_test, Y=Y_test, win_len=win_len, batch_size=300, num_leads_signal=num_leads_signal))

print (test_set[0].shape)

model = make_model(num_leads_signal)
model.summary()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

history = model.fit_generator(train_generator,
                    epochs=5,
                    steps_per_epoch=10,
                    validation_data=(test_set[0], test_set[1]))

name = "mymodel"
model.save(os.path.join("trained_models", name + '.h5'))
save_set_to_pkl(X_test, Y_test, os.path.join("trained_models", name + ".pkl"))
save_history(history, name)