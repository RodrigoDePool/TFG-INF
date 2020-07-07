from modules.util import *
import json

fichero_salida = 'hiperparameters.json'
# Data to train and validate
trainw, traino, testw, testo, scaler = scaled_train_test_division(
    df, ip_example, train_size, test_size)
n_values = [10 * i for i in range(1, 10)]
epochs = 300
data = {n: (None, None) for n in n_values}
for n in n_values:
    model = Sequential()
    model.add(LSTM(n, activation='relu', input_shape=(input_window, 1)))
    model.add(Dense(input_window))
    model.compile(optimizer='adam', loss='mse')
    hist = model.fit(
        trainw,
        traino,
        epochs=epochs,
        validation_data=(testw, testo),
        verbose=1)
    plt.clf()
    plt.title('MSE(epochs)')
    plt.ylabel('Validation MSE')
    plt.xlabel('Epochs')
    plt.plot(hist.history['val_loss'])
    plt.savefig('graphics/{}.pdf'.format(n))
    i = hist.history['val_loss'].index(max(hist.history['val_loss']))
    data[n] = (i + 1, hist.history['val_loss'][i])

# Store the information for future use
with open(fichero_salida, 'w') as f:
    f.write(json.dumps(data))
