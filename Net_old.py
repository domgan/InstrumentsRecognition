from tensorflow import keras
import Preprocess_old as Preprocess

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(128, 293)),
    keras.layers.Dense(128*2, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(Preprocess.train_input, Preprocess.train_labels, epochs=5)

results = model.evaluate(Preprocess.test_input, Preprocess.test_labels)
print('test loss, test acc:', results)

model.save("model.h5")
print("Saved model to disk")
