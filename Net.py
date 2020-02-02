from tensorflow import keras
import Preprocess

model = keras.Sequential([
    keras.layers.Conv2D(32, (5, 5), input_shape=(128, 293, 1), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(3, activation='sigmoid')
    ])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(Preprocess.train_input, Preprocess.train_labels, epochs=5)

results = model.evaluate(Preprocess.test_input, Preprocess.test_labels)
print('test loss, test acc:', results)

model.save("model/model.h5")
print("Saved model to disk")
