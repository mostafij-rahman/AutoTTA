import numpy as np
from keras import models, layers, datasets, optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import regularizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

# Experiment parameters
learning_rate = 0.1
lr_decay = 1e-6
lr_drop = 20
        
def lr_scheduler(epoch):
    return learning_rate * (0.5 ** (epoch // lr_drop))
    
reduce_lr = LearningRateScheduler(lr_scheduler)
filepath = 'child_models/cifar10vgg_baseline_best.h5'
checkpoint = ModelCheckpoint(filepath=filepath, 
                             monitor='val_loss',
                             verbose=1, 
                             save_best_only=True,
                             mode='min')
callbacks = [reduce_lr, checkpoint]

def get_train_batches(X, y, batches, batch_size):
    while True:
        ix = np.arange(len(X))
        np.random.shuffle(ix)
        for i in range(batches):
            _ix = ix[i*batch_size:(i+1)*batch_size]
            _X = X[_ix]
            _y = y[_ix]
            _X = _X.astype(np.float32) / 255
            yield _X, _y
       
class ChildNetwork:
    def __init__(self, input_shape, n_samples = 80000, num_classes = 10):
        self.num_classes = num_classes
        self.N_SAMPLES = n_samples
        self.weight_decay = 0.0005
        self.x_shape = input_shape #[32,32,3]
        #self.model = self.create_model(input_shape)
        self.model = self.build_model()
        #optimizer = optimizers.SGD(decay=1e-4)
        optimizer = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        self.model.compile(optimizer, 'categorical_crossentropy', ['accuracy'])

    def create_model(self, input_shape):
        x = input_layer = layers.Input(shape=input_shape)
        x = layers.Conv2D(32, 3, activation='relu')(x)
        x = layers.Conv2D(64, 3, activation='relu')(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(10, activation='softmax')(x)
        return models.Model(input_layer, x)
        
    # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.
    def build_model(self):
        model = Sequential()
        weight_decay = self.weight_decay

        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=self.x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))
        return model

    def fit(self, Xtrain, ytrain, Xval, yval, n_epoch, batch_size):
    	child_batches = len(Xtrain) // batch_size
    	gen = get_train_batches(Xtrain, ytrain, child_batches, batch_size)
    	self.model.fit_generator(gen, child_batches, epochs = n_epoch, validation_data=(Xval, yval),callbacks=callbacks,verbose=1)
    	return self

    def evaluate(self, X, Y):
        return self.model.evaluate(X, Y, verbose=1)[1]
    
    def predict(self, X):
        return self.model.predict(X, verbose=1)

