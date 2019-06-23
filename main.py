from detector import Detector
from generator import Generator
from tensorflow import keras


adv = keras.models.Sequential()
det = Detector([28, 28], [5, 1], [2, 2], [2, 2], [0.5, 0.5], 5, 1)
gen = Generator([3,3], 28, [5, 1], [2,2], [2, 2], [0.5, 0.5], 1)
adv.add(gen.model)
adv.add(det.model)

optimizer = RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8)
adv.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

