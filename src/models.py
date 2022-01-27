from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# Neural Network
# split into input (X) and output (y) variables
trainFeats, trainLabel, testFeats, testLabel = partition(supervisedPreset(), .75)

# define the keras model
model = Sequential()
model.add(Dense((trainFeats.shape[1] + 1), input_dim=(trainFeats.shape[1]), kernel_initializer='normal', activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
history=model.fit(trainFeats, trainLabel, epochs=25, batch_size=150, verbose=1, validation_split=0.2)
predictions = model.predict(testFeats)

print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

regMetrics(testLabel, predictions)

# Plot NN effects on individual PCA components
plotReg(testFeats, testLabel, predictions)
plotSeparatePCA(testFeats, testLabel, predictions)

# SVM Anomaly Detection
matrix = getDesign()
matrix = matrix[:, 1:]
anomalyDetect(matrix)

print("Begin original design testing")
design = supervisedPreset()
supKitchenSink(design)

# PCA did not work, as the values were out of the range for regression. 

print("Begin testing after removal of low variance")
lowVar = runElimLowVarFeatsExclLabel(design)
supKitchenSink(lowVar)

supKitchenSink(both)


# Create elbow graphs

print("No preprocessing")
original = unsupPreset()
ids, original = separate(original)
allResults = unsupKitchenSink(original, ids)


screePCA = PCA(14)
screePCA.fit(original)
PC_values = np.arange(screePCA.n_components_) + 1
plt.plot(PC_values, screePCA.explained_variance_ratio_, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.show()
print("With Feature Reduction")
wPCA = runPCA(original)
allResults = np.append(allResults, unsupKitchenSink(wPCA, ids, testNumInit = allResults.shape[0]), axis = 0)

print("With Low Variance Reduction")
lowVar = runElimLowVarFeats(original)
allResults = np.append(allResults, unsupKitchenSink(lowVar, ids, testNumInit = allResults.shape[0]), axis = 0)

print("With Both")
both = runPCA(runElimLowVarFeats(original))
allResults = np.append(allResults, unsupKitchenSink(both, ids, testNumInit = allResults.shape[0]), axis = 0)

for entry in allResults:
  print(entry)