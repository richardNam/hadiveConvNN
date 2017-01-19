-- model 9, added batchnorm and relu to the classifier
-- then the last pool step as a stride equal to kernel width/height
npy4th = require 'npy4th'
require 'image'
require 'nn'



-- initate the model
model = nn.Sequential()
--nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padW])
model:add(nn.SpatialConvolution(3,32, 3,3, 1,1, 0,0))
model:add(nn.Dropout(0.4))
model:add(nn.SpatialBatchNormalization(32,1e-3))
model:add(nn.ReLU())
model:add(nn.SpatialConvolution(32,64, 3,3, 1,1, 0,0))
model:add(nn.SpatialBatchNormalization(64,1e-3))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2,2,1,1))

model:add(nn.SpatialConvolution(64,128, 3,3, 1,1, 0,0))
model:add(nn.Dropout(0.4))
model:add(nn.SpatialBatchNormalization(128,1e-3))
model:add(nn.ReLU())
model:add(nn.SpatialConvolution(128,128, 3,3, 1,1, 0,0))
model:add(nn.SpatialBatchNormalization(128,1e-3))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2,2,1,1))

model:add(nn.SpatialConvolution(128,128, 5,5, 1,1, 0,0))
model:add(nn.Dropout(0.4))
model:add(nn.SpatialBatchNormalization(128,1e-3))
model:add(nn.ReLU())
model:add(nn.SpatialConvolution(128,128, 5,5, 1,1, 0,0))
model:add(nn.SpatialBatchNormalization(128,1e-3))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2,2,2,2))
dim = 128*7*3

-- flatten the matrix, find the compenents by taking the size of the matrices after the last pooling
model:add(nn.View(dim)) 

-- build a classifier
classifer = nn.Sequential()
classifer:add(nn.Dropout(0.1))
classifer:add(nn.Linear(dim, 256))
classifer:add(nn.BatchNormalization(256))
classifer:add(nn.ReLU())
classifer:add(nn.Dropout(0.1))
classifer:add(nn.Linear(256,2))

-- add the classifier to the model
model:add(classifer)

return model
