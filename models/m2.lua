-- more planes, and no dropout
npy4th = require 'npy4th'
require 'image'
require 'nn'



-- initate the model
model = nn.Sequential()
--nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padW])
model:add(nn.SpatialConvolution(3,32, 3,3, 1,1, 0,0))
model:add(nn.ReLU())
model:add(nn.SpatialConvolution(32,64, 3,3, 1,1, 0,0))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2,2,1,1))

model:add(nn.SpatialConvolution(64,128, 3,3, 1,1, 0,0))
model:add(nn.ReLU())
model:add(nn.SpatialConvolution(128,128, 3,3, 1,1, 0,0))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2,2,1,1))
dim = 128*22*14

-- flatten the matrix, find the compenents by taking the size of the matrices after the last pooling
model:add(nn.View(dim))

-- build a classifier
classifer = nn.Sequential()
classifer:add(nn.Linear(dim, 900))
classifer:add(nn.ReLU())
classifer:add(nn.Linear(900,2))

-- add the classifier to the model
model:add(classifer)

return model