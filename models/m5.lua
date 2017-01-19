-- baseline model
-- https://github.com/htwaijry/npy4th
npy4th = require 'npy4th'
require 'image'
require 'nn'



-- builder the model
model = nn.Sequential()
--nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padW])
model:add(nn.SpatialConvolution(3,16, 3,3, 1,1, 0,0))
model:add(nn.ReLU())
model:add(nn.SpatialConvolution(16,32, 3,3, 1,1, 0,0))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2,2))
dim = 32*14*10

-- flatten the matrix, find the compenents by taking the size of the matrices after the last pooling
model:add(nn.View(dim))

-- build a classifier
classifer = nn.Sequential()
classifer:add(nn.Linear(dim, 600))
classifer:add(nn.ReLU())
classifer:add(nn.Linear(600,2))

-- add the classifier to the model
model:add(classifer)

return model