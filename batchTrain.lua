-- training script
c = require 'trepl.colorize'
npy4th = require 'npy4th'
lapp = require 'pl.lapp'
path = require 'pl.path'
require 'math'
require 'xlua'
require 'optim'
require 'image'
require "lfs"
require 'scripts.metrics'

timer = torch.Timer()
print(c.yellow 'Starting...')
local args = lapp [[
    --save               (default "m1")                          save model name
    --output_dir         (default "output/")                     directory to save the model output
    --model              (default "models/m1.lua")               location of saving the model, full lua filename
    --batch_size         (default 4)                             minibatch size
    --epochs             (default 5)                             total epochs
    --gpu                                                        train the gru network
    --table              (default FALSE)                         table with processed data
    --dropout            (default 0.0) 
    --init_weight        (default 0.1)                           random weight initialization limits
    --lr                 (default 0.0001)                        learning rate
    --lrd                (default 0)                             learning rate decay
    --weightDecay        (default 0)                             sgd only
    --momentum           (default 0)                             sgd only
    --nesterov                                                   enables Nesterov momentum
    --evaluate           (default 5)                             number of epochs before evaluating
    --gkernel            (default 3)                             gaussian kernel for normalization
    --yuv                                                        flag for converting to YUV color 
    --train_images       (default 'tensors/X_train.npy')         training images
    --train_labels       (default 'tensors/y_train.npy')         training labels
    --test_images        (default 'tensors/X_val.npy')           test images
    --test_labels        (default 'tensors/y_val.npy')           test labels
    ]]

-- check and create args.output_dir path
if path.isdir(args.output_dir) == false then
    lfs.mkdir(args.output_dir)
end

print(args)
-- convert to cuda
if args.gpu then
    print(c.red 'Training on the gpu')
    require 'cunn'
    -- import model
    local model = nn.Sequential()
    model:add(dofile(args.model))
    model = model:cuda()
    criterion = nn.CrossEntropyCriterion():cuda()
    print(model)
else
    print(c.red 'Not training on the gpu')
    require 'nn'
    -- import model
    local model = nn.Sequential()
    model:add(dofile(args.model))
    criterion = nn.CrossEntropyCriterion()
    print(model)
end
-- define optimzation criterion and parameters
print(c.blue '===>'..' Configuring training method')
optimState = {
    learningRate = args.lr,
    learnRatDecay = args.lrd,
    weightDecay = args.weightDecay,
    nesterov = args.nesterov,
    momentum = args.momentum,
    learningRateDecay = 1e-7}
optimMethod = optim.sgd
trainlogger = optim.Logger(args.output_dir..args.save..'_train.log')
testlogger = optim.Logger(args.output_dir..args.save..'_validation.log')
trainlogger:setNames{'epoch', 'global_acc'}
testlogger:setNames{'epoch', 'global_acc', 'f1', 'precision', 'recall'}

if model then
   parameters,gradParameters = model:getParameters()
end

-- load data
print(c.blue '===>'..' Loading data')
if args.table ~= 'FALSE' then
    print(c.blue '===>'..' Loading from table')
    counter = torch.load(args.table)
    print(counter)
    dimage = counter.data:double()
    dlabel = counter.labels:double() + 1
    test_image = npy4th.loadnpy(args.test_images):double()
    test_label = npy4th.loadnpy(args.test_labels):double() + 1
else
    print(c.blue '===>'..' Loading from source')
    dimage = npy4th.loadnpy(args.train_images):double()
    dlabel = npy4th.loadnpy(args.train_labels):double() + 1
    test_image = npy4th.loadnpy(args.test_images):double()
    test_label = npy4th.loadnpy(args.test_labels):double() + 1
end
if args.gpu then dimage = dimage:cuda() end
if args.gpu then test_image = test_image:cuda() end
-- store predictions
test_predictions = torch.zeros(test_label:size()[1])

-- preprocess the data # -- okay
-- convert to color space
if args.yuv then 
    _channels = {'y','u','v'}
    print(c.blue'===>'..' Preprocessing data: colorspace YUV')
    for i = 1, dimage:size()[1] do
        dimage[i] = image.rgb2yuv(dimage[i])
    end
    for i = 1, test_image:size()[1] do
        test_image[i] = image.rgb2yuv(test_image[i])
    end
else
    print(c.blue'===>'..' Preprocessing data: colorspace RGB')
    _channels = {'r','g','b'}
end
    
-- subtract the mean, divide by the std
_mean={}
_std={}
-- normalize each channel in training data globally
for i,channel in ipairs(_channels) do
    _mean[i] = dimage[{ {},i,{},{} }]:mean()
    _std[i] = dimage[{ {},i,{},{} }]:std()
    dimage[{ {},i,{},{} }]:add(-_mean[i])
    dimage[{ {},i,{},{} }]:div(_std[i])
end
-- normalize each channel in test data globally
for i,channel in ipairs(_channels) do
   test_image[{ {},i,{},{} }]:add(-_mean[i])
   test_image[{ {},i,{},{} }]:div(_std[i])
end

-- normalize training set locally
kernel = image.gaussian1D(args.gkernel)
normalization = nn.SpatialContrastiveNormalization(1, kernel):double()
-- normalize all channels locally, TODO: make the cuda() part cleaner
for c in ipairs(_channels) do
    if args.gpu then
        print('Normalizing train on channel: '.._channels[c])
        for i = 1, dimage:size()[1] do
            dimage[{ i,{c},{},{} }] = normalization:forward(dimage[{ i,{c},{},{} }]:double()):cuda()
            xlua.progress(i, dimage:size()[1])
        end
        print('Normalizing test on channel: '.._channels[c])    
        for i = 1, test_image:size()[1] do
            test_image[{ i,{c},{},{} }] = normalization:forward(test_image[{ i,{c},{},{} }]:double()):cuda()
            xlua.progress(i, test_image:size()[1])
        end
    else
        print('Normalizing train on channel: '.._channels[c])       
        for i = 1, dimage:size()[1] do
            dimage[{ i,{c},{},{} }] = normalization:forward(dimage[{ i,{c},{},{} }])
            xlua.progress(i, dimage:size()[1])
        end
        print('Normalizing test on channel: '.._channels[c])
        for i = 1, test_image:size()[1] do
            test_image[{ i,{c},{},{} }] = normalization:forward(test_image[{ i,{c},{},{} }])
            xlua.progress(i, test_image:size()[1])
        end
    end
end

-- new training methods
print(c.blue '===>'..' Training')
previous_score = 0.0
model_best = nil
classes = {'1','2'}
for e=1, args.epochs do
    confusion = optim.ConfusionMatrix(classes)
    local targets = torch.CudaTensor(args.batch_size)
    --local targets = torch.Tensor(args.batch_size)
    local indices = torch.randperm(dimage:size(1)):long():split(args.batch_size)
    -- remove last element so that all the batches have equal size
    indices[#indices] = nil
    local tic = torch.tic()
    for t,v in ipairs(indices) do
        xlua.progress(t, #indices)
        local inputs = dimage:index(1,v)
        targets:copy(dlabel:index(1,v))
        local feval = function(x)
          if x ~= parameters then parameters:copy(x) end
          gradParameters:zero()
          local outputs = model:forward(inputs)
          local f = criterion:forward(outputs, targets)
          local df_do = criterion:backward(outputs, targets)
          model:backward(inputs, df_do)
          confusion:batchAdd(outputs, targets)
          return f,gradParameters
        end
        optim.sgd(feval, parameters, optimState)
    end
    confusion:updateValids()
    print(c.yellow 'Completed epoch: '..e)
    print(confusion)
    trainlogger:add{e, confusion.totalValid * 100}
    confusion:zero()
    -- evaluate the model after N number of epochs
    if e % args.evaluate == 0 then
        print(c.red '===>'..' Evaluate at epoch: '..e)
        _confusion = optim.ConfusionMatrix(classes)
        model:evaluate()
        for i=1,test_image:size()[1],args.batch_size do
            _upper = math.min(test_image:size()[1]-i, args.batch_size)
            local outputs = model:forward(test_image:narrow(1,i,_upper))
            _confusion:batchAdd(outputs, test_label:narrow(1,i,_upper))
            xlua.progress(i, test_image:size()[1])
        end
        print(_confusion)
        valid_score = _confusion.totalValid * 100
        ftr, ptr, rtr = fpr(test_label, test_predictions)
        testlogger:add{e, valid_score, ftr, ptr, rtr}
        print(c.red'f1: '..ftr..c.red', precision: '..ptr..c.red', recall: '..rtr)
        print(c.red'===>'..' Saving model to '..args.output_dir..args.save..'.net')
        confusion:zero()
        -- update best model
        if model_best==nil then
            model_best=model:clone()
            model_best_epoch = e
        end
        if previous_score < valid_score then
            previous_score = valid_score
            model_best = model:clone()
            model_best_epoch = e
            print(c.red'===>'..' Updating best model')
        end
        dump = {
            model=model,
            mean=_mean,
            std=_std,
            channels=_channels,
            parameters=args,
            model_best=model_best,
            model_best_epoch=model_best_epoch
            }
        torch.save(args.output_dir..args.save..'.net', dump)
        model:training()
    end
end


print('Time elapsed for '..args.epochs..' epochs ' .. timer:time().real/60 .. ' minutes')
print(c.yellow 'made it to the end')







