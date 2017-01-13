-- @Richard Nam
-- this script will balance and augment input data
-- balancing is done through data augmentation
c = require 'trepl.colorize'
npy4th = require 'npy4th'
lapp = require 'pl.lapp'
require 'math'
require 'xlua'
require 'optim'
require 'image'
require 'scripts.metrics'

timer = torch.Timer()
print(c.yellow 'Starting...')
local args = lapp [[
    --save               (default "data/dev_data.table")             save table to
    --max_class          (default 2)                                 gaussian kernel for normalization
    --images             (default 'tensors/X_train.npy')               training images
    --labels             (default 'tensors/y_train.npy')               training labels
    ]]

print(args)

-- read in data from source
dimage = npy4th.loadnpy(args.images):double()
dlabel = npy4th.loadnpy(args.labels):double()

-- create a table with augmentations
augmentations = {}
function hflip(test) 
    return image.hflip(test:clone())
end
function rotate(test)
    angle = torch.uniform(-1.0, 1.0)
    return image.rotate(test:clone(), angle)
end
function translate(test)
    x = math.random(-4,4)
    y = math.random(-4,4)
    return image.translate(test:clone(), x, y)
end
augmentations[1] = hflip
augmentations[2] = rotate
augmentations[3] = translate

-- count up the class n
counter = {negative_n=0, positive_n=0}
for i=1, dlabel:size()[1] do
    _label = dlabel[i]
    if _label == 0 then
        counter.negative_n = counter.negative_n + 1
    else
        counter.positive_n = counter.positive_n + 1
    end
end

-- put label indexes in a tensor and add to counter
counter.negative_idx = torch.zeros(counter.negative_n)
counter.positive_idx = torch.zeros(counter.positive_n)

-- fill the tensor with indexes
pos_cnt = 1
neg_cnt = 1
for i=1, dlabel:size()[1] do
    _label = dlabel[i]
    if _label == 0 then
        counter.negative_idx[neg_cnt] = i
        neg_cnt = neg_cnt + 1
    else
        counter.positive_idx[pos_cnt] = i
        pos_cnt = pos_cnt + 1
    end
end

-- max count class N (multiplier)
counter.max_class_multiplier = args.max_class

-- min count class N (multiplier)
counter.min_class_multiplier = math.floor(counter.negative_n * counter.max_class_multiplier / (counter.positive_n))

-- image dimensions
counter.planes = dimage[1]:size()[1]
counter.height = dimage[1]:size()[2]
counter.width = dimage[1]:size()[3]

-- new dataset size N
counter.n = counter.negative_n*counter.max_class_multiplier + counter.positive_n * counter.min_class_multiplier

-- initate the tensors
counter.data = torch.zeros(counter.n, counter.planes, counter.height, counter.width)
counter.labels = torch.zeros(counter.n)

-- augment
_idx = 1
-- augment the major class
-- number of interations of major class
print(c.blue 'Processing negatives...')
for i=1, counter.max_class_multiplier do
    -- iterate all images in major class
    for j=1, counter.negative_n do
        xlua.progress(_idx, (counter.max_class_multiplier*counter.negative_n))       
        -- pull index, label, real image
        _image_idx = counter.negative_idx[j]
        _label = 0
        _image = dimage[_image_idx]:clone()
        -- to augment or not
        aug = torch.rand(1)[1]
        if aug > (1/counter.max_class_multiplier) then
            -- pick random augmentation
            pick = math.random(1,3)
            first_augment = augmentations[pick]
            -- apply augmentation
            _image = first_augment(_image)
        end
        -- check second augment threshold ((1 - major class prob) / 2) + major class prob
        if aug > (((1-(1/counter.max_class_multiplier))/2) + (1/counter.max_class_multiplier)) then
            trig = true
            -- make sure dont pick the same augmentation as the first
            while trig do
                pick2 = math.random(1,3)
                if pick2 ~= pick then
                    -- pull the second augmentation (non-dup)
                    second_augment = augmentations[pick2]
                    _image = second_augment(_image)
                    trig = false
                end
            end
        end
        -- put it in the counter data and labels
        counter.labels[_idx] = _label
        counter.data[_idx] = _image
--         print(_image_idx, _idx)        
        _idx = _idx + 1
        
    end
end

-- augment the minor class
-- number of interations of minor class
print(c.blue 'Processing positives...')  
for i=1, counter.min_class_multiplier do
    -- iterate all images in minor class
    for j=1, counter.positive_n do
        xlua.progress(_idx-(counter.max_class_multiplier*counter.negative_n), 
            (counter.min_class_multiplier*counter.positive_n))      
        -- pull index, label, real image
        _image_idx = counter.positive_idx[j]
        _label = 1
        _image = dimage[_image_idx]:clone()
        -- to augment or not
        aug = torch.rand(1)[1]
        if aug > (1/counter.min_class_multiplier) then
            -- pick random augmentation
            pick = math.random(1,3)
            first_augment = augmentations[pick]
            -- apply augmentation
            _image = first_augment(_image)
        end
        -- check second augment threshold ((1 - major class prob) / 2) + major class prob
        if aug > (((1-(1/counter.min_class_multiplier))/2) + (1/counter.min_class_multiplier)) then
            trig = true
            -- make sure dont pick the same augmentation as the first
            while trig do
                pick2 = math.random(1,3)
                if pick2 ~= pick then
                    -- pull the second augmentation (non-dup)
                    second_augment = augmentations[pick2]
                    _image = second_augment(_image)
                    trig = false
                end
            end
        end
        -- put it in the counter data and labels
        counter.labels[_idx] = _label
        counter.data[_idx] = _image      
        _idx = _idx + 1
    end
end
print(c.blue'Saving table to: '..args.save)
torch.save(args.save, counter)
--return counter