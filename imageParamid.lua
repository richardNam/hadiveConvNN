--[[
@Richard Nam
Input: a directory fill of images, RGB images 0-255 scale
Ouput: writes a output matrix to the same directory with t7
Note: Fixed input size of three
TODO: Create output matrix and fill elements with pixel scores
TODO: Input i/o so it reads images in real-time
TODO: Scoring function for output matrix after smoothing
TODO: Write normalize function in YUV color space
]]
timer = torch.Timer()
require 'xlua'
require 'nn'
require 'cunn'
require 'image'
require 'math'
npy4th = require 'npy4th'

print('START')
-- log function
function log(x) return nn.Exp():forward(nn.LogSoftMax():forward(x)) end

-- normalize the patch kernel
function NormalizePatch(patch, model_channels)
    input_image = image.scale(torch.Tensor(3,32,24),patch:clone())
    trans_image = torch.zeros(input_image:size())
    kernel = image.gaussian1D(7)
    normalization = nn.SpatialContrastiveNormalization(1, kernel):double()
    for i=1, #model_channels do
        trans_image[{ {i},{},{} }] = normalization:forward(input_image[{ {i},{},{} }]:double())
    end
    return trans_image
end

-- normalize the local patch
function NormalizeImage(im, mean, std)
    for k=1, #mean do
        im[k] = im[{{k},{},{}}]:add(-mean[k])
        im[k] = im[{{k},{},{}}]:div(std[k])
    end
    return im
end    

-- scan a directory
function scandir(directory)
    local i, t, popen = 0, {}, io.popen
    for filename in popen('ls -a "'..directory..'"'):lines() do
        i = i + 1
        t[i] = filename
    end
    return t
end

-- define fix boxes
boxes = {}
for j=1, 3 do
    boxes[j]={}
end
boxes[1].width=13-1
boxes[1].height=17-1
boxes[2].width=17-1
boxes[2].height=22-1
boxes[3].width=22-1
boxes[3].height=29-1


print('CREATING CUDA MODEL')
-- import the model table
model_table = torch.load('/home/{USER}/model.net')
--model_table = torch.load('/home/rnam/Documents/ped/run/output/20160712_full_aug_balanced_m8.net')
-- pull the best model and convert to a double, pull the mean and std and channels
model = model_table.model_best:double()
model_mean = model_table.mean
model_std = model_table.std
model_channels = model_table.channels
-- initate a cuda model
model_cuda = model:cuda()


-- images
root = '/home/{USER}/images/'
images = scandir(root)


for i,v in pairs(images) do
    if string.sub(v,-3)=='npy' then
        local arg_image = v
        print('READING IMAGE: '..arg_image)
        -- Read in an image, here as test
        raw_image = npy4th.loadnpy(root..arg_image):double()
        channels = raw_image:size()[1]
        height = raw_image:size()[2]
        width = raw_image:size()[3]


        -- normalize the image
        local full_image = NormalizeImage(raw_image, model_mean, model_std)


        -- pull the patches based fixed box sizes, also store the upper left (UL) and lower right (LR) in a different lua table 
        print('CREATING PATCHES and INPUT TENSOR')
        local coords = {}
        local patches = {}
        local scores = {}
        local cnt = 1
        N = 0
        for p=1, #boxes do
            N = N + (height-boxes[p].height) * (width-boxes[p].width)
        end


        -- create a tensor to put the patches in 
        local input_tensor = torch.zeros(N,3,32,24)

        -- first normalize the whole image
        for i=1, #boxes do -- this loop is for each box size
            for j=1, height-boxes[i].height do -- this one goes over y axes
                for k=1, width-boxes[i].width do
                    top=j
                    bottom=j+boxes[i].height
                    left=k
                    right=k+boxes[i].width
                    patch = full_image[{{},{top,bottom},{left,right}}] -- create the patches here
                    patches[cnt] = NormalizePatch(patch, model_channels) -- normalize on kernel on the patches
                    input_tensor[cnt] = patches[cnt]
                    coords[cnt] = {left,top,right,bottom} -- saved as UL(x,y), LR(x,y)
                    xlua.progress(cnt, N)
                    cnt = cnt + 1
                end
            end
        end


        -- put push through the model in batch mode to save memory
        bs = 400
        print('FORWARD PASS')
        for n=1, input_tensor:size()[1], bs do
            _max = math.min(n+bs,input_tensor:size()[1])
            _input = input_tensor[{{n,_max},{},{},{}}]
            out = model_cuda:forward(_input:cuda())
            _inner = torch.range(n,_max)
            for _j=1, _inner:size()[1] do
                scores[_inner[_j]] = log(out[_j]:double())
            end
            xlua.progress(n, input_tensor:size()[1])
        end


        print('FILLING THE MATRIX')
        -- fill the matrix
        -- {left,top,right,bottom}
        local matrix = torch.zeros(1,height, width)
        for i=1, #scores do
            max, index = torch.max(log(scores[i]),1)
            if index[1] == 2 then
                left = coords[i][1]
                top = coords[i][2]
                right = coords[i][3]
                bottom = coords[i][4]
                matrix[{{},{top,bottom},{left,right}}]:add(1)
            end
        end


        print('SAVING OUTPUT MATRIX')
        -- save the matrix
        torch.save(root..string.sub(arg_image,1,-5)..'.t7', matrix)
    end
    --matrix = nil
    --input_tensor = nil
    --out = nil
    collectgarbage()
end

print('Number of patches: '..#patches..', seconds: '..timer:time().real)

