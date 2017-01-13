-- get the classes and count
function fpr(label, prediction)
    classes = {}
    for i=1, label:size()[1] do
        _label = label[i]
        if classes[_label] == nil then
            classes[_label] = 1
        else
            classes[_label] = classes[_label] + 1
        end
    end

    -- precision, recall
    tp = 0
    psum = 0
    for j=1, prediction:size()[1] do
        _p = prediction[j]
        _l = label[j]
        if (_l==#classes and _p==#classes) then
            tp = tp + 1 -- count up the tp
        end
        if _p==#classes then
            psum = psum + 1 -- sum up the total predictions in the true class
        end
    end

    precision = tp / psum
    recall = tp / classes[2]
    f1 = 2 * (precision*recall) / (precision+recall)
    return f1, precision, recall
end