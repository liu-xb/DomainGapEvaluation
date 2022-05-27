clear;clc;
fid = fopen('entropy.txt', 'w');

test_set = dir('cls');
test_set = {test_set.name};


for i = 1 : length(test_set)
    if test_set{i}(end) ~= 't'
        continue
    end
	load(['cls/', test_set{i}])
	cls=softmax(eval([test_set{i}(1:1:end-4)]));
    this_set_entropy = 0;
	for j = 1:size(cls,2)
        sub_cls = cls(:,j);
        entropy = - sub_cls.*log(sub_cls);
        entropy = sum(entropy);
        this_set_entropy = this_set_entropy + entropy;
    end
    this_set_entropy = - this_set_entropy / size(cls,2) /  log(1/size(cls,1)) ;
    fprintf(fid, '%.3f\n', this_set_entropy);
end

fclose(fid);