%Loading the dataset
load("cifar-10-data.mat");
data = double(data);
rng(20);

%Displaying sample images with labels
figure;
for i = 1:4
    idx = randi(size(data, 1));
    img = squeeze(data(idx, :, :, :)) / 255;  
    subplot(1, 4, i);
    imagesc(img);
    label_idx = labels(idx);
    if label_idx >= 1 && label_idx <= length(label_names)
        title(label_names{label_idx}, 'Interpreter', 'none');  
    else
        title('Unknown', 'Interpreter', 'none');  
    end
end


saveas(gcf, 'image.png');  

%Mersenne twister generator with Student ID
studentID = 38824515;
rng(studentID, 'twister'); 

%Selecting the Random classes 
class_indices = randperm(10, 3);
classes = class_indices';
selected_data = [];
selected_labels = [];

%Extracting the images and labels for the selected classes
for i = 1:length(classes)
    class_idx = classes(i);
    class_data = data(labels == class_idx, :, :, :);
    class_labels = labels(labels == class_idx);
    
    selected_data = cat(1, selected_data, class_data);
    selected_labels = cat(1, selected_labels, class_labels);
end

%Size information [18000, 32, 32, 3] and [18000, 1]

disp(size(selected_data));  
disp(size(selected_labels));  

save('/home/ks10/h-drive/361/cw1.mat', 'classes');

%Splitting data into training and testing 

total_samples = size(selected_data, 1);
training_samples = total_samples / 2;

training_index = randperm(total_samples, training_samples)';

training_data = selected_data(training_index, :, :, :);
training_labels = selected_labels(training_index);

testing_index = setdiff(1:total_samples, training_index)';
testing_data = selected_data(testing_index, :, :, :);
testing_labels = selected_labels(testing_index);

%Verifying dimensions [9000, 32, 32, 3], [9000, 1], [9000, 32, 32, 3] and [9000, 1]

disp(size(training_data));  
disp(size(training_labels)); 
disp(size(testing_data));  
disp(size(testing_labels));  
save('/home/ks10/h-drive/361/cw1.mat', 'classes', 'training_index');


training_data_reshaped = reshape(training_data, [size(training_data, 1), 32*32*3]);
training_labels_reshaped = training_labels;
testing_data_reshaped = reshape(testing_data, [size(testing_data, 1), 32*32*3]);
testing_labels_reshaped = testing_labels;

%These are the reshaped sizes [9000, 3072], [9000, 1], [9000, 3072] and [9000, 1]

disp(size(training_data_reshaped));  
disp(size(training_labels_reshaped));  
disp(size(testing_data_reshaped));  
disp(size(testing_labels_reshaped));  

save('/home/ks10/h-drive/361/cw1.mat', 'classes', 'training_index', 'training_data_reshaped', ...
    'training_labels_reshaped', 'testing_data_reshaped', 'testing_labels_reshaped');

%Defining the KNN function

function predicted_labels = custom_knn(train_data, train_labels, test_data, k, distance_metric)
    num_test = size(test_data, 1);
    predicted_labels = zeros(num_test, 1);
    for i = 1:num_test
        distances = pdist2(test_data(i, :), train_data, distance_metric);
        [~, sorted_idx] = sort(distances, 'ascend');
        nearest_labels = train_labels(sorted_idx(1:k));
        predicted_labels(i) = mode(nearest_labels);
    end
end

%Model Training KNN with L2 Euclidean distance

tic;
predicted_labels_L2 = custom_knn(training_data_reshaped, training_labels_reshaped, testing_data_reshaped, 5, 'euclidean');
knnL2_time = toc;
knnL2_accuracy = sum(predicted_labels_L2 == testing_labels_reshaped) / length(testing_labels_reshaped);
knnL2_confusion = confusionmat(testing_labels_reshaped, predicted_labels_L2);

%Model Training KNN with Cosine distance

tic;
predicted_labels_cosine = custom_knn(training_data_reshaped, training_labels_reshaped, testing_data_reshaped, 5, 'cosine');
knnCosine_time = toc;
knnCosine_accuracy = sum(predicted_labels_cosine == testing_labels_reshaped) / length(testing_labels_reshaped);
knnCosine_confusion = confusionmat(testing_labels_reshaped, predicted_labels_cosine);

%Training and Evaluating SVM

tic;
svm_model = fitcecoc(training_data_reshaped, training_labels_reshaped);
predicted_labels_svm = predict(svm_model, testing_data_reshaped);
svm_time = toc;
svm_accuracy = sum(predicted_labels_svm == testing_labels_reshaped) / length(testing_labels_reshaped);
svm_confusion = confusionmat(testing_labels_reshaped, predicted_labels_svm);

%Training and Evaluating Decision Tree
tic;
tree_model = fitctree(training_data_reshaped, training_labels_reshaped);
predicted_labels_tree = predict(tree_model, testing_data_reshaped);
tree_time = toc;
tree_accuracy = sum(predicted_labels_tree == testing_labels_reshaped) / length(testing_labels_reshaped);
tree_confusion = confusionmat(testing_labels_reshaped, predicted_labels_tree);


save('/home/ks10/h-drive/361/cw1.mat', 'classes', 'training_index', ...
    'training_data_reshaped', 'training_labels_reshaped', 'testing_data_reshaped', 'testing_labels_reshaped', ...
    'knnL2_accuracy', 'knnL2_confusion', 'knnL2_time', ...
    'knnCosine_accuracy', 'knnCosine_confusion', 'knnCosine_time', ...
    'svm_accuracy', 'svm_confusion', 'svm_time', ...
    'tree_accuracy', 'tree_confusion', 'tree_time');

