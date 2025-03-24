using MLDatasets: MNIST
using DecisionTree
using Statistics

nr_of_training_data = 60000     # <= 60 000
nr_of_testing_data = 10000      # <= 10 000

nr_of_trees = 100
part_samp = 0.8
max_dep = 20
min_samp_leaf = 1

function accuracy_all(predictions, labels)
    return mean(predictions .== labels)
end

function metrics_by_class(predictions, labels, class_index)
    tp = sum((predictions .== class_index) .& (labels .== class_index))
    tn = sum((predictions .!= class_index) .& (labels .!= class_index))
    fp = sum((predictions .== class_index) .& (labels .!= class_index))
    fn = sum((predictions .!= class_index) .& (labels .== class_index))

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    specifity = tn / (tn + fp)
    accuracy = (tn + tp) / (tn + tp + fn + fp)
    f1 = 2 / ((1/recall) + (1/precision))
    return recall, precision, specifity, accuracy, f1
end

trainSet = MNIST(;Tx=Float32, split=:train)
testSet = MNIST(;Tx=Float32, split=:test)

x_train = trainSet.features[:, :, 1:nr_of_training_data]
y_train = trainSet.targets[1:nr_of_training_data]
x_train = transpose(reshape(x_train, 28*28, size(x_train, 3)))

x_test = testSet.features[:, :, 1:nr_of_testing_data]
y_test = testSet.targets[1:nr_of_testing_data]
x_test = transpose(reshape(x_test, 28*28, size(x_test, 3)))

rf_model = RandomForestClassifier(
        n_trees = nr_of_trees, 
        partial_sampling = part_samp,
        max_depth = max_dep,
        min_samples_leaf = min_samp_leaf
    )

@time fit!(rf_model, x_train, y_train)

y_pred = predict(rf_model, x_test)

test_accuracy = accuracy_all(y_pred, y_test)
println("\nAccuracy: $(test_accuracy)")

println("\nMetrics for each class:")
for class_index in 0:9
    recall, precision, specifity, accuracy, f1 = metrics_by_class(y_pred, y_test, class_index)
    println("Class $(class_index): Recall = $(recall), Precision = $(precision), Specifity = $(specifity), Accuracy = $(accuracy), F1 = $(f1)")
end
