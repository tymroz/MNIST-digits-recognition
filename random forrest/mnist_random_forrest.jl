using MLDatasets: MNIST
using DecisionTree
using Statistics
using Plots

nr_of_training_data = 60000     # <= 60 000
nr_of_testing_data = 10000      # <= 10 000

nr_of_trees = 100
part_samp = 0.8
max_dep = 20
min_samp_leaf = 1

function print_confusion_matrix(predictions, labels)
    num_classes = 10
    confusion_matrix = zeros(Int, num_classes, num_classes)
    
    for i in 1:length(labels)
        true_label = labels[i] + 1
        pred_label = predictions[i] + 1
        confusion_matrix[true_label, pred_label] += 1
    end

    plt = heatmap(
        0:9, 0:9, confusion_matrix, 
        xlabel="Predicted", ylabel="True", 
        title="Confusion Matrix",
        color=:thermal,
        aspect_ratio=:equal,
        xticks=0:1:9,
        yticks=0:1:9
    )
    
    for i in 0:9, j in 0:9
        annotate!(j, i, text(string(confusion_matrix[i+1,j+1]), 8, :white))
    end
    
    savefig(plt, "confusion_matrix_rf.png")
    
    diagonal_sum = sum(confusion_matrix[i, i] for i in 1:num_classes)
    total_sum = sum(confusion_matrix)
    acc = round(diagonal_sum / total_sum * 100, digits=2)
    
    println("\nCorrectly classified: $(diagonal_sum) out of $(total_sum) ($(acc)%)")
    println("\nAccuracy: $(acc)%")
    
    #return confusion_matrix
end

function metrics_by_class(predictions, labels, class_index)
    tp = sum((predictions .== class_index) .& (labels .== class_index))
    tn = sum((predictions .!= class_index) .& (labels .!= class_index))
    fp = sum((predictions .== class_index) .& (labels .!= class_index))
    fn = sum((predictions .!= class_index) .& (labels .== class_index))

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    specificity = tn / (tn + fp)
    accuracy = (tn + tp) / (tn + tp + fn + fp)
    f1 = 2 / ((1/recall) + (1/precision))

    return recall, precision, specificity, accuracy, f1
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

print_confusion_matrix(y_pred, y_test)

println("\nMetrics for each class:")
for class_index in 0:9
    recall, precision, specifity, accuracy, f1 = metrics_by_class(y_pred, y_test, class_index)
    println("Class $(class_index): Recall = $(recall), Precision = $(precision), Specifity = $(specifity), Accuracy = $(accuracy), F1 = $(f1)")
end
