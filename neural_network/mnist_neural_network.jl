using MLDatasets: MNIST
using Flux
using Statistics
using Plots
#using BSON

nr_of_training_data = 60000     # <= 60 000
nr_of_testing_data = 10000      # <= 10 000
epochs = 12
batches = 256

function train_model(model, x, y, epochs, opt, batches)
    for i in 1:epochs
        @show i,loss(model, x, y)
        data = Flux.DataLoader((x,y), batchsize = batches, shuffle=true) 
        for d in data 
            grad = Flux.gradient(loss, model, d[1], d[2]) 
            Flux.update!(opt, model, grad[1]) 
        end
    end 
end

function generate_confusion_matrix(model, x, y)
    predictions = Flux.onecold(model(x)) .-1
    labels = Flux.onecold(y) .-1
    
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
    
    savefig(plt, "confusion_matrix.png")
    
    diagonal_sum = sum(confusion_matrix[i, i] for i in 1:num_classes)
    total_sum = sum(confusion_matrix)
    acc = round(diagonal_sum/total_sum*100, digits=2)
    
    println("\nCorrectly classified: $(diagonal_sum) out of $(total_sum) ($(acc)%)")
    println("\nAccuracy: $(acc)%")
    #return confusion_matrix
end

function metrics_by_class(model, x, y, class_index)
    predictions = Flux.onecold(model(x)) .-1
    labels = Flux.onecold(y) .-1

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

model = Chain(
    Conv((3,3), 1=>4, tanh),    # convolution 3x3, so the actual size is 28-2, output: 26x26x4
    MaxPool((2,2)),             # reduce the size by half, so the output is 13x13x4
    Conv((3,3), 4=>8, tanh),    # convolution 3x3, so the actual size is 13-2, output: 11x11x8
    MaxPool((2,2)),             # reduce the size by half, so the output is 5x5x8
    Flux.flatten,               # transform into a vector 5*5*8 = 200
    Dense(200 => 60, tanh),     # 200 inputs to 60 neurons
    Dense(60 => 10)             # 60 inputs to 10 neurons
)

trainSet = MNIST(;Tx=Float32, split=:train)
testSet = MNIST(;Tx=Float32, split=:test)

x_train = trainSet.features[:, :, 1:nr_of_training_data]
y_train = trainSet.targets[1:nr_of_training_data]
x_train = reshape(x_train, 28, 28, 1, nr_of_training_data)
y_train = Flux.onehotbatch(y_train, 0:9)

x_test = testSet.features[:, :, 1:nr_of_testing_data]
y_test = testSet.targets[1:nr_of_testing_data]
x_test = reshape(x_test, 28, 28, 1, nr_of_testing_data)
y_test = Flux.onehotbatch(y_test, 0:9)

model(x_train)

loss(model, x, y) = Flux.logitcrossentropy(model(x), y)

opt = Flux.setup(Adam(), model)

println("loss in epochs:")
@time train_model(model, x_train, y_train, epochs, opt, batches)
@show loss(model, x_test, y_test)
#BSON.@save "mnist_model.bson" model                                # save the trained model to a file

generate_confusion_matrix(model, x_test, y_test)

println("\nMetrics for each class:")
for class_index in 0:9
    recall, precision, specificity, accuracy, f1 = metrics_by_class(model, x_test, y_test, class_index)
    println("Class $(class_index): Recall = $(recall), Precision = $(precision), Specificity = $(specificity), Accuracy = $(accuracy), F1 = $(f1)")
end
