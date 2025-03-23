using MLDatasets: MNIST
using Flux
using Statistics

nr_of_training_data = 60000     # <= 60 000
nr_of_testing_data = 10000      # <= 10 000
epochs = 12
batches = 256

function train_model(model, x, y, epochs, opt, batches)
    @assert 1 <= batches <= size(x,4)
    @inbounds for i in 1:epochs
        @show i,loss(model, x, y)
        data = Flux.DataLoader((x,y), batchsize = batches, shuffle=true) 
        for d in data 
            grad = Flux.gradient(loss, model, d[1], d[2]) 
            Flux.update!(opt, model, grad[1]) 
        end
    end 
end

function accuracy_all(model, x, y)
    predictions = Flux.onecold(model(x)) .-1
    labels = Flux.onecold(y) .-1
    
    return mean(predictions .== labels)
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
    specifity = tn / (tn + fp)
    accuracy = (tn + tp) / (tn + tp + fn + fp)
    f1 = 2 / ((1/recall) + (1/precision))
    return recall, precision, specifity, accuracy, f1
end

model = Chain(
    Conv((3,3), 1=>4, tanh),    # konwulcja 3x3, wiec rzeczywisty rozmiar 28-2, wyjscie:26x26x4
    MaxPool((2,2)),             # zmniejszamy rozmiar o polowe, czyli wyjscie 13x13x4
    Conv((3,3), 4=>8, tanh),    # konwulcja 3x3, wiec rzeczywisty rozmiar 13-2, wyjscie:11x11x8
    MaxPool((2,2)),             # zmniejszamy rozmiar o polowe, czyli wyjscie 5x5x8
    Flux.flatten,               # przeksztalcamy na wektor 5*5*8 = 200,
    Dense(200 => 60, tanh),     # 200 wejsc na 60 neuronow
    Dense(60 => 10)             # 60 wejsc na 10 neuronow
)

trainSet = MNIST(;Tx=Float32, split=:train)
testSet = MNIST(;Tx=Float32, split=:test)

x_train = trainSet.features[:, :, 1:nr_of_training_data]
x_train = reshape(x_train, 28, 28, 1, nr_of_training_data)
y_train = Flux.onehotbatch(trainSet.targets[1:nr_of_training_data], 0:9)

x_test = testSet.features[:, :, 1:nr_of_testing_data]
x_test = reshape(x_test, 28, 28, 1, nr_of_testing_data)
y_test = Flux.onehotbatch(testSet.targets[1:nr_of_testing_data], 0:9)

model(x_train)

loss(model, x, y) = Flux.logitcrossentropy(model(x), y)
@show loss(model, x_test, y_test)

opt = Flux.setup(Adam(), model)

@time train_model(model, x_train, y_train, epochs, opt, batches)
@show loss(model, x_test, y_test)

test_accuracy = accuracy_all(model, x_test, y_test)
println("\nDokładność (Accuracy) na zbiorze testowym: $(test_accuracy)")

println("\nRecall i Precision dla poszczególnych klas:")
for class_index in 0:9
    recall, precision, specifity, accuracy, f1 = metrics_by_class(model, x_test, y_test, class_index)
    println("Klasa $(class_index): Recall = $(recall), Precision = $(precision), Specifity = $(specifity), Accuracy = $(accuracy), F1 = $(f1)")
end
