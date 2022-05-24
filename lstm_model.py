## LSTM MODEL ARCHITECTURE



import torch 
import torch.nn as nn
from torch.autograd import Variable

# Defining the LSTM model as a class. 

class collisionClassifier(nn.Module):


    def __init__(self, no_of_features, features_to_hidden, features_in_hidden, max_sequence_length,no_of_layers = 1):
        
        super().__init__()
        
        # Parameters

        self.features_in_hidden = features_in_hidden
        self.features_to_hidden = features_to_hidden
        self.no_of_features = no_of_features
        self.no_of_layers = no_of_layers
        self.max_sequence_length = max_sequence_length
        


        #Pre-LSTM Layers
        
        self.linear_layer_1 = nn.Linear(no_of_features, features_to_hidden)
        self.relu_layer1 = nn.ReLU()
        self.dropout_layer1 = nn.Dropout(0.2)
        
        # LSTM Cell Layers
        
        self.lstm_cell = nn.LSTM(features_to_hidden, features_in_hidden, no_of_layers)
        

        # Post-LSTM Attention Vector Layers
        
        self.linear_layer_2 = nn.Linear(features_in_hidden,1)
        self.smc1 = nn.Softmax()

        # Post Attention Vector Multiplication Layers

        features_in_hidden += 4   # 4 additional parameters of the warning
        self.linear_layer_3 = nn.Linear(features_in_hidden,8)
        self.linear_layer_4 = nn.Linear(8,4)
        self.dropout_layer2 = nn.Dropout(0.2)
        self.linear_layer_5 = nn.Linear(4,2)
        self.smc2 = nn.Softmax()
        
        
    def __init__hidden(self, max_sequence_length):
        
        hidden = torch.rand(1, max_sequence_length, self.features_in_hidden)
        
        return Variable(hidden, requires_grad = True)
        
    
    def __init__cellstate(self, max_sequence_length):
        
        cellstate = torch.rand(1,max_sequence_length , self.features_in_hidden)
        
        return Variable(cellstate, requires_grad = True)
    
    
    def attention_analyzer(self, output, sequence_length):
        
        results_vector = []
        
        for i in range(output.shape[1]):
            
            # passing each hidden layer through a linear layer to obtain a scalar output
            # and then recoding that value in the results_vector to pass to the softmax

            result = self.linear_layer_2(output[0, i, :])
            results_vector.append(result)

        results_vector = torch.FloatTensor(results_vector)
        #print(results_vector)
        return self.smc1(results_vector)


    

    def forward_pass(self, input_data,sequence_length, scenario_vect):
        
        # Pre-LSTM Layers

        output = self.linear_layer_1(input_data)
        output = self.relu_layer1(output)
        output = self.dropout_layer1(output)

        # LSTM Cell

        output, _ = self.lstm_cell(output)

        # Attention Vector

        attention_vector = self.attention_analyzer(output,sequence_length)
        print(output.shape,attention_vector.shape)
        output = attention_vector @output[0, : ,:] 
        print(output)

        output = torch.cat((output,scenario_vect),0) 
        
        # Post Attention Vector Layers

        output = self.linear_layer_3(output)
        output = self.linear_layer_4(output)
        output = self.dropout_layer2(output)
        output = self.linear_layer_5(output)

        return self.smc2(output)