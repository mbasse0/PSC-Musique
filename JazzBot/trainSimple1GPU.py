from model import *
from data_loader import *


device = "cuda" if torch.cuda.is_available() else "cpu"
model = Transformer(
    num_tokens=len(CV), dim_model=256, num_heads=2, num_encoder_layers=1, num_decoder_layers=6, dropout_p=0.1
).to(device)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
dataloader = dataloader("../Data/",batch_size = 32,N=31)


def train_loop(model, opt, loss_fn, dataloader):
    model.train()
    total_loss = 0
    

    # Entraîner le modèle sur une boucle

    for batch in dataloader:
        y_input, y_expected = batch

        # X est ce qu'on donne à l'encoder. Un vecteur nul dans notre cas en l'absence d'informations contextuelles
        X = torch.tensor([0]*len(y_input))
        X, y_input, y_expected = X.clone().detach() , y_input.clone().detach() , y_expected.clone().detach() 

        
        # Get mask to mask out the next words
        sequence_length = y_input.size(1)
        tgt_mask = model.get_tgt_mask(sequence_length).to(device)

        # Standard training except we pass in y_input and tgt_mask
        pred = model(X, y_input, tgt_mask)

        # Permute pred to have batch size first again
        pred = pred.permute(0, 2, 1)      
        loss = loss_fn(pred, y_expected)

        opt.zero_grad()
        loss.backward()
        opt.step()
    
        total_loss += loss.detach().item()
        
    return total_loss / len(dataloader)

def fit(model, opt, loss_fn, train_dataloader, epochs):
    # Used for plotting later on
    train_loss_list= []
    
    print("Training model")
    for epoch in range(epochs):
        print("-"*25, f"Epoch {epoch + 1}","-"*25)
        
        train_loss = train_loop(model, opt, loss_fn, train_dataloader)
        train_loss_list += [train_loss]
        
        #validation_loss = validation_loop(model, loss_fn, val_dataloader)
        #validation_loss_list += [validation_loss]
        
        print(f"Training loss: {train_loss:.4f}")
        #print(f"Validation loss: {validation_loss:.4f}")
        print()
        
    return train_loss_list#, validation_loss_list

train_loss_list = fit(model, opt, loss_fn, dataloader, 2)
print(train_loss_list)