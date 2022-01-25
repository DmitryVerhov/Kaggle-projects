from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
PRE_TRAINED_MODEL_NAME = 'DeepPavlov/rubert-base-cased-sentence'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MAX_LEN = 300

'''Class with nlp model'''

class TransformerModel(nn.Module):

    def __init__(self, n_classes):
        super(TransformerModel, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME,
                                              return_dict=False)
        self.drop = nn.Dropout(p=0.5)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids,
                                     attention_mask=attention_mask)
        output = self.drop(pooled_output)
        return self.out(output)

'''This class will prepare our data to feed the model'''

class RatingDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
  
    def __len__(self):
        return len(self.reviews)
  
    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
          review,
          add_special_tokens=True,
          max_length=self.max_len,
          return_token_type_ids=False,
          padding='max_length',
          return_attention_mask=True,
          return_tensors='pt',
          truncation=True )

        return {'review_text': review,'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)}

'''It's global class to work with text rating prediction 
from preparing raw data for the model to evaluating the results'''

class RatingClassifier():
    
    def __init__(self,class_names = ['negative', 'neutral', 'positive']):
        self.class_names = class_names # User may change class quantity and names
        self.tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    
    # This will create data loaders to feed the model
    def create_data_loader(self,df, tokenizer, max_len, batch_size):
    
        ds = RatingDataset(reviews=df.text.to_numpy(),
                           targets=df.rating.to_numpy(),
                           tokenizer=tokenizer,
                           max_len=max_len)
        
        return DataLoader(ds,batch_size=batch_size,num_workers=2)  
        
    # Train model foe one epoch
    def train_epoch(self,model,data_loader,loss_fn,optimizer, 
                device,scheduler,n_examples):
        model = model.train()
        losses = []
        correct_predictions = 0

        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask)

            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        return correct_predictions.double() / n_examples, np.mean(losses)
    
    # Evaluating epoch results
    def eval_model(self,model, data_loader, loss_fn, device, n_examples):
        model = model.eval()

        losses = []
        correct_predictions = 0

        with torch.no_grad():
            for d in data_loader:
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                targets = d["targets"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                _, preds = torch.max(outputs, dim=1)

                loss = loss_fn(outputs, targets)

                correct_predictions += torch.sum(preds == targets)
                losses.append(loss.item())

        return correct_predictions.double() / n_examples, np.mean(losses)
    
    # Fit the model
    def fit(self,dataset, train_size = 0.8, # Here we can change train size 
            model_name ='Untitled'): # and model name to save the results
    
        # Splitting data
        df_train, df_test = train_test_split(dataset,
                                         train_size=train_size,
                                         stratify = dataset.rating,
                                         random_state=RANDOM_SEED)
        df_val, df_test = train_test_split(df_test, test_size=0.5,
                                           stratify = df_test.rating,
                                           random_state=RANDOM_SEED)
        
        # Checking shapes:
        print(f' Train shape: {df_train.shape}\n',
          f'Validation shape: {df_val.shape}\n',
          f'Test shape: {df_test.shape}')

        # Make loaders
        train_data_loader = self.create_data_loader(df_train, self.tokenizer, 
                                               MAX_LEN, BATCH_SIZE)
        val_data_loader = self.create_data_loader(df_val, self.tokenizer, 
                                             MAX_LEN, BATCH_SIZE)
        self.test_data_loader = self.create_data_loader(df_test, self.tokenizer, 
                                              MAX_LEN, BATCH_SIZE)

        # Forward data to device 
        data = next(iter(train_data_loader))
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        
        self.model = TransformerModel(len(self.class_names))
        self.model = self.model.to(device)
        # Add softmax layer to split results by categories:
        F.softmax(self.model(input_ids, attention_mask), dim=1)
        # This parameters are adviced by developers:    
        optimizer = AdamW(self.model.parameters(), lr=2e-5, correct_bias=False)
        loss_fn = nn.CrossEntropyLoss().to(device)
        total_steps = len(train_data_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
            )

        self.history = defaultdict(list)# here metrics will be saved
        best_accuracy = 0# This is to choose the best model
        # Trainging... 
        for epoch in range(EPOCHS):

            print(f'Epoch {epoch + 1}/{EPOCHS}')
            print('-' * 10)

            train_acc, train_loss = self.train_epoch(
                self.model, train_data_loader,    
                loss_fn,optimizer, 
                device, scheduler,len(df_train))

            print(f'Train loss {train_loss} accuracy {train_acc}')

            val_acc, val_loss = self.eval_model(
                self.model, val_data_loader,
                loss_fn, device, len(df_val))

            print(f'Val   loss {val_loss} accuracy {val_acc}')
            print()
            # Saving metrics
            self.history['train_acc'].append(train_acc)
            self.history['train_loss'].append(train_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_loss'].append(val_loss)

            if val_acc > best_accuracy:# Save best model
                torch.save(self.model.state_dict(),
                           f'{path}models/{model_name}.bin')
                best_accuracy = val_acc
           
        self.show_report()# Show classification report 
    
    def plot_history(self):# Visualize training process
        try:
            plt.plot(self.history['train_acc'],color = 'b', label='train accuracy')
            plt.plot(self.history['val_acc'], color = 'g', label='validation accuracy')
            plt.title('Training history')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend()
            plt.ylim([0.5, 1])
        
        except AttributeError:# operate the error if model hasn't been trained:
            print("Fit model at first!") 
    
    def show_report(self):# Use sklearn library to show metrics
        try:
            model = self.model.eval()

            self.review_texts = []
            predictions = []
            prediction_probs = []
            real_values = []

            with torch.no_grad():
                for d in self.test_data_loader:

                    texts = d["review_text"]
                    input_ids = d["input_ids"].to(device)
                    attention_mask = d["attention_mask"].to(device)
                    targets = d["targets"].to(device)

                    outputs = model(input_ids=input_ids,
                                    attention_mask=attention_mask)
                    _, preds = torch.max(outputs, dim=1)

                    probs = F.softmax(outputs, dim=1)

                    self.review_texts.extend(texts)
                    predictions.extend(preds)
                    prediction_probs.extend(probs)
                    real_values.extend(targets)

            self.predictions = torch.stack(predictions).cpu()
            self.prediction_probs = torch.stack(prediction_probs).cpu()
            self.real_values = torch.stack(real_values).cpu()

            print('Classification Report\n')
            print(classification_report(self.real_values, self.predictions,
                                        target_names=self.class_names))
        except AttributeError:
            print("Fit model at first!")
    
    # The most descriptive visualization for classifying models
    def show_confusion_matrix(self):
        try:    
            cm = confusion_matrix(self.real_values, self.predictions)
            df_cm = pd.DataFrame(cm, index=self.class_names, columns=self.class_names)

            hmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
            hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
            hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
            plt.ylabel('True rating')
            plt.xlabel('Predicted rating')
    
        except AttributeError:
            print("Fit model at first!")
    
    # This method shows us text from dataset and it's class probability
    def show_probability(self,index):
        try:
        
            pred_df = pd.DataFrame({'class_names': self.class_names,
                                    'values': self.prediction_probs[index]})

            print("\n".join(wrap(self.review_texts[index])))
            print()
            print(f'True rating: {self.class_names[self.real_values[index]]}')
            sns.barplot(x='values', y='class_names', data=pred_df, orient='h')
            plt.ylabel('rating')
            plt.xlabel('probability')
            plt.xlim([0, 1])
        
        except AttributeError:
            print("Fit model at first!")
    
    # Predicing the class of random text: 
    def predict(self,text):
        try:
            encoded_review = self.tokenizer.encode_plus(
              text,
              max_length=MAX_LEN,
              add_special_tokens=True,
              return_token_type_ids=False,
              pad_to_max_length=True,
              return_attention_mask=True,
              return_tensors='pt',
              padding='max_length',
              truncation=True     
            )

            input_ids = encoded_review['input_ids'].to(device)
            attention_mask = encoded_review['attention_mask'].to(device)

            output = self.model(input_ids, attention_mask)
            _, prediction = torch.max(output, dim=1)
        
        except AttributeError:
            print("Fit model at first!")
        
        return int(prediction) 
    
    # load early saved model parameters
    def load_model(self,file_name):
           
        self.model = TransformerModel(len(self.class_names))
        self.model.load_state_dict(torch.load(file_name,
                                   map_location=torch.device(device)))
        self.model = self.model.to(device)
    
    # This visualisation will help us in choosing the length to cut the sentense
    def show_sentenses_lengths(self,series,max_length = 500):

        token_lenghts = []

        for txt in series:
            tokens = self.tokenizer.encode(txt, max_length=max_length,
                                           truncation=True)
            token_lenghts.append(len(tokens))

        sns.displot(token_lenghts)
        plt.xlim([0, max_length + 100])
        plt.xlabel('Sentense length')   
