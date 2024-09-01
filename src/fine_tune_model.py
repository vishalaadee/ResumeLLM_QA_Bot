from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForQuestionAnswering
import json
import torch
import os

def fine_tune_model(training_data_path, model_output_dir):
    if not os.path.exists(training_data_path):
        raise FileNotFoundError(f"Training data file not found at {training_data_path}")
    
    try:
        with open(training_data_path, 'r') as file:
            training_data = json.load(file)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error reading JSON from {training_data_path}: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error loading training data: {e}")

    if not training_data:
        raise ValueError("Training data is empty. Please provide a valid dataset.")
    
    model_name = "bert-base-uncased"  
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    except Exception as e:
        raise Exception(f"Error loading model or tokenizer: {e}")

    def tokenize_data(examples):
        contexts = [ex['context'] for ex in examples]
        questions = [ex['question'] for ex in examples]
        
        try:
            encodings = tokenizer(
                contexts, 
                questions, 
                truncation=True, 
                padding='max_length', 
                max_length=512, 
                return_tensors='pt'
            )
        except Exception as e:
            raise Exception(f"Error during tokenization: {e}")
        
        try:
            start_positions = [ex['start_idx'] for ex in examples]
            end_positions = [ex['end_idx'] for ex in examples]
        except KeyError as e:
            raise KeyError(f"Missing key in training data: {e}")

        return encodings, start_positions, end_positions

    def preprocess_data(data):
        processed_data = []
        for item in data:
            try:
                context = item['context']
                question = item['question']
                answer = item['answer']
            except KeyError as e:
                raise KeyError(f"Missing key in data item: {e}")

            start_idx = context.find(answer)
            end_idx = start_idx + len(answer)
            if start_idx == -1:  
                start_idx = end_idx = 0
            
            processed_data.append({
                'context': context,
                'question': question,
                'start_idx': start_idx,
                'end_idx': end_idx
            })
        
        if not processed_data:
            raise ValueError("Processed data is empty after preprocessing.")
        
        return processed_data

    
    try:
        train_data = preprocess_data(training_data)
        tokenized_data, start_positions, end_positions = tokenize_data(train_data)
    except Exception as e:
        raise Exception(f"Error in data preparation: {e}")

    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, start_positions, end_positions):
            self.encodings = encodings
            self.start_positions = start_positions
            self.end_positions = end_positions
        
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['start_positions'] = torch.tensor(self.start_positions[idx])
            item['end_positions'] = torch.tensor(self.end_positions[idx])
            return item
        
        def __len__(self):
            return len(self.start_positions)
    
    dataset = CustomDataset(tokenized_data, start_positions, end_positions)
    
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=100,  
        evaluation_strategy="steps",
        save_total_limit=2  
    )
    
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset
        )
        trainer.train()
    except Exception as e:
        raise Exception(f"Error during training: {e}")

    try:
        model.save_pretrained(model_output_dir)
        tokenizer.save_pretrained(model_output_dir)
    except Exception as e:
        raise Exception(f"Error saving model and tokenizer: {e}")

if __name__ == "__main__":
    try:
        fine_tune_model('data/fine_tuning_data.json', 'models/fine_tuned_model')
    except Exception as e:
        print(f"An error occurred: {e}")
