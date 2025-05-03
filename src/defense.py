import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


## Defense Module

class EmailDataset(Dataset):
    """Dataset for emails"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        """Initialize dataset
        
        Args:
            texts: List of email texts
            labels: List of labels (0=legitimate, 1=phishing)
            tokenizer: Transformer tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class TransformerDefender:
    """Transformer-based phishing email detector"""
    
    def __init__(self, model_name='distilbert-base-uncased'):
        """Initialize transformer defender
        
        Args:
            model_name: Name of pre-trained transformer model
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2)
        self.model.to(device)
        
        # Default training parameters
        self.batch_size = 8
        self.max_length = 512
        self.learning_rate = 2e-5
        self.weight_decay = 0.01
    
    def train(self, train_df, val_df, epochs=3):
        """Train the model
        
        Args:
            train_df: Training dataframe with 'text' and 'label' columns
            val_df: Validation dataframe with 'text' and 'label' columns
            epochs: Number of training epochs
            
        Returns:
            Dictionary of training metrics
        """
        # Create datasets
        train_dataset = EmailDataset(
            train_df['text'].tolist(),
            train_df['label'].tolist(),
            self.tokenizer,
            self.max_length
        )
        
        val_dataset = EmailDataset(
            val_df['text'].tolist(),
            val_df['label'].tolist(),
            self.tokenizer,
            self.max_length
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f'./models/{self.model_name}-detector',
            num_train_epochs=epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=500,
            weight_decay=self.weight_decay,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False
        )
        
        # Metrics function
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            
            return {
                'accuracy': accuracy_score(labels, predictions),
                'precision': precision_score(labels, predictions),
                'recall': recall_score(labels, predictions),
                'f1': f1_score(labels, predictions)
            }
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            tokenizer=self.tokenizer
        )
        
        # Train the model
        train_output = trainer.train()
        
        # Evaluate
        eval_output = trainer.evaluate()
        
        return {
            'train_loss': train_output.metrics['train_loss'],
            'eval_loss': eval_output['eval_loss'],
            'eval_accuracy': eval_output['eval_accuracy'],
            'eval_precision': eval_output['eval_precision'],
            'eval_recall': eval_output['eval_recall'],
            'eval_f1': eval_output['eval_f1']
        }
    
    def predict(self, emails, threshold=0.5, calibrate=False):
        """Predict phishing probability for emails
        
        Args:
            emails: List of email dictionaries or texts
            threshold: Classification threshold
            calibrate: Whether to apply calibration
            
        Returns:
            List of dictionaries with predictions
        """
        # Extract text from email dictionaries if needed
        if isinstance(emails[0], dict):
            texts = [email['text'] for email in emails]
        else:
            texts = emails
        
        # Create dataset
        dataset = EmailDataset(
            texts,
            [0] * len(texts),  # Dummy labels
            self.tokenizer,
            self.max_length
        )
        
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        
        # Make predictions
        self.model.eval()
        all_probs = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move tensors to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Convert to probabilities
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())
        
        # Concatenate results
        all_probs = np.vstack(all_probs)
        phishing_probs = all_probs[:, 1]  # Probability of class 1 (phishing)
        
        # Apply calibration if requested
        if calibrate and hasattr(self, 'calibrator'):
            phishing_probs = self.calibrator.predict_proba(phishing_probs.reshape(-1, 1))[:, 1]
        
        # Create prediction results
        results = []
        for i, prob in enumerate(phishing_probs):
            is_phishing = prob >= threshold
            
            # Create result dictionary
            if isinstance(emails[0], dict):
                result = emails[i].copy()
            else:
                result = {'text': texts[i]}
                
            result['phishing_prob'] = float(prob)
            result['is_phishing'] = bool(is_phishing)
            result['detected'] = bool(is_phishing)  # For compatibility with attack results
            
            results.append(result)
        
        return results
    
    def calibrate(self, val_df):
        """Calibrate model probabilities
        
        Args:
            val_df: Validation dataframe with 'text' and 'label' columns
            
        Returns:
            Calibration performance metrics
        """
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.isotonic import IsotonicRegression
        
        # Get raw predictions on validation set
        val_preds = self.predict(val_df['text'].tolist(), threshold=0.5, calibrate=False)
        val_probs = np.array([pred['phishing_prob'] for pred in val_preds])
        val_labels = val_df['label'].values
        
        # Train isotonic regression calibrator
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(val_probs.reshape(-1, 1), val_labels)
        
        # Evaluate calibration
        calibrated_probs = self.calibrator.predict(val_probs.reshape(-1, 1))
        
        # Simple metrics for calibration quality
        original_brier = np.mean((val_probs - val_labels) ** 2)
        calibrated_brier = np.mean((calibrated_probs - val_labels) ** 2)
        
        return {
            'original_brier_score': original_brier,
            'calibrated_brier_score': calibrated_brier,
            'improvement': original_brier - calibrated_brier
        }
    
    def save(self, path):
        """Save model and tokenizer
        
        Args:
            path: Directory path to save to
        """
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Save calibrator if available
        if hasattr(self, 'calibrator'):
            import pickle
            with open(os.path.join(path, 'calibrator.pkl'), 'wb') as f:
                pickle.dump(self.calibrator, f)
    
    def load(self, path):
        """Load model and tokenizer
        
        Args:
            path: Directory path to load from
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.to(device)
        
        # Load calibrator if available
        calibrator_path = os.path.join(path, 'calibrator.pkl')
        if os.path.exists(calibrator_path):
            import pickle
            with open(calibrator_path, 'rb') as f:
                self.calibrator = pickle.load(f)


class HandcraftedFeatures:
    """Extract handcrafted features from emails for detection"""
    
    def __init__(self):
        """Initialize feature extractor"""
        # Common phishing keywords and phrases
        self.suspicious_keywords = [
            'account', 'update', 'verify', 'login', 'password', 'credit card',
            'bank', 'secure', 'urgent', 'important', 'alert', 'notification',
            'suspended', 'limited', 'access', 'click', 'link', 'confirm',
            'information', 'authenticate', 'unusual activity', 'security',
            'fraud', 'verify your identity', 'dear customer', 'expire', 'validate'
        ]
    
    def extract_features(self, emails):
        """Extract features from emails
        
        Args:
            emails: List of email dictionaries or texts
            
        Returns:
            Numpy array of features for each email
        """
        # Extract text from email dictionaries if needed
        if isinstance(emails[0], dict):
            texts = [email['text'] for email in emails]
        else:
            texts = emails
        
        features = []
        
        for text in texts:
            email_features = []
            
            # 1. URL counts
            url_count = len(re.findall(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', text))
            email_features.append(url_count)
            
            # 2. URL to text ratio
            text_length = len(text)
            url_ratio = url_count / max(1, text_length)
            email_features.append(url_ratio)
            
            # 3. Suspicious keyword count and ratio
            keyword_count = sum(1 for keyword in self.suspicious_keywords if keyword.lower() in text.lower())
            keyword_ratio = keyword_count / max(1, len(text.split()))
            email_features.append(keyword_count)
            email_features.append(keyword_ratio)
            
            # 4. Capitalization ratio
            caps_count = sum(1 for c in text if c.isupper())
            caps_ratio = caps_count / max(1, text_length)
            email_features.append(caps_ratio)
            
            # 5. Punctuation ratio
            punct_count = sum(1 for c in text if c in '!?.,;:\'\"()[]{}<>')
            punct_ratio = punct_count / max(1, text_length)
            email_features.append(punct_ratio)
            
            # 6. Word count
            word_count = len(text.split())
            email_features.append(word_count)
            
            # 7. Average word length
            avg_word_len = sum(len(word) for word in text.split()) / max(1, word_count)
            email_features.append(avg_word_len)
            
            # 8. HTML tag count
            html_tag_count = len(re.findall(r'<[^>]+>', text))
            email_features.append(html_tag_count)
            
            # 9. JavaScript presence
            js_presence = 1 if 'javascript:' in text.lower() else 0
            email_features.append(js_presence)
            
            # 10. Misspelling estimation (simplified)
            words = text.lower().split()
            common_words = {'the', 'and', 'to', 'of', 'a', 'in', 'is', 'that', 'for', 'you'}
            misspelled_estimate = sum(1 for word in words 
                                     if word not in common_words 
                                     and re.search(r'[^a-z]', word)
                                     and len(word) > 3)
            misspelled_ratio = misspelled_estimate / max(1, word_count)
            email_features.append(misspelled_ratio)
            
            features.append(email_features)
        
        return np.array(features)
    
    def get_feature_names(self):
        """Get names of extracted features
        
        Returns:
            List of feature names
        """
        return [
            'url_count',
            'url_ratio',
            'suspicious_keyword_count',
            'suspicious_keyword_ratio',
            'capitalization_ratio',
            'punctuation_ratio',
            'word_count',
            'avg_word_length',
            'html_tag_count',
            'javascript_presence',
            'misspelling_ratio'
        ]


class EnsembleDefender:
    """Ensemble of multiple defenders"""
    
    def __init__(self):
        """Initialize ensemble defender"""
        self.defenders = []
        self.weights = []
    
    def add_defender(self, defender, weight=1.0):
        """Add a defender to the ensemble
        
        Args:
            defender: Defender object with predict() method
            weight: Weight for this defender's predictions
        """
        self.defenders.append(defender)
        self.weights.append(weight)
    
    def normalize_weights(self):
        """Normalize weights to sum to 1.0"""
        total = sum(self.weights)
        if total > 0:
            self.weights = [w / total for w in self.weights]
    
    def predict(self, emails, threshold=0.5):
        """Predict phishing probability using the ensemble
        
        Args:
            emails: List of email dictionaries or texts
            threshold: Classification threshold
            
        Returns:
            List of dictionaries with predictions
        """
        if not self.defenders:
            raise ValueError("No defenders in the ensemble")
        
        # Normalize weights
        self.normalize_weights()
        
        # Get predictions from all defenders
        all_predictions = []
        for defender in self.defenders:
            preds = defender.predict(emails, threshold=0.0, calibrate=True)
            all_predictions.append(preds)
        
        # Combine predictions
        results = []
        for i in range(len(emails)):
            # Extract text from email if needed
            if isinstance(emails[i], dict):
                result = emails[i].copy()
            else:
                result = {'text': emails[i]}
            
            # Calculate weighted average probability
            weighted_prob = sum(
                preds[i]['phishing_prob'] * weight
                for preds, weight in zip(all_predictions, self.weights)
            )
            
            result['phishing_prob'] = weighted_prob
            result['is_phishing'] = weighted_prob >= threshold
            result['detected'] = weighted_prob >= threshold
            
            # Store individual model predictions
            result['model_probs'] = [preds[i]['phishing_prob'] for preds in all_predictions]
            
            results.append(result)
        
        return results
    
    def tune_weights(self, val_df):
        """Tune ensemble weights using validation data
        
        Args:
            val_df: Validation dataframe with 'text' and 'label' columns
            
        Returns:
            Dictionary with tuning metrics
        """
        from scipy.optimize import minimize
        
        # Get individual model predictions on validation set
        individual_preds = []
        for defender in self.defenders:
            preds = defender.predict(val_df['text'].tolist(), threshold=0.0, calibrate=True)
            probs = np.array([p['phishing_prob'] for p in preds])
            individual_preds.append(probs)
        
        individual_preds = np.column_stack(individual_preds)
        true_labels = val_df['label'].values
        
        # Define objective function to minimize (negative F1 score)
        def objective(weights):
            # Ensure weights are non-negative and sum to 1
            weights = np.maximum(weights, 0)
            weights = weights / np.sum(weights)
            
            # Calculate weighted predictions
            weighted_probs = individual_preds.dot(weights)
            predictions = (weighted_probs >= 0.5).astype(int)
            
            # Calculate F1 score
            f1 = f1_score(true_labels, predictions)
            return -f1  # Minimize negative F1
        
        # Initial weights (normalized)
        initial_weights = np.array(self.weights) / sum(self.weights)
        
        # Optimize weights
        bounds = [(0, 1) for _ in range(len(self.defenders))]
        result = minimize(
            objective,
            initial_weights,
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        # Update weights
        optimized_weights = result.x
        optimized_weights = np.maximum(optimized_weights, 0)
        optimized_weights = optimized_weights / np.sum(optimized_weights)
        self.weights = optimized_weights.tolist()
        
        # Calculate metrics with optimized weights
        weighted_probs = individual_preds.dot(optimized_weights)
        predictions = (weighted_probs >= 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(true_labels, predictions),
            'precision': precision_score(true_labels, predictions),
            'recall': recall_score(true_labels, predictions),
            'f1': f1_score(true_labels, predictions),
            'weights': optimized_weights.tolist()
        }
        
        return metrics


class AdaptiveDefender:
    """Defender that adapts to attacks over time"""
    
    def __init__(self, base_defender=None):
        """Initialize adaptive defender
        
        Args:
            base_defender: Base defender to start with
        """
        # Initialize with a transformer defender if none provided
        self.defender = base_defender or TransformerDefender()
        
        # Track attack history
        self.attack_history = []
        
        # Initialize memory buffer for online learning
        self.memory_buffer = {
            'texts': [],
            'labels': []
        }
        self.buffer_size = 1000  # Maximum examples to keep in memory
    
    def predict(self, emails, threshold=0.5):
        """Predict phishing probability
        
        Args:
            emails: List of email dictionaries or texts
            threshold: Classification threshold
            
        Returns:
            List of dictionaries with predictions
        """
        # Forward to the current defender
        return self.defender.predict(emails, threshold=threshold, calibrate=True)
    
    def update(self, attack_results, retrain=True):
        """Update defender based on attack results
        
        Args:
            attack_results: List of dictionaries with attack results
            retrain: Whether to retrain the model
            
        Returns:
            Update metrics
        """
        # Add results to attack history
        self.attack_history.extend(attack_results)
        
        # Add to memory buffer
        for result in attack_results:
            email = result['text'] if isinstance(result, dict) and 'text' in result else result['email']['text']
            # Phishing emails should have label 1
            label = 1 if 'label' in result and result['label'] == 1 else 1
            
            self.memory_buffer['texts'].append(email)
            self.memory_buffer['labels'].append(label)
        
        # Trim buffer if it gets too large (keep most recent examples)
        if len(self.memory_buffer['texts']) > self.buffer_size:
            self.memory_buffer['texts'] = self.memory_buffer['texts'][-self.buffer_size:]
            self.memory_buffer['labels'] = self.memory_buffer['labels'][-self.buffer_size:]
        
        # Retrain if requested
        if retrain and len(self.memory_buffer['texts']) >= 50:
            # Create a small dataset for fine-tuning
            df = pd.DataFrame({
                'text': self.memory_buffer['texts'],
                'label': self.memory_buffer['labels']
            })
            
            # Split into train/val
            train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
            
            # Fine-tune the model
            metrics = self.defender.train(train_df, val_df, epochs=1)
            
            # Re-calibrate
            self.defender.calibrate(val_df)
            
            return {
                'update_size': len(attack_results),
                'memory_buffer_size': len(self.memory_buffer['texts']),
                'retrain_metrics': metrics
            }
        
        return {
            'update_size': len(attack_results),
            'memory_buffer_size': len(self.memory_buffer['texts']),
            'retrain': False
        }

