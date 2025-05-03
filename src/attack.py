import random
import numpy as np
import torch
import nltk
from nltk.corpus import wordnet
from transformers import AutoTokenizer, AutoModelForSequenceClassification

## Attack Module

class TemplateGenerator:
    """Generate phishing emails from templates"""
    
    def __init__(self):
        self.templates = [
            # Amazon templates
            {
                "subject": "Amazon: Action Required - Update Payment Information",
                "body": "Dear Amazon Customer,\n\nWe need you to update your payment information for your recent purchase. "
                        "If you do not update within 24 hours, your order will be canceled.\n\n"
                        "Please click here to update: {url}\n\n"
                        "Amazon Customer Service"
            },
            {
                "subject": "Your Amazon order has shipped",
                "body": "Hello,\n\nYour order #{order_num} has shipped and will be delivered on {date}.\n\n"
                        "However, we noticed an issue with your payment method. Please verify your information: {url}\n\n"
                        "Amazon Shipping Team"
            },
            
            # Banking templates
            {
                "subject": "Important: Your account access has been limited",
                "body": "Dear {bank} Customer,\n\nWe have temporarily limited access to your account due to "
                        "failed login attempts. To restore full access, please verify your identity: {url}\n\n"
                        "Security Department, {bank}"
            },
            {
                "subject": "Security Alert: Unusual Activity Detected",
                "body": "Dear {name},\n\nWe detected unusual activity on your {bank} account on {date}. "
                        "If this was not you, please secure your account immediately: {url}\n\n"
                        "Thank you,\n{bank} Fraud Prevention Team"
            },
            
            # HR/Corporate templates
            {
                "subject": "Urgent: Update your company credentials",
                "body": "Dear {name},\n\nDue to recent security upgrades, all employees are required to update "
                        "their login credentials by end of day. Click here to update: {url}\n\n"
                        "IT Department"
            },
            {
                "subject": "Important: New company policy document",
                "body": "All staff,\n\nA new company policy regarding remote work has been published. "
                        "All employees must read and acknowledge receipt by tomorrow.\n\n"
                        "Download the document here: {url}\n\n"
                        "Human Resources"
            },
            
            # Tax/Government templates
            {
                "subject": "IRS: Tax Refund Notification",
                "body": "Tax Refund Notice #{notice_num}\n\nDear Taxpayer,\n\nAfter the annual calculation of your fiscal activity, "
                        "we have determined that you are eligible for a refund of ${amount}.\n\n"
                        "Submit your refund request here: {url}\n\n"
                        "Internal Revenue Service"
            },
            {
                "subject": "Action Required: Government Stimulus Payment",
                "body": "NOTICE: Economic Impact Payment\n\nYou have qualified for a government stimulus payment of ${amount}. "
                        "To receive your payment, please confirm your information: {url}\n\n"
                        "Department of the Treasury"
            }
        ]
        
        self.names = ["John Smith", "Mary Johnson", "Robert Williams", "Patricia Brown", "Michael Davis"]
        self.banks = ["Chase", "Bank of America", "Wells Fargo", "Citibank", "Capital One"]
        self.dates = ["May 15, 2023", "June 22, 2023", "July 8, 2023", "August 30, 2023"]
        self.amounts = ["1,247.63", "958.29", "2,361.45", "785.12", "1,503.87"]
        self.order_nums = ["A23B7C", "X92Y14", "L67M39", "P45Q81", "R72S05"]
        self.notice_nums = ["CP-1234", "RF-5678", "TX-9012", "IR-3456", "GV-7890"]
        
        # Phishing URLs for templates
        self.urls = [
            "http://amazonn-secure.com/verify",
            "http://security-bankaccess.net/login",
            "http://company-portal.co/document",
            "http://tax-refund-secure.com/claim",
            "http://accountverify-secure.com/auth"
        ]
    
    def generate(self, n_samples=10, use_template_idx=None):
        """Generate phishing emails from templates
        
        Args:
            n_samples: Number of emails to generate
            use_template_idx: If specified, use only this template index
            
        Returns:
            List of generated phishing emails
        """
        generated_emails = []
        
        for _ in range(n_samples):
            if use_template_idx is not None and use_template_idx < len(self.templates):
                template = self.templates[use_template_idx]
            else:
                template = random.choice(self.templates)
            
            # Fill in template placeholders
            body = template["body"].format(
                name=random.choice(self.names),
                bank=random.choice(self.banks),
                date=random.choice(self.dates),
                amount=random.choice(self.amounts),
                order_num=random.choice(self.order_nums),
                notice_num=random.choice(self.notice_nums),
                url=random.choice(self.urls)
            )
            
            generated_emails.append({
                "text": body,
                "label": 1,  # 1 indicates phishing
                "template_id": self.templates.index(template)
            })
        
        return generated_emails


class PerturbationEngine:
    """Apply various perturbation techniques to phishing emails to evade detection"""
    
    def __init__(self):
        # Download NLTK resources if not already available
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)
        
        # Character-level perturbations
        self.char_perturbations = [
            self._swap_chars,
            self._add_char,
            self._remove_char,
            self._similar_char_replacement
        ]
        
        # Word-level perturbations
        self.word_perturbations = [
            self._synonym_replacement,
            self._word_insertion,
            self._word_deletion
        ]
        
        # Style perturbations
        self.style_perturbations = [
            self._change_case,
            self._add_html_formatting,
            self._add_unicode_chars
        ]
        
        # URL hiding techniques
        self.url_perturbations = [
            self._hide_url_in_text,
            self._url_shortener_simulation,
            self._html_url_obfuscation
        ]
        
        # Similar-looking character mappings
        self.char_map = {
            'a': ['а', '@', '4'],  # Cyrillic 'а' looks like Latin 'a'
            'b': ['b', '6', 'б'],
            'c': ['с', '('],  # Cyrillic 'с' looks like Latin 'c'
            'e': ['е', '3'],  # Cyrillic 'е' looks like Latin 'e'
            'i': ['і', '1', '!'],
            'l': ['l', '1', '|'],
            'o': ['о', '0'],  # Cyrillic 'о' looks like Latin 'o'
            's': ['ѕ', '5', '$'],
            't': ['т', '+'],  # Cyrillic 'т' looks like Latin 't'
            'w': ['vv', 'ѡ'],
            'g': ['g', '9'],
            'r': ['r', 'г'],
            'n': ['n', 'п'],
            'm': ['m', 'м']
        }
    
    def perturb(self, email, technique=None, intensity=0.1):
        """Apply perturbations to an email
        
        Args:
            email: Dictionary containing email text
            technique: Specific technique to use, or None for random selection
            intensity: Perturbation intensity (0.0 to 1.0)
            
        Returns:
            Dictionary with perturbed email text
        """
        email_text = email["text"]
        
        # Choose perturbation technique if not specified
        if technique is None:
            all_techniques = (self.char_perturbations + self.word_perturbations + 
                             self.style_perturbations + self.url_perturbations)
            technique = random.choice(all_techniques)
        
        # Apply selected perturbation
        perturbed_text = technique(email_text, intensity)
        
        # Create perturbed email
        perturbed_email = email.copy()
        perturbed_email["text"] = perturbed_text
        perturbed_email["perturbation"] = technique.__name__
        
        return perturbed_email
    
    def _swap_chars(self, text, intensity):
        """Swap adjacent characters in words"""
        words = text.split()
        new_words = []
        
        for word in words:
            if len(word) <= 1 or random.random() > intensity:
                new_words.append(word)
                continue
                
            # Choose a random position for swapping
            pos = random.randint(0, len(word) - 2)
            chars = list(word)
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
            new_words.append(''.join(chars))
                
        return ' '.join(new_words)
    
    def _add_char(self, text, intensity):
        """Add extra characters to words"""
        words = text.split()
        new_words = []
        
        for word in words:
            if len(word) == 0 or random.random() > intensity:
                new_words.append(word)
                continue
                
            # Insert a random character at a random position
            pos = random.randint(0, len(word))
            char = random.choice('abcdefghijklmnopqrstuvwxyz')
            new_word = word[:pos] + char + word[pos:]
            new_words.append(new_word)
                
        return ' '.join(new_words)
    
    def _remove_char(self, text, intensity):
        """Remove characters from words"""
        words = text.split()
        new_words = []
        
        for word in words:
            if len(word) <= 1 or random.random() > intensity:
                new_words.append(word)
                continue
                
            # Remove a random character
            pos = random.randint(0, len(word) - 1)
            new_word = word[:pos] + word[pos + 1:]
            new_words.append(new_word)
                
        return ' '.join(new_words)
    
    def _similar_char_replacement(self, text, intensity):
        """Replace characters with similar-looking ones"""
        new_text = ""
        
        for char in text:
            lower_char = char.lower()
            if lower_char in self.char_map and random.random() < intensity:
                replacements = self.char_map[lower_char]
                new_char = random.choice(replacements)
                new_text += new_char
            else:
                new_text += char
                
        return new_text
    
    def _synonym_replacement(self, text, intensity):
        """Replace words with synonyms"""
        words = nltk.word_tokenize(text)
        new_words = []
        
        for word in words:
            if len(word) <= 3 or not word.isalpha() or random.random() > intensity:
                new_words.append(word)
                continue
            
            # Find synonyms using WordNet
            synonyms = []
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    if lemma.name() != word.lower():
                        synonyms.append(lemma.name())
            
            if len(synonyms) > 0:
                new_word = random.choice(synonyms).replace('_', ' ')
                new_words.append(new_word)
            else:
                new_words.append(word)
                
        return ' '.join(new_words)
    
    def _word_insertion(self, text, intensity):
        """Insert benign words into text"""
        benign_words = ["please", "kindly", "important", "notification", "information", 
                        "update", "confirm", "verify", "secure", "official"]
        
        words = text.split()
        new_words = []
        
        for word in words:
            new_words.append(word)
            if random.random() < intensity:
                new_words.append(random.choice(benign_words))
                
        return ' '.join(new_words)
    
    def _word_deletion(self, text, intensity):
        """Delete some words from text"""
        words = text.split()
        new_words = []
        
        for word in words:
            if random.random() > intensity:
                new_words.append(word)
                
        return ' '.join(new_words)
    
    def _change_case(self, text, intensity):
        """Change case of words or characters"""
        words = text.split()
        new_words = []
        
        for word in words:
            if random.random() < intensity:
                if random.random() < 0.5:
                    # Change to uppercase
                    new_word = word.upper()
                else:
                    # Randomize capitalization
                    new_word = ''.join([c.upper() if random.random() < 0.5 else c.lower() for c in word])
                new_words.append(new_word)
            else:
                new_words.append(word)
                
        return ' '.join(new_words)
    
    def _add_html_formatting(self, text, intensity):
        """Add simple HTML formatting to text"""
        if random.random() < intensity:
            # Add basic HTML tags
            html_tags = [
                (f"<div style='font-family: Arial;'>{text}</div>"),
                (f"<p>{text}</p>"),
                (f"<span style='color: #000000;'>{text}</span>"),
                (f"<div style='text-align: left;'>{text}</div>"),
                (f"<font face='Arial'>{text}</font>")
            ]
            return random.choice(html_tags)
        return text
    
    def _add_unicode_chars(self, text, intensity):
        """Add invisible or zero-width unicode characters"""
        # Zero-width characters
        zwc = ['\u200B', '\u200C', '\u200D', '\u200E', '\u200F', '\u2060', '\u2061', '\u2062', '\u2063', '\u2064']
        
        if random.random() < intensity:
            # Insert ZWCs at random positions
            chars = list(text)
            num_insertions = max(1, int(len(text) * intensity * 0.1))
            
            for _ in range(num_insertions):
                pos = random.randint(0, len(chars))
                chars.insert(pos, random.choice(zwc))
                
            return ''.join(chars)
        return text
    
    def _hide_url_in_text(self, text, intensity):
        """Replace URLs with text-based hyperlinks"""
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        urls = re.findall(url_pattern, text)
        
        if not urls or random.random() > intensity:
            return text
        
        for url in urls:
            mask_texts = [
                "Click here to secure your account",
                "Verify your information here",
                "Access your account",
                "Login now",
                "Secure login portal"
            ]
            mask = random.choice(mask_texts)
            text = text.replace(url, mask)
            
        return text
    
    def _url_shortener_simulation(self, text, intensity):
        """Replace URLs with simulated shortened URLs"""
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        urls = re.findall(url_pattern, text)
        
        if not urls or random.random() > intensity:
            return text
        
        for url in urls:
            shorteners = [
                "bit.ly/secure-login",
                "tinyurl.com/account-verify",
                "goo.gl/auth-portal",
                "t.co/secure-access",
                "is.gd/verify-now"
            ]
            shortened_url = random.choice(shorteners)
            text = text.replace(url, shortened_url)
            
        return text
    
    def _html_url_obfuscation(self, text, intensity):
        """Use HTML to obfuscate URLs"""
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        urls = re.findall(url_pattern, text)
        
        if not urls or random.random() > intensity:
            return text
        
        for url in urls:
            obfuscation_methods = [
                f"<a href='{url}'>secure verification link</a>",
                f"<a href='{url}' style='color:blue; text-decoration:underline;'>click here</a>",
                f"<a href='{url}'><img src='secure-badge.png' alt='Secure Link'></a>"
            ]
            obfuscated_url = random.choice(obfuscation_methods)
            text = text.replace(url, obfuscated_url)
            
        return text
    

class AdaptiveAttacker:
    """Learning-based attacker that adapts to defender's behavior"""
    
    def __init__(self, base_generator=None, base_perturbation=None):
        """Initialize adaptive attacker
        
        Args:
            base_generator: TemplateGenerator instance to generate base emails
            base_perturbation: PerturbationEngine instance to apply perturbations
        """
        self.template_generator = base_generator or TemplateGenerator()
        self.perturbation_engine = base_perturbation or PerturbationEngine()
        
        # Track successful and failed attack patterns
        self.successful_patterns = []  # Patterns that evaded detection
        self.failed_patterns = []      # Patterns that were detected
        
        # Initialize a simple neural language model for adaptive attacks
        # For the Kaggle notebook, we'll use a simplified model
        self.initialize_model()
        
    def initialize_model(self):
        """Initialize a simplified language model for adaptive attacks"""
        # For demonstration, we'll use a simple model
        # In a real implementation, this could be GPT-2 or similar
        
        try:
            # Load a small pre-trained model
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased", num_labels=2)
            self.model.to(device)
            
            # Initialize with random weights - this is just for demonstration
            for param in self.model.parameters():
                param.data = torch.randn_like(param.data) * 0.01
                
            print("Initialized adaptive attacker model")
        except Exception as e:
            print(f"Error initializing model: {e}")
            self.tokenizer = None
            self.model = None
            print("Using template-based attacks only")
    
    def generate_attack(self, n_samples=10, strategy=None):
        """Generate attack emails
        
        Args:
            n_samples: Number of attack emails to generate
            strategy: Attack strategy (template, perturbation, adaptive, mixed)
            
        Returns:
            List of attack emails
        """
        if strategy is None:
            # Choose a strategy based on past successes
            if len(self.successful_patterns) > 0 and random.random() < 0.7:
                # Use successful strategies more often
                strategy = random.choice([p['strategy'] for p in self.successful_patterns])
            else:
                strategy = random.choice(['template', 'perturbation', 'adaptive', 'mixed'])
        
        if strategy == 'template':
            # Simple template-based attack
            return self.template_generator.generate(n_samples)
            
        elif strategy == 'perturbation':
            # Apply perturbations to template emails
            base_emails = self.template_generator.generate(n_samples)
            perturbed_emails = []
            
            for email in base_emails:
                # Choose a random perturbation technique
                perturbed = self.perturbation_engine.perturb(email, intensity=random.uniform(0.1, 0.5))
                perturbed['strategy'] = 'perturbation'
                perturbed_emails.append(perturbed)
                
            return perturbed_emails
            
        elif strategy == 'adaptive':
            # Use language model to generate variations
            if self.model is None:
                # Fall back to perturbation if model isn't available
                return self.generate_attack(n_samples, 'perturbation')
            
            base_emails = self.template_generator.generate(n_samples)
            adaptive_emails = []
            
            for email in base_emails:
                # This is a simplified implementation
                # In a real system, you would fine-tune the model on successful evasions
                
                # Apply a random sequence of perturbations
                perturbed = email.copy()
                num_perturbations = random.randint(1, 3)
                
                for _ in range(num_perturbations):
                    # Avoid perturbations that frequently fail
                    fail_funcs = [p['perturbation'] for p in self.failed_patterns[-10:]] if self.failed_patterns else []
                    
                    all_perturbations = (self.perturbation_engine.char_perturbations + 
                                         self.perturbation_engine.word_perturbations + 
                                         self.perturbation_engine.style_perturbations + 
                                         self.perturbation_engine.url_perturbations)
                    
                    # Filter out frequently failing perturbations if we have enough data
                    if len(fail_funcs) >= 5:
                        available_perturbations = [f for f in all_perturbations 
                                               if f.__name__ not in fail_funcs[:5]]
                    else:
                        available_perturbations = all_perturbations
                    
                    if not available_perturbations:
                        available_perturbations = all_perturbations
                        
                    technique = random.choice(available_perturbations)
                    perturbed = self.perturbation_engine.perturb(
                        perturbed, technique=technique, intensity=random.uniform(0.1, 0.5))
                
                perturbed['strategy'] = 'adaptive'
                adaptive_emails.append(perturbed)
                
            return adaptive_emails
            
        elif strategy == 'mixed':
            # Mix different attack strategies
            attacks_per_strategy = n_samples // 3
            remainder = n_samples % 3
            
            template_attacks = self.generate_attack(attacks_per_strategy, 'template')
            perturb_attacks = self.generate_attack(attacks_per_strategy, 'perturbation')
            adaptive_attacks = self.generate_attack(attacks_per_strategy + remainder, 'adaptive')
            
            mixed_attacks = template_attacks + perturb_attacks + adaptive_attacks
            random.shuffle(mixed_attacks)
            
            return mixed_attacks
        
        else:
            raise ValueError(f"Unknown attack strategy: {strategy}")
    
    def update_model(self, attack_results):
        """Update model based on attack results
        
        Args:
            attack_results: List of dictionaries with attack results
                Each dict should have 'email' and 'detected' keys
        """
        # Add attack patterns to our history
        for result in attack_results:
            email = result['email']
            was_detected = result['detected']
            
            pattern = {
                'text': email['text'],
                'strategy': email.get('strategy', 'unknown'),
                'perturbation': email.get('perturbation', None)
            }
            
            if was_detected:
                self.failed_patterns.append(pattern)
            else:
                self.successful_patterns.append(pattern)
        
        # Skip fine-tuning for the Kaggle notebook demo
        # In a real implementation, you would fine-tune the model here
        print(f"Updated attacker model with {len(attack_results)} results")
        print(f"Successful evasions: {sum(1 for r in attack_results if not r['detected'])}")


class GameTheoreticPlanner:
    """Plan attacks using game theory"""
    
    def __init__(self, attack_strategies=None):
        """Initialize game-theoretic attack planner
        
        Args:
            attack_strategies: List of available attack strategies
        """
        self.attack_strategies = attack_strategies or [
            'template', 'perturbation', 'adaptive', 'mixed'
        ]
        
        # Payoff matrix: strategy x strategy -> utility
        self.payoff_matrix = {
            strategy: {defense: 0.5 for defense in self.attack_strategies}
            for strategy in self.attack_strategies
        }
        
        # Initialize strategy distribution (uniform)
        self.strategy_distribution = {
            strategy: 1.0 / len(self.attack_strategies)
            for strategy in self.attack_strategies
        }
    
    def update_payoffs(self, attack_results):
        """Update payoff matrix based on attack results
        
        Args:
            attack_results: List of dictionaries with attack results
                Each should have 'email' and 'detected' keys
        """
        # Group results by strategy
        strategy_results = {}
        
        for result in attack_results:
            strategy = result['email'].get('strategy', 'unknown')
            if strategy not in strategy_results:
                strategy_results[strategy] = []
            strategy_results[strategy].append(result['detected'])
        
        # Update payoff matrix
        for strategy, results in strategy_results.items():
            if not results:
                continue
                
            # Payoff is the evasion rate (1 - detection rate)
            detection_rate = sum(results) / len(results)
            evasion_rate = 1 - detection_rate
            
            # Update payoff for all defense strategies (simplified)
            # In a real implementation, you would model the defender's strategies
            for defense in self.attack_strategies:
                # Discount old payoff and add new observation
                alpha = 0.3  # Learning rate
                old_payoff = self.payoff_matrix[strategy][defense]
                new_payoff = (1 - alpha) * old_payoff + alpha * evasion_rate
                self.payoff_matrix[strategy][defense] = new_payoff
    
    def solve_mixed_strategy(self):
        """Solve for Nash equilibrium mixed strategy
        
        This is a simplified implementation using fictitious play.
        For a real implementation, you might use linear programming.
        
        Returns:
            Dictionary mapping strategies to probabilities
        """
        # Calculate expected payoffs for each strategy
        expected_payoffs = {}
        
        for attack_strategy in self.attack_strategies:
            # Assume defender uses same distribution (zero-sum game)
            payoff = sum(
                self.payoff_matrix[attack_strategy][defense] * self.strategy_distribution[defense]
                for defense in self.attack_strategies
            )
            expected_payoffs[attack_strategy] = payoff
        
        # Find best-response strategy
        best_payoff = max(expected_payoffs.values())
        best_strategies = [
            strategy for strategy, payoff in expected_payoffs.items()
            if abs(payoff - best_payoff) < 1e-6
        ]
        
        # Update strategy distribution (softmax)
        temperature = 0.1  # Exploration parameter
        exps = {
            strategy: np.exp(expected_payoffs[strategy] / temperature)
            for strategy in self.attack_strategies
        }
        total_exp = sum(exps.values())
        
        self.strategy_distribution = {
            strategy: exp / total_exp
            for strategy, exp in exps.items()
        }
        
        return self.strategy_distribution
    
    def choose_strategy(self):
        """Choose a strategy based on current distribution
        
        Returns:
            String name of chosen strategy
        """
        strategies = list(self.strategy_distribution.keys())
        probabilities = list(self.strategy_distribution.values())
        
        return np.random.choice(strategies, p=probabilities)

