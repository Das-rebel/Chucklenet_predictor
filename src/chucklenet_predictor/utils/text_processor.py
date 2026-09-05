"""
Text Processor for Chucklenet Predictor

This module provides comprehensive text processing capabilities including:
- Feature extraction for humor analysis
- Linguistic analysis
- Syntactic analysis
- Semantic analysis
- Humor type classification

Author: Subho Das
"""

import re
import string
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

logger = logging.getLogger(__name__)


class TextProcessor:
    """
    Comprehensive text processing for humor analysis.
    """
    
    def __init__(self, 
                 humor_patterns: Optional[Dict] = None,
                 use_pretrained: bool = True,
                 model_name: str = "roberta-base"):
        """
        Initialize text processor.
        
        Args:
            humor_patterns: Custom humor patterns and regex
            use_pretrained: Whether to use pretrained model
            model_name: Name of pretrained model
        """
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Custom humor patterns
        self.humor_patterns = humor_patterns or self._default_humor_patterns()
        
        # Initialize models
        self.use_pretrained = use_pretrained
        if use_pretrained:
            try:
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    tokenizer=model_name,
                    device=0 if torch.cuda.is_available() else -1
                )
                self.emotion_analyzer = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    device=0 if torch.cuda.is_available() else -1
                )
                logger.info(f"Loaded pretrained models: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load pretrained models: {e}")
                self.sentiment_analyzer = None
                self.emotion_analyzer = None
        
        # Feature vectorizers
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100)
        self.count_vectorizer = CountVectorizer(max_features=50)
    
    def _default_humor_patterns(self) -> Dict:
        """Default humor patterns and regex."""
        return {
            'puns': {
                'patterns': [
                    r'\b(why|what|how|when|where|who)\s+\w+\s+(is|are|was|were)\s+\w+\b',
                    r'\b(\w+)\s+(\w+)\s+(\w+)\s+\1\b',
                    r'\b(\w+)\s+(\w+)\s+\1\b'
                ],
                'keywords': ['punny', 'pun', 'wordplay', 'double entendre']
            },
            'sarcasm': {
                'patterns': [
                    r'\b(sure|yeah|right|totally|absolutely)\s+(but|however|but really)',
                    r'\b(great|wonderful|amazing)\s+(not|really|actually)',
                    r'\b(obviously|clearly|apparently)\s+(not|never)'
                ],
                'keywords': ['sarcasm', 'sarcastic', 'ironic', 'mocking']
            },
            'absurd': {
                'patterns': [
                    r'\b(talking|walking|swimming)\s+(to|with|at)\s+\w+\s+(chair|plant|lamp)',
                    r'\b(\w+)\s+(eats|drinks|sleeps|flies)\s+(paper|rock|air|thoughts)',
                    r'\b(the|a)\s+\w+\s+(\w+)\s+(\w+)\s+(is|are)\s+(\w+)\s+(\w+)\s+(\w+)'
                ],
                'keywords': ['absurd', 'ridiculous', 'nonsense', 'illogical']
            },
            'irony': {
                'patterns': [
                    r'\b(exactly|perfect|ideal)\s+(except|but|except for)',
                    r'\b(great|awesome|fantastic)\s+(except|but|except for)',
                    r'\b(not exactly|quite|sort of)\s+(but|however)'
                ],
                'keywords': ['irony', 'ironic', 'paradox', 'contradiction']
            },
            'slapstick': {
                'patterns': [
                    r'\b(fell|tripped|slipped|stumbled|bumped|crashed)',
                    r'\b(hit|smacked|bonked|whacked|slapped|punched)',
                    r'\b(slippery|wet|oily|icy|uneven)\s+(floor|ground|path)'
                ],
                'keywords': ['slapstick', 'physical', 'comedy', 'fall', 'crash']
            },
            'surprise': {
                'patterns': [
                    r'\b(suddenly|unexpectedly|out of nowhere|boom|bam)',
                    r'\b(to my\s+surprise|i\s+can\'t\s+believe|imagine\s+that)',
                    r'\b(who\s+knew|never\s+saw\s+that|not\s+what\s+i\s+expected)'
                ],
                'keywords': ['surprise', 'unexpected', 'unexpectedly', 'shock']
            },
            'relatable': {
                'patterns': [
                    r'\b(everyone\s+knows|we\s+all\s+know|you\s+know\s+that\s+feeling)',
                    r'\b(like\s+me|same\s+here|can\s+relate)',
                    r'\b(remember\s+when|isn\'t\s+it\s+true|don\'t\s+you\s+hate\s+when)'
                ],
                'keywords': ['relatable', 'common', 'universal', 'shared']
            },
            'exaggeration': {
                'patterns': [
                    r'\b(never|always|everyone|nobody|nothing|everything)',
                    r'\b(\w+)\s+million|\w+\s+billion|\w+\s+trillion',
                    r'\b(the\s+biggest|the\s+most|the\s+very|extremely|super|ultra)'
                ],
                'keywords': ['exaggeration', 'hyperbole', 'overstatement', 'extreme']
            },
            'understatement': {
                'patterns': [
                    r'\b(a\s+little|somewhat|slightly|kinda|sorta|maybe)',
                    r'\b(not\s+bad|pretty\s+good|alright|decent)',
                    r'\b(small|minor|slight|minor|tiny)\s+(problem|issue|concern)'
                ],
                'keywords': ['understatement', 'downplay', 'understate', 'minimize']
            },
            'wordplay': {
                'patterns': [
                    r'\b(homophone|pun|play\s+on\s+words|double\s+entendre)',
                    r'\b(\w+)\s+(\w+)\s+(\w+)\s+(\w+)\s+(\w+)\b',
                    r'\b(same|similar|sounds\s+like|resembles)\s+\w+'
                ],
                'keywords': ['wordplay', 'pun', 'clever', 'witty', 'smart']
            },
            'setup_punchline': {
                'patterns': [
                    r'\b(why|what|how|when|where)\s+\w+\s+(so|therefore|thus|and)',
                    r'\b(a\s+man|a\s+woman|an\s+old\s+man|a\s+priest)\s+walks\s+into',
                    r'\b(suddenly|then\s+finally|after\s+a\s+moment)\s+\w+'
                ],
                'keywords': ['setup', 'punchline', 'joke', 'funny', 'comedy']
            },
            'abrupt_endings': {
                'patterns': [
                    r'\b(suddenly\s+boom|and\s+then\s+nothing|cut\s+to\s+black)',
                    r'\b(the\s+end\.|fade\s+to\s+black\.|roll\s+credits\.)',
                    r'\b(\.\.\.that\'s\s+it|done|finished|over)'
                ],
                'keywords': ['abrupt', 'sudden', 'unexpected', 'quick', 'short']
            },
            'pauses': {
                'patterns': [
                    r'\b(\.\.\.|-|—|—\s+—|\s+—\s+—)',
                    r'\b(um|uh|er|ah|well|so|actually)',
                    r'\b(let\s+me\s+think|i\s+mean|in\s+other\s+words)'
                ],
                'keywords': ['pause', 'hesitation', 'silence', 'break', 'wait']
            }
        }
    
    def extract_all_features(self, text: str) -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
        """
        Extract all features from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing all extracted features
        """
        features = {
            'basic': self._extract_basic_features(text),
            'syntactic': self._extract_syntactic_features(text),
            'semantic': self._extract_semantic_features(text),
            'humor_types': self._analyze_humor_types(text),
            'pattern_matching': self._match_humor_patterns(text),
            'emotional': self._analyze_emotional_content(text),
            'linguistic': self._extract_linguistic_features(text),
            'sentiment': self._analyze_sentiment(text)
        }
        
        return features
    
    def _extract_basic_features(self, text: str) -> Dict[str, float]:
        """Extract basic text features."""
        text = str(text).lower()
        
        return {
            'char_count': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(sent_tokenize(text)),
            'paragraph_count': len(text.split('\n')),
            'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
            'avg_sentence_length': np.mean([len(sent.split()) for sent in sent_tokenize(text)]) if sent_tokenize(text) else 0,
            'avg_paragraph_length': np.mean([len(paragraph.split()) for paragraph in text.split('\n')]) if text.split('\n') else 0,
            'reading_time': len(text.split()) / 200,  # Average reading speed: 200 wpm
            'unique_words': len(set(text.split())),
            'lexical_diversity': len(set(text.split())) / len(text.split()) if text.split() else 0,
            'punctuation_count': sum(1 for char in text if char in string.punctuation),
            'punctuation_ratio': sum(1 for char in text if char in string.punctuation) / len(text) if text else 0,
            'uppercase_ratio': sum(1 for char in text if char.isupper()) / len(text) if text else 0,
            'digit_count': sum(1 for char in text if char.isdigit()),
            'digit_ratio': sum(1 for char in text if char.isdigit()) / len(text) if text else 0,
            'whitespace_count': text.count(' '),
            'newline_count': text.count('\n'),
            'tab_count': text.count('\t'),
        }
    
    def _extract_syntactic_features(self, text: str) -> Dict[str, float]:
        """Extract syntactic features."""
        try:
            sentences = sent_tokenize(text)
            words = word_tokenize(text.lower())
            
            # Count different parts of speech (simplified)
            pos_tags = {}
            for word, pos in nltk.pos_tag(words):
                pos_tags[pos] = pos_tags.get(pos, 0) + 1
            
            total_words = len(words)
            return {
                'total_words': total_words,
                'total_sentences': len(sentences),
                'avg_words_per_sentence': total_words / len(sentences) if sentences else 0,
                'nouns': pos_tags.get('NN', 0) + pos_tags.get('NNS', 0) + pos_tags.get('NNP', 0) + pos_tags.get('NNPS', 0),
                'verbs': pos_tags.get('VB', 0) + pos_tags.get('VBD', 0) + pos_tags.get('VBG', 0) + pos_tags.get('VBN', 0) + pos_tags.get('VBP', 0) + pos_tags.get('VBZ', 0),
                'adjectives': pos_tags.get('JJ', 0) + pos_tags.get('JJR', 0) + pos_tags.get('JJS', 0),
                'adverbs': pos_tags.get('RB', 0) + pos_tags.get('RBR', 0) + pos_tags.get('RBS', 0),
                'pronouns': pos_tags.get('PRP', 0) + pos_tags.get('PRP$', 0),
                'prepositions': pos_tags.get('IN', 0),
                'conjunctions': pos_tags.get('CC', 0),
                'interjections': pos_tags.get('UH', 0),
                'question_marks': text.count('?'),
                'exclamation_marks': text.count('!'),
                'question_ratio': text.count('?') / len(sentences) if sentences else 0,
                'exclamation_ratio': text.count('!') / len(sentences) if sentences else 0,
                'comma_count': text.count(','),
                'semicolon_count': text.count(';'),
                'colon_count': text.count(':'),
                'dash_count': text.count('-') + text.count('—') + text.count('–'),
                'parenthesis_count': text.count('(') + text.count(')'),
                'quote_count': text.count('"') + text.count("'"),
                'complex_words': sum(1 for word in words if len(word) > 7),
                'complex_word_ratio': sum(1 for word in words if len(word) > 7) / total_words if total_words else 0,
                'long_sentences': sum(1 for sent in sentences if len(sent.split()) > 20),
                'long_sentence_ratio': sum(1 for sent in sentences if len(sent.split()) > 20) / len(sentences) if sentences else 0,
            }
        except Exception as e:
            logger.warning(f"Error extracting syntactic features: {e}")
            return {}
    
    def _extract_semantic_features(self, text: str) -> Dict[str, float]:
        """Extract semantic features."""
        try:
            # Word tokenization and lemmatization
            words = word_tokenize(text.lower())
            lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in self.stop_words]
            
            # Basic semantic metrics
            word_count = len(lemmatized_words)
            unique_words = len(set(lemmatized_words))
            
            return {
                'word_count': word_count,
                'unique_words': unique_words,
                'semantic_density': unique_words / word_count if word_count else 0,
                'stop_words_ratio': sum(1 for word in words if word in self.stop_words) / len(words) if words else 0,
                'content_words': sum(1 for word in words if word.isalpha() and word not in self.stop_words),
                'content_word_ratio': sum(1 for word in words if word.isalpha() and word not in self.stop_words) / len(words) if words else 0,
                'bigram_count': len(list(nltk.bigrams(lemmatized_words))) if lemmatized_words else 0,
                'trigram_count': len(list(nltk.trigrams(lemmatized_words))) if lemmatized_words else 0,
            }
        except Exception as e:
            logger.warning(f"Error extracting semantic features: {e}")
            return {}
    
    def _analyze_humor_types(self, text: str) -> Dict[str, float]:
        """Analyze different types of humor."""
        humor_scores = {}
        
        for humor_type, patterns in self.humor_patterns.items():
            try:
                # Pattern matching
                pattern_score = 0
                for pattern in patterns['patterns']:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    pattern_score += len(matches)
                
                # Keyword matching
                keyword_score = sum(1 for keyword in patterns['keywords'] if keyword in text.lower())
                
                # Combine scores
                humor_scores[humor_type] = pattern_score * 0.7 + keyword_score * 0.3
                
            except Exception as e:
                logger.warning(f"Error analyzing humor type {humor_type}: {e}")
                humor_scores[humor_type] = 0
        
        # Normalize scores
        total_score = sum(humor_scores.values())
        if total_score > 0:
            for humor_type in humor_scores:
                humor_scores[humor_type] /= total_score
        
        return humor_scores
    
    def _match_humor_patterns(self, text: str) -> Dict[str, int]:
        """Match humor patterns in text."""
        matches = {}
        
        for humor_type, patterns in self.humor_patterns.items():
            try:
                match_count = 0
                for pattern in patterns['patterns']:
                    found_matches = re.findall(pattern, text, re.IGNORECASE)
                    match_count += len(found_matches)
                matches[humor_type] = match_count
            except Exception as e:
                logger.warning(f"Error matching patterns for {humor_type}: {e}")
                matches[humor_type] = 0
        
        return matches
    
    def _analyze_emotional_content(self, text: str) -> Dict[str, float]:
        """Analyze emotional content."""
        if not self.emotion_analyzer:
            return {}
        
        try:
            # Truncate if too long
            if len(text) > 512:
                text = text[:512]
            
            # Analyze emotions
            emotion_results = self.emotion_analyzer(text)
            
            # Extract emotion scores
            emotions = {}
            for result in emotion_results:
                emotions[result['label'].lower()] = result['score']
            
            # Add dominant emotion
            if emotions:
                dominant_emotion = max(emotions, key=emotions.get)
                emotions['dominant'] = dominant_emotion
                emotions['dominant_score'] = emotions[dominant_emotion]
            
            return emotions
            
        except Exception as e:
            logger.warning(f"Error analyzing emotional content: {e}")
            return {}
    
    def _extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """Extract linguistic features."""
        try:
            words = word_tokenize(text.lower())
            
            # Syllable counting (simplified)
            syllable_count = sum(max(1, sum(1 for char in word if char in 'aeiouAEIOU')) for word in words if word.isalpha())
            
            # Readability metrics
            word_count = len([w for w in words if w.isalpha()])
            sentence_count = len(sent_tokenize(text))
            
            # Flesch Reading Ease
            flesch_reading_ease = 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllable_count / word_count) if word_count > 0 else 0
            
            # Dale-Chall Readability
            easy_words = sum(1 for word in words if word.lower() not in self.stop_words and len(word) > 3)
            dale_chall_score = 0.1579 * (100 * (word_count - easy_words) / word_count) + 0.0496 * (word_count / sentence_count) if word_count > 0 else 0
            
            return {
                'syllable_count': syllable_count,
                'avg_syllables_per_word': syllable_count / word_count if word_count > 0 else 0,
                'flesch_reading_ease': max(0, min(100, flesch_reading_ease)),
                'dale_chall_score': dale_chall_score,
                'very_easy_words': sum(1 for word in words if len(word) <= 3 and word.isalpha()),
                'medium_difficulty_words': sum(1 for word in words if 4 <= len(word) <= 6 and word.isalpha()),
                'difficult_words': sum(1 for word in words if len(word) >= 7 and word.isalpha()),
                'very_easy_word_ratio': sum(1 for word in words if len(word) <= 3 and word.isalpha()) / word_count if word_count > 0 else 0,
                'medium_difficulty_word_ratio': sum(1 for word in words if 4 <= len(word) <= 6 and word.isalpha()) / word_count if word_count > 0 else 0,
                'difficult_word_ratio': sum(1 for word in words if len(word) >= 7 and word.isalpha()) / word_count if word_count > 0 else 0,
            }
        except Exception as e:
            logger.warning(f"Error extracting linguistic features: {e}")
            return {}
    
    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text."""
        if not self.sentiment_analyzer:
            return {}
        
        try:
            # Truncate if too long
            if len(text) > 512:
                text = text[:512]
            
            # Analyze sentiment
            sentiment_result = self.sentiment_analyzer(text)[0]
            
            return {
                'sentiment': sentiment_result['label'].lower(),
                'sentiment_score': sentiment_result['score'],
                'positive_score': sentiment_result['score'] if sentiment_result['label'] == 'POSITIVE' else 1 - sentiment_result['score'],
                'negative_score': sentiment_result['score'] if sentiment_result['label'] == 'NEGATIVE' else 1 - sentiment_result['score'],
                'neutral_score': 1 - sentiment_result['score'],
            }
        except Exception as e:
            logger.warning(f"Error analyzing sentiment: {e}")
            return {}
    
    def extract_humor_indicators(self, text: str) -> Dict[str, float]:
        """
        Extract specific humor indicators.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary of humor indicator scores
        """
        indicators = {
            'joke_structure': self._analyze_joke_structure(text),
            'wordplay_score': self._analyze_wordplay(text),
            'absurdity_score': self._analyze_absurdity(text),
            'surprise_element': self._analyze_surprise(text),
            'relatability': self._analyze_relatability(text),
            'cultural_references': self._analyze_cultural_references(text),
            'timing_quality': self._analyze_timing(text),
            'setup_quality': self._analyze_setup(text),
            'punchline_effectiveness': self._analyze_punchline(text),
            'audience_engagement': self._analyze_audience_engagement(text),
        }
        
        return indicators
    
    def _analyze_joke_structure(self, text: str) -> float:
        """Analyze joke structure quality."""
        # Check for classic joke patterns
        structure_indicators = 0
        total_indicators = 0
        
        # Setup indicators
        if 'why' in text.lower() or 'what' in text.lower() or 'how' in text.lower():
            structure_indicators += 1
            total_indicators += 1
        
        # Question pattern
        if '?' in text and 'because' in text.lower():
            structure_indicators += 1
            total_indicators += 1
        
        # Setup-punchline structure
        if text.count(',') < 10 and len(text.split()) > 10:  # Not too long, has some punctuation
            structure_indicators += 1
            total_indicators += 1
        
        return structure_indicators / max(total_indicators, 1)
    
    def _analyze_wordplay(self, text: str) -> float:
        """Analyze wordplay and pun potential."""
        wordplay_indicators = 0
        total_indicators = 0
        
        # Homophone patterns
        homophone_patterns = [
            r'\b(to|too|two)\b',
            r'\b(there|their|they\'re)\b',
            r'\b(your|you\'re)\b',
            r'\b(its|it\'s)\b',
            r'\b(know|no)\b',
            r'\b(see|sea)\b',
            r'\b(right|write)\b',
            r'\b(whole|hole)\b'
        ]
        
        for pattern in homophone_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                wordplay_indicators += 1
                total_indicators += 1
        
        # Alliteration
        words = word_tokenize(text.lower())
        for i in range(len(words) - 1):
            if words[i] and words[i + 1] and words[i][0] == words[i + 1][0]:
                wordplay_indicators += 0.5
                total_indicators += 1
        
        return wordplay_indicators / max(total_indicators, 1)
    
    def _analyze_absurdity(self, text: str) -> float:
        """Analyze absurdity and illogical elements."""
        absurdity_indicators = 0
        total_indicators = 0
        
        # Impossible scenarios
        absurd_patterns = [
            r'\b(a\s+chair|a\s+plant|a\s+lamp)\s+(talks|walks|talk|walk)',
            r'\b(a\s+dog|a\s+cat|an\s+animal)\s+(talks|reads|writes|drives)',
            r'\b(\w+)\s+(flies|swims|talks)\s+(without|but|except)',
            r'\b(impossible|ridiculous|absurd|nonsensical)',
            r'\b(magic|wizard|fairy|unicorn|dragon)'
        ]
        
        for pattern in absurd_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                absurdity_indicators += 1
                total_indicators += 1
        
        # Exaggerated statements
        if any(word in text.lower() for word in ['never', 'always', 'everyone', 'nobody', 'nothing']):
            absurdity_indicators += 0.5
            total_indicators += 1
        
        return absurdity_indicators / max(total_indicators, 1)
    
    def _analyze_surprise(self, text: str) -> float:
        """Analyze surprise elements."""
        surprise_indicators = 0
        total_indicators = 0
        
        # Surprise keywords
        surprise_keywords = ['suddenly', 'unexpectedly', 'boom', 'bam', 'out of nowhere', 'who knew', 'imagine that']
        
        for keyword in surprise_keywords:
            if keyword in text.lower():
                surprise_indicators += 1
                total_indicators += 1
        
        # Unexpected twist indicators
        if any(word in text.lower() for word in ['actually', 'really', 'but then', 'however', 'surprisingly']):
            surprise_indicators += 0.5
            total_indicators += 1
        
        return surprise_indicators / max(total_indicators, 1)
    
    def _analyze_relatability(self, text: str) -> float:
        """Analyze relatability to audience."""
        relatability_indicators = 0
        total_indicators = 0
        
        # Common experiences
        relatable_topics = [
            r'\b(traffic|commute|driving)',
            r'\b(work|office|job|career)',
            r'\b(family|parents|children|kids)',
            r'\b(friends|social|party)',
            r'\b(money|budget|expenses|bills)',
            r'\b(weather|rain|snow|sun)',
            r'\b(food|eating|restaurant|cooking)',
            r'\b(sleep|tired|awake|bed)',
            r'\b(technology|phone|computer|internet)',
            r'\b(health|doctor|hospital|medicine)'
        ]
        
        for pattern in relatable_topics:
            if re.search(pattern, text, re.IGNORECASE):
                relatability_indicators += 1
                total_indicators += 1
        
        # Universal experiences
        universal_words = ['everyone', 'we', 'us', 'people', 'everyone knows', 'we all']
        for word in universal_words:
            if word in text.lower():
                relatability_indicators += 0.5
                total_indicators += 1
        
        return relatability_indicators / max(total_indicators, 1)
    
    def _analyze_cultural_references(self, text: str) -> float:
        """Analyze cultural references."""
        cultural_indicators = 0
        total_indicators = 0
        
        # Common cultural references
        cultural_patterns = [
            r'\b(movie|film|cinema|hollywood|netflix)',
            r'\b(tv|television|show|series|episode)',
            r'\b(book|novel|author|literature)',
            r'\b(music|song|album|artist|band)',
            r'\b(sports|game|match|team|player)',
            r'\b(news|politics|government|election)',
            r'\b(holiday|celebration|party|event)',
            r'\b(social media|facebook|twitter|instagram|tiktok)',
            r'\b(pop culture|trending|viral|meme)',
            r'\b(history|past|ancient|modern)'
        ]
        
        for pattern in cultural_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                cultural_indicators += 1
                total_indicators += 1
        
        return cultural_indicators / max(total_indicators, 1)
    
    def _analyze_timing(self, text: str) -> float:
        """Analyze timing and rhythm."""
        timing_indicators = 0
        total_indicators = 0
        
        # Pauses and breaks
        if '...' in text or '-' in text or '—' in text:
            timing_indicators += 1
            total_indicators += 1
        
        # Sentence length variation
        sentences = sent_tokenize(text)
        if len(sentences) > 1:
            lengths = [len(sent.split()) for sent in sentences]
            length_variance = np.var(lengths)
            if length_variance > 10:  # Good variation
                timing_indicators += 1
                total_indicators += 1
        
        # Rhythm indicators
        if text.count(',') > 0 and text.count(',') < len(text.split()) // 3:
            timing_indicators += 0.5
            total_indicators += 1
        
        return timing_indicators / max(total_indicators, 1)
    
    def _analyze_setup(self, text: str) -> float:
        """Analyze setup quality."""
        setup_indicators = 0
        total_indicators = 0
        
        # Setup length (should be appropriate)
        word_count = len(text.split())
        if 10 <= word_count <= 50:  # Good length for setup
            setup_indicators += 1
            total_indicators += 1
        
        # Setup complexity
        complexity_score = 0
        if '?' in text:  # Question setup
            complexity_score += 1
        if 'why' in text.lower() or 'what' in text.lower() or 'how' in text.lower():  # Setup question
            complexity_score += 1
        if ',' in text:  # Multiple clauses
            complexity_score += 0.5
        
        setup_indicators += complexity_score / 2.5
        total_indicators += 1
        
        return setup_indicators / max(total_indicators, 1)
    
    def _analyze_punchline(self, text: str) -> float:
        """Analyze punchline effectiveness."""
        punchline_indicators = 0
        total_indicators = 0
        
        # Punchline length (should be concise)
        word_count = len(text.split())
        if word_count <= 30:  # Concise punchline
            punchline_indicators += 1
            total_indicators += 1
        
        # Exclamation or surprise
        if '!' in text:
            punchline_indicators += 1
            total_indicators += 1
        
        # Unexpected twist
        unexpected_words = ['actually', 'really', 'but', 'however', 'surprisingly', 'unexpectedly']
        if any(word in text.lower() for word in unexpected_words):
            punchline_indicators += 1
            total_indicators += 1
        
        return punchline_indicators / max(total_indicators, 1)
    
    def _analyze_audience_engagement(self, text: str) -> float:
        """Analyze audience engagement potential."""
        engagement_indicators = 0
        total_indicators = 0
        
        # Involvement
        if 'you' in text.lower() or 'we' in text.lower() or 'us' in text.lower():
            engagement_indicators += 1
            total_indicators += 1
        
        # Direct address
        direct_patterns = [r'\b(you\s+know|we\s+all|don\'t\s+you|you\s+should)', r'\b(imagine|think\s+about|consider)']
        for pattern in direct_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                engagement_indicators += 1
                total_indicators += 1
        
        # Emotional appeal
        emotional_words = ['funny', 'hilarious', 'hilarious', 'amazing', 'wonderful', 'terrible', 'awful']
        if any(word in text.lower() for word in emotional_words):
            engagement_indicators += 0.5
            total_indicators += 1
        
        return engagement_indicators / max(total_indicators, 1)
    
    def calculate_humor_score(self, text: str) -> float:
        """
        Calculate overall humor score for text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Humor score between 0 and 1
        """
        # Extract all features
        features = self.extract_all_features(text)
        
        # Calculate weighted score
        weights = {
            'basic': 0.1,
            'syntactic': 0.15,
            'semantic': 0.1,
            'humor_types': 0.3,
            'pattern_matching': 0.2,
            'emotional': 0.05,
            'linguistic': 0.05,
            'sentiment': 0.05
        }
        
        total_score = 0
        for category, weight in weights.items():
            if category in features:
                category_features = features[category]
                if isinstance(category_features, dict):
                    numeric_values = [v for v in category_features.values() if isinstance(v, (int, float))]
                    category_score = sum(numeric_values) / len(numeric_values) if numeric_values else 0
                    total_score += category_score * weight
        
        return min(1.0, max(0.0, total_score))
    
    def analyze_humor_composition(self, text: str) -> Dict[str, Dict[str, float]]:
        """
        Comprehensive humor composition analysis.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Detailed humor composition analysis
        """
        # Extract all features
        features = self.extract_all_features(text)
        
        # Calculate humor indicators
        indicators = self.extract_humor_indicators(text)
        
        # Calculate overall score
        overall_score = self.calculate_humor_score(text)
        
        # Compile results
        results = {
            'text_features': features,
            'humor_indicators': indicators,
            'overall_humor_score': overall_score,
            'text_length': len(text),
            'word_count': len(text.split()),
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
        
        return results
    
    def preprocess_text(self, text: str, 
                       remove_punctuation: bool = False,
                       remove_stopwords: bool = False,
                       lemmatize: bool = False,
                       lowercase: bool = True) -> str:
        """
        Preprocess text for analysis.
        
        Args:
            text: Input text to preprocess
            remove_punctuation: Whether to remove punctuation
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to lemmatize words
            lowercase: Whether to convert to lowercase
            
        Returns:
            Preprocessed text
        """
        processed_text = str(text)
        
        if lowercase:
            processed_text = processed_text.lower()
        
        if remove_punctuation:
            processed_text = processed_text.translate(str.maketrans('', '', string.punctuation))
        
        if remove_stopwords or lemmatize:
            words = word_tokenize(processed_text)
            
            if remove_stopwords:
                words = [word for word in words if word not in self.stop_words]
            
            if lemmatize:
                words = [self.lemmatizer.lemmatize(word) for word in words]
            
            processed_text = ' '.join(words)
        
        return processed_text
    
    def get_feature_importance(self, text: str) -> Dict[str, float]:
        """
        Get feature importance for humor analysis.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary of feature importance scores
        """
        features = self.extract_all_features(text)
        humor_types = features.get('humor_types', {})
        
        # Calculate importance based on humor type scores
        importance_scores = {}
        
        for humor_type, score in humor_types.items():
            if score > 0:
                # Weight by score
                importance_scores[humor_type] = score
        
        # Add overall importance
        total_importance = sum(importance_scores.values())
        if total_importance > 0:
            for humor_type in importance_scores:
                importance_scores[humor_type] /= total_importance
        
        return importance_scores
    
    def batch_process(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Process multiple texts in batches.
        
        Args:
            texts: List of texts to process
            batch_size: Batch size for processing
            
        Returns:
            List of analysis results
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = []
            
            for text in batch_texts:
                try:
                    result = self.analyze_humor_composition(text)
                    batch_results.append(result)
                except Exception as e:
                    logger.warning(f"Error processing text: {e}")
                    batch_results.append({'error': str(e)})
            
            results.extend(batch_results)
        
        return results