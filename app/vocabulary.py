"""Vocabulary management for the ASR model"""
from typing import List, Optional
import os


class Vocabulary:
    def __init__(self):
        self.tokens: List[str] = []
        self.blank_index = 128
        self._load_default_vocab()

    def _load_default_vocab(self):
        """Load the default BPE vocabulary for Hindi ASR"""
        self.tokens = [
            '<unk>',  # 0
            'ा',  # 1
            'र',  # 2
            'ी',  # 3
            '▁',  # 4 - word boundary marker
            'े',  # 5
            'न',  # 6
            'ि',  # 7
            'त',  # 8
            'क',  # 9
            '्',  # 10
            'ल',  # 11
            'म',  # 12
            'स',  # 13
            'ं',  # 14
            '▁स',  # 15
            'ह',  # 16
            'ो',  # 17
            'ु',  # 18
            'द',  # 19
            'य',  # 20
            'प',  # 21
            '▁है',  # 22
            '▁के',  # 23
            'ग',  # 24
            '▁ब',  # 25
            '▁म',  # 26
            'व',  # 27
            '▁क',  # 28
            '▁में',  # 29
            'ट',  # 30
            '▁अ',  # 31
            'ज',  # 32
            '▁द',  # 33
            '▁प',  # 34
            '▁आ',  # 35
            '्र',  # 36
            'ू',  # 37
            '▁ज',  # 38
            '▁की',  # 39
            '▁र',  # 40
            'ध',  # 41
            'र्',  # 42
            'ों',  # 43
            'ख',  # 44
            '▁का',  # 45
            '्य',  # 46
            'च',  # 47
            'ए',  # 48
            'ब',  # 49
            'भ',  # 50
            'ने',  # 51
            '▁को',  # 52
            '▁से',  # 53
            '▁ल',  # 54
            '▁और',  # 55
            '▁प्र',  # 56
            '▁त',  # 57
            '▁कर',  # 58
            '▁व',  # 59
            'ता',  # 60
            'श',  # 61
            '▁कि',  # 62
            '▁ह',  # 63
            '▁न',  # 64
            '▁ग',  # 65
            'ना',  # 66
            '▁हो',  # 67
            'ै',  # 68
            '▁पर',  # 69
            'थ',  # 70
            '▁उ',  # 71
            'ड',  # 72
            '▁च',  # 73
            'िक',  # 74
            'ण',  # 75
            'ई',  # 76
            '▁हैं',  # 77
            'िया',  # 78
            '▁इस',  # 79
            'फ',  # 80
            '▁वि',  # 81
            'वा',  # 82
            '▁जा',  # 83
            'ष',  # 84
            'ित',  # 85
            '▁श',  # 86
            'ें',  # 87
            '▁ने',  # 88
            'ेश',  # 89
            'ते',  # 90
            'इ',  # 91
            '▁भी',  # 92
            'का',  # 93
            '▁एक',  # 94
            '्या',  # 95
            '▁हम',  # 96
            '▁सं',  # 97
            'िल',  # 98
            'ंग',  # 99
            'ड़',  # 100
            'छ',  # 101
            'क्ष',  # 102
            'ौ',  # 103
            'ठ',  # 104
            '़',  # 105
            'ॉ',  # 106
            'ओ',  # 107
            'ढ',  # 108
            'घ',  # 109
            'आ',  # 110
            'झ',  # 111
            'ऐ',  # 112
            'ँ',  # 113
            'ऊ',  # 114
            'उ',  # 115
            'ः',  # 116
            'औ',  # 117
            ',',  # 118
            'ऍ',  # 119
            'ॅ',  # 120
            'ॠ',  # 121
            'ऋ',  # 122
            'ऑ',  # 123
            'ञ',  # 124
            'ृ',  # 125
            'अ',  # 126
            'ङ',  # 127
        ]

    def load_from_file(self, vocab_path: str) -> bool:
        """Load vocabulary from file"""
        try:
            if os.path.exists(vocab_path):
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    self.tokens = [line.strip() for line in f]
                print(f"Loaded vocabulary from {vocab_path}: {len(self.tokens)} tokens")
                return True
        except Exception as e:
            print(f"Error loading vocabulary: {e}")
        return False

    def get_token(self, index: int) -> Optional[str]:
        """Get token by index"""
        if 0 <= index < len(self.tokens):
            return self.tokens[index]
        return None

    def __len__(self) -> int:
        return len(self.tokens)


# Global vocabulary instance
vocabulary = Vocabulary()