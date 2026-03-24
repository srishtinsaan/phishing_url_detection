import re
import math
from urllib.parse import urlparse

def extract_features(url):
    features = []

    url = url.lower()
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path

    # 🔹 Basic URL features
    features.append(len(url))                          # length
    features.append(url.count('.'))                    # dots
    features.append(url.count('-'))                    # hyphens
    features.append(1 if '@' in url else 0)            # @
    features.append(1 if 'https' in url else 0)        # https
    features.append(url.count('/'))                    # slash
    features.append(url.count('?'))                    # query
    features.append(url.count('='))                    # equals

    # 🔹 digits
    digits = sum(c.isdigit() for c in url)
    features.append(digits)                            # total digits
    features.append(digits / len(url) if len(url) > 0 else 0)  # digit ratio

    # 🔹 special characters
    special_chars = sum(not c.isalnum() for c in url)
    features.append(special_chars)                     # total special chars
    features.append(special_chars / len(url) if len(url) > 0 else 0)

    # 🔹 IP address detection
    features.append(1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0)

    # 🔹 suspicious words (expanded)
    suspicious_words = [
        'login','bank','verify','secure','account',
        'update','free','bonus','signin','confirm','password'
    ]
    features.append(1 if any(word in url for word in suspicious_words) else 0)

    # 🔹 domain features
    features.append(len(domain))                       # domain length
    features.append(domain.count('.'))                 # subdomains
    features.append(1 if any(c.isdigit() for c in domain) else 0)

    # 🔹 path features
    features.append(len(path))                         # path length
    features.append(path.count('/'))                   # subdirectories

    # 🔹 shortening service
    shorteners = ['bit.ly', 'tinyurl', 'goo.gl', 't.co']
    features.append(1 if any(s in url for s in shorteners) else 0)

    # 🔹 suspicious TLDs
    suspicious_tlds = ['.tk', '.ml', '.ga', '.cf']
    features.append(1 if any(tld in domain for tld in suspicious_tlds) else 0)

    # 🔹 entropy (randomness)
    if len(url) > 0:
        prob = [url.count(c)/len(url) for c in set(url)]
        entropy = -sum([p * math.log2(p) for p in prob])
    else:
        entropy = 0
    features.append(entropy)

    # 🔹 word-based features
    words = re.split(r'\W+', url)
    words = [w for w in words if w]

    features.append(max([len(w) for w in words], default=0))  # longest word
    features.append(
        sum(len(w) for w in words)/len(words) if words else 0
    )                                                        # avg word length

    # 🔹 redirection tricks
    features.append(url.count('//'))                # multiple slashes

    # 🔹 http vs https
    features.append(1 if url.startswith('http://') else 0)

    # 🔹 repeating characters (e.g. aaaa)
    features.append(1 if re.search(r'(.)\1{3,}', url) else 0)

    # 🔹 encoded characters
    features.append(url.count('%'))

    # 🔹 length difference
    features.append(len(url) - len(domain))

    return features