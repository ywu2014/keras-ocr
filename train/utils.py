import string
import random
def get_text_generator(alphabet=None, lowercase=False, max_string_length=None):
    """Generates strings of sentences using only the letters in alphabet.

    Args:
        alphabet: The alphabet of permitted characters
        lowercase: Whether to convert all strings to lowercase.
        max_string_length: The maximum length of the string
    """
    while True:
        sentence = next_string()
        if lowercase:
            sentence = sentence.lower()
        sentence = "".join([s for s in sentence if (alphabet is None or s in alphabet)])
        if max_string_length is not None:
            sentence = sentence[:max_string_length]
        yield sentence

def randon_boolean():
    return random.choice([True, False])


def next_random_letter():
    count = random.randint(1, 4)
    return ''.join(random.choices(string.ascii_lowercase, k=count))

def next_number_str():
    """eg: 110, 0.03"""
    gen_integer = randon_boolean()
    if gen_integer:
        number_str = str(random.randint(1, 100))
    else:
        number_str = str(round(random.uniform(1, 10), 2))
    return number_str

def next_Φ_str():
    """eg: Φ36, Φ0.02"""
    return "Φ" + next_number_str()

def next_Ra_str():
    """eg: Ra12.5, Ra25"""
    return next_random_letter() + next_number_str()

def next_positive_negative_str():
    """eg: 128±0.05"""
    return str(random.randint(1, 100)) + "±" + str(round(random.uniform(1, 10), 2))

def next_multiple_str():
    """eg: 2×Φ17"""
    return next_number_str() + "×" + next_Φ_str()

def next_composite_str():
    """eg: M24×1.5-7H"""
    return next_random_letter() + next_number_str() + "×" + next_number_str() + "-" + next_number_str()+next_random_letter()

def next_string():
    num = random.randint(1, 7)
    if num == 1:
        return next_positive_negative_str()
    elif num == 2:
        return next_Φ_str()
    elif num == 3:
        return next_Ra_str()
    elif num == 4:
        return next_number_str()
    elif num == 5:
        return next_multiple_str()
    elif num == 6:
        return next_composite_str()
    else:
        return next_random_letter()

if __name__ == "__main__":
    # 110, 0.03
    # Φ36, Φ0.02
    # Ra12.5, Ra25
    # 128±0.05
    # 2×Φ17
    # M24×1.5-7H
    alphabet = string.digits + string.ascii_letters + '+-×±Φ. '
    # print(next_positive_negative_str())
    text_generator = get_text_generator(alphabet=alphabet)
    for _ in range(10):
        print(next(text_generator))