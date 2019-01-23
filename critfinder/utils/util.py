import random
import string


def random_string(N):
    return "".join([random.choice(string.ascii_letters) for _ in range(N)])
