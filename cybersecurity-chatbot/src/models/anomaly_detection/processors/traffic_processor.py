import numpy as np
import re

def process_input(text: str):
    """
    Converts user natural language into a simulated 41-feature vector for the NSL-KDD model.
    This is a placeholder and should eventually be improved with NLP and traffic simulation.
    """
    text = text.lower()
    vector = np.zeros((1, 41), dtype=np.float32)

    # Examples of keyword mapping (simulate network activity features)
    if "dos" in text or "syn flood" in text:
        vector[0][0] = 1.0  # example: duration or wrong fragment
        vector[0][4] = 3.0  # wrong fragments (dummy)
        vector[0][23] = 1.0  # count

    if "scan" in text or "probe" in text:
        vector[0][2] = 1.0  # protocol_type
        vector[0][25] = 10.0  # srv_count

    if "ftp" in text or "r2l" in text:
        vector[0][1] = 1.0  # service

    if "tcp" in text:
        vector[0][2] = 1.0  # protocol_type: TCP

    if "icmp" in text:
        vector[0][2] = 2.0  # protocol_type: ICMP

    if re.search(r'\b\d+\b', text):
        nums = [int(n) for n in re.findall(r'\b\d+\b', text)]
        vector[0][6] = min(sum(nums), 100)  # num_failed_logins or duration

    return vector
