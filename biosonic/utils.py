from numpy.typing import NDArray
import numpy as np
import logging

def exclude_trailing_and_leading_zeros(envelope: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Removes leading and trailing zeros from a NumPy array.

    This function identifies and excludes the leading and trailing
    zeros in the input array `envelope`. It finds the first and last non-zero
    elements, and returns a new array that starts from the first non-zero value
    and ends at the last non-zero value.

    Args:
        envelope (np.ndarray): A 1D NumPy array containing numerical values, 
                                potentially with leading and/or trailing zeros.

    Returns:
        np.ndarray: A 1D NumPy array with leading and trailing zeros excluded. 
                    The returned array will only contain values between the first 
                    and last non-zero elements from the original array.

    Example:
        >>> import numpy as np
        >>> arr = np.array([0, 0, 1, 2, 3, 0, 0])
        >>> exclude_trailing_and_leading_zeros(arr)
        array([1, 2, 3])

    Notes:
        - If the array consists entirely of zeros, an empty array will be returned.
        - The function assumes that the input array is 1D.

    """
    if envelope.ndim != 1:
        raise ValueError("Input array must be 1D.")

    try:
        # Exclude trailing zeros (identify last non-zero element)
        non_zero_end = np.argmax(envelope[::-1] > 0)
        envelope = envelope[:len(envelope) - non_zero_end]
        
        # Exclude leading zeros (identify first non-zero value)
        non_zero_start = np.argmax(envelope > 0)
        envelope = envelope[non_zero_start:]
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise
    
    return envelope
