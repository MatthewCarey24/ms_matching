from numba import njit
import numpy as np
import pickle 
import pandas as pd



@njit
def compute_pair_scores(mz_a, mz_b, int_a, int_b, tolerance_ppm):
    # shape things up for numba
    max_pairs = len(mz_a) * len(mz_b)
    pairs = np.zeros((max_pairs, 2), dtype=np.int32)
    scores = np.zeros(max_pairs, dtype=np.float64)

    count = 0
    for i in range(len(mz_a)):
        for j in range(len(mz_b)):
            tolerance_da = mz_a[i] * tolerance_ppm / 1e6
            if abs(mz_a[i] - mz_b[j]) <= tolerance_da:

                # custom score function, option to include intensities
                score = 1/ 1 + abs(mz_a[i] - mz_b[j])


                pairs[count] = [i, j]
                scores[count] = score
                count += 1
    return pairs[:count], scores[:count]


    

def greedy_peak_matching(spec_a, spec_b, tolerance_ppm):
    """
    Perform greedy peak matching between two spectra using ppm tolerance
    
    Args:
        spec_a, spec_b: Lists of [mz, intensity] pairs 
        tolerance_ppm: Maximum m/z difference in parts per million 
    
    Returns:
        matched_spec: List of (mz, int_a, int_b) tuples, with intensity = 0 for unmatched peaks.
    """
    # Convert spectra to NumPy arrays for Numba
    spec_a = np.array(spec_a, dtype=np.float64)
    spec_b = np.array(spec_b, dtype=np.float64)
    mz_a, int_a = spec_a[:, 0], spec_a[:, 1]
    mz_b, int_b = spec_b[:, 0], spec_b[:, 1]
    
    pairs, scores = compute_pair_scores(mz_a, mz_b, int_a, int_b, tolerance_ppm)
    
    # Initialize combined spectrum with all spec_a peaks (unmatched have int_b=0)
    combined_spec = [(int_a[i], 0.0) for i in range(len(mz_a))]
    matched_b = set()
    
    if len(pairs) > 0:
        sort_indices = np.argsort(-scores)
        pairs = pairs[sort_indices]
        
        used_a = set()
        for i, j in pairs:
            if i not in used_a and j not in matched_b:
                combined_spec[i] = (int_a[i], int_b[j])
                used_a.add(i)
                matched_b.add(j)
    
    # Add unmatched spec_b peaks w no spec_a match
    for j in range(len(mz_b)):
        if j not in matched_b:
            combined_spec.append((0.0, int_b[j]))
    
    intensities_a = [a for a, b in combined_spec]
    intensities_b = [b for a, b in combined_spec]
    
    return intensities_a, intensities_b



if __name__ == "__main__":
    
    with open("gnps_highres.pkl", "rb") as f:
        data = pickle.load(f)
    # Convert to list if df
    if isinstance(data, pd.DataFrame):
        data = data.to_dict('records')
    
    tolerance_ppm = 10.0 
    
    for i in range(5):
        for j in range(i + 1, 5):
            spec_a = data[i]['spectrum']
            spec_b = data[j]['spectrum']
            print(f"\nMatching spectrum {i} with spectrum {j}")
            try:
                spec_a, spec_b = greedy_peak_matching(spec_a, spec_b, tolerance_ppm=tolerance_ppm)
                print(f"spec_a: {spec_a[:5]}\nspec_b: {spec_b[:5]}")
                        
            except Exception as e:
                print(f"Error matching spectra {i} and {j}: {e}")

    