import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt

def analyze_mz_range(df):
    print("Analyzing m/z range across all spectra...")
    all_mz = []
    all_precursors = df['precursor'].dropna().astype(float)
    
    for idx, spec in enumerate(df['spectrum']):
        try:
            # Case 1: List of tuples (most common)
            if isinstance(spec, (list, tuple)) and len(spec) > 0:
                if isinstance(spec[0], (list, tuple)) and len(spec[0]) == 2:
                    mz_vals = [float(mz) for mz, _ in spec]
                elif isinstance(spec[0], str):
                    mz_vals = [float(pair.split(':')[0]) for pair in spec]
                else:
                    continue
            # Case 2: String like "123.04:100,105.03:50"
            elif isinstance(spec, str):
                pairs = [p.strip() for p in spec.split(',') if ':' in p]
                mz_vals = [float(p.split(':')[0]) for p in pairs]
            # Case 3: NumPy array
            elif hasattr(spec, 'shape') and spec.shape[1] == 2:
                mz_vals = spec[:, 0].astype(float)
            else:
                continue
                
            all_mz.extend(mz_vals)
            
            # Optional: print first valid spectrum
            if len(all_mz) > 0 and idx == 0:
                print(f"First valid spectrum format: {type(spec)}")
                print(f"First 3 peaks: {mz_vals[:3]}")
                
        except Exception as e:
            if idx < 5:
                print(f"Failed on row {idx}: {e}")
            continue
    
    if len(all_mz) == 0:
        print("No valid m/z values found! Check 'spectrum' format.")
        return None
        
    all_mz = np.array(all_mz)
    precursors = np.array(all_precursors)
    
    print(f"Total peaks: {len(all_mz):,}")
    print(f"Precursor m/z range: [{precursors.min():.2f}, {precursors.max():.2f}]")
    print(f"Fragment m/z range: [{all_mz.min():.2f}, {all_mz.max():.2f}]")
    print(f"95th percentile: {np.percentile(all_mz, 95):.2f}")
    print(f"99th percentile: {np.percentile(all_mz, 99):.2f}")
    print(f"99.9th percentile: {np.percentile(all_mz, 99.9):.2f}")
    
    # Plot
    plt.figure(figsize=(10, 4))
    plt.hist(all_mz, bins=200, range=(0, 1200), alpha=0.7, color='teal')
    plt.axvline(np.percentile(all_mz, 99), color='red', linestyle='--', label='99th %')
    plt.xlabel("m/z")
    plt.ylabel("Peak count")
    plt.title("Distribution of All Fragment m/z Values")
    plt.legend()
    plt.tight_layout()
    plt.savefig("mz_distribution.png", dpi=150)
    print("→ m/z histogram saved: mz_distribution.png")
    
    return all_mz


df = pd.read_pickle("metlin.pkl")
print(df['collision_energy'].value_counts())
print(analyze_mz_range(df))