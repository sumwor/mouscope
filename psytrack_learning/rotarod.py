import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')

# Load the Excel file
file_path = r'Z:\HongliWang\Juvi_ASD Deterministic\CntnapOdorRotarod4.xlsx'
df = pd.read_excel(file_path)

# Fix column names by removing trailing spaces
df.columns = df.columns.str.strip()

print("Column names in the dataset:")
print(df.columns.tolist())
print(f"\nDataset shape: {df.shape}")

# Show all animals and their genotypes
animal_genotype_map = df[['Animal', 'Genotype']].drop_duplicates().sort_values('Animal')
print(f"\nAll animals in dataset:")
for _, row in animal_genotype_map.iterrows():
    print(f"  Animal {row['Animal']}: {row['Genotype']}")

print(f"\nTotal unique animals: {len(animal_genotype_map)}")
genotype_counts = animal_genotype_map['Genotype'].value_counts()
print(f"Overall genotype distribution: {dict(genotype_counts)}")

# Check which conditions each animal appears in
print(f"\nCondition participation by animal:")
important_conditions = ['AB1', 'AB2', 'CD1', 'CD2', 'DC5', 'DC6']
for _, row in animal_genotype_map.iterrows():
    animal = row['Animal']
    genotype = row['Genotype']
    animal_conditions = []
    for condition in important_conditions:
        if len(df[(df['Animal'] == animal) & (df['MyDay'] == condition)]) > 0:
            animal_conditions.append(condition)
    print(f"  Animal {animal} ({genotype}): {animal_conditions}")

# Color mapping for genotypes
color_map = {'KO': 'lightblue', 'WT': 'black'}



def aggregate_animal_data(data, condition_name):
    """Aggregate data by animal for a given condition"""
    print(f"\nProcessing {condition_name}:")
    print(f"  Raw data rows: {len(data)}")

    # Group by Animal and take the mean of numeric columns (this handles multiple trials per animal)
    # For non-numeric columns like Genotype, take the first value (should be same for each animal)
    numeric_cols = ['avgAB', 'avgCD', 'avgDC', 'rot1-3avg', 'rot10-12avg']

    animal_data = []

    for animal in data['Animal'].unique():
        animal_rows = data[data['Animal'] == animal]

        # Get genotype (should be consistent for each animal)
        genotype = animal_rows['Genotype'].iloc[0]

        # Average the numeric values for this animal across trials
        animal_record = {'Animal': animal, 'Genotype': genotype}

        for col in numeric_cols:
            if col in animal_rows.columns:
                # Take mean of non-NaN values for this animal
                non_nan_values = animal_rows[col].dropna()
                if len(non_nan_values) > 0:
                    animal_record[col] = non_nan_values.mean()
                else:
                    animal_record[col] = np.nan
            else:
                animal_record[col] = np.nan

        animal_data.append(animal_record)
        print(f"    Animal {animal} ({genotype}): {len(animal_rows)} trials")

    result_df = pd.DataFrame(animal_data)
    print(f"  Final data points: {len(result_df)}")
    print(f"  Genotype distribution: {dict(result_df['Genotype'].value_counts())}")

    return result_df


def create_correlation_plot(animal_df, x_col, y_col, title, x_label, y_label):
    """Create scatter plot with correlation analysis using aggregated animal data"""

    # Remove animals with NaN values in required columns
    mask = ~(pd.isna(animal_df[x_col]) | pd.isna(animal_df[y_col]) | pd.isna(animal_df['Genotype']))
    clean_df = animal_df[mask].copy()

    print(f"\n{title}:")
    print(f"  Animals with valid data: {len(clean_df)}")
    print(f"  Genotype distribution: {dict(clean_df['Genotype'].value_counts())}")

    if len(clean_df) < 2:
        print(f"  Insufficient data for correlation")
        return None, None

    # Calculate Pearson correlation
    try:
        correlation, p_value = pearsonr(clean_df[x_col], clean_df[y_col])
    except Exception as e:
        print(f"  Could not calculate correlation: {e}")
        return None, None

    # Create the plot
    plt.figure(figsize=(10, 8))

    # Plot points by genotype
    for genotype in ['KO', 'WT']:
        genotype_data = clean_df[clean_df['Genotype'] == genotype]
        if len(genotype_data) > 0:
            plt.scatter(genotype_data[x_col], genotype_data[y_col],
                        c=color_map[genotype], label=f'{genotype} (n={len(genotype_data)})',
                        alpha=0.7, s=80)

            # Print individual animal data for verification
            for _, row in genotype_data.iterrows():
                print(f"    Animal {row['Animal']} ({genotype}): {x_col}={row[x_col]:.3f}, {y_col}={row[y_col]:.3f}")

    # Add trend line
    if len(clean_df) > 1:
        z = np.polyfit(clean_df[x_col], clean_df[y_col], 1)
        p = np.poly1d(z)
        x_range = np.linspace(clean_df[x_col].min(), clean_df[x_col].max(), 100)
        plt.plot(x_range, p(x_range), "r--", alpha=0.8, linewidth=2)

    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(f'{title}\nr = {correlation:.3f}, p = {p_value:.3f}, n = {len(clean_df)}',
              fontsize=14, pad=20)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the figure instead of showing it
    filename = f"{title.replace(':', '_').replace(' ', '_').replace('(', '').replace(')', '').replace('&', 'and')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  Saved plot as: {filename}")
    plt.close()  # Close the figure to free memory

    print(f"  Correlation: r = {correlation:.3f}, p = {p_value:.3f}")
    return correlation, p_value


print("\n" + "=" * 60)
print("STARTING CORRELATION ANALYSES - ONE POINT PER ANIMAL")
print("=" * 60)

# 1-2. AB1 Analysis
ab1_data = df[df['MyDay'] == 'AB1'].copy()
if not ab1_data.empty:
    ab1_animals = aggregate_animal_data(ab1_data, 'AB1')

    create_correlation_plot(ab1_animals, 'avgAB', 'rot1-3avg',
                            'AB1: avgAB vs rot1-3avg', 'avgAB Performance', 'rot1-3avg')

    create_correlation_plot(ab1_animals, 'avgAB', 'rot10-12avg',
                            'AB1: avgAB vs rot10-12avg', 'avgAB Performance', 'rot10-12avg')
else:
    print("No AB1 data found")

# 3-4. AB2 Analysis
ab2_data = df[df['MyDay'] == 'AB2'].copy()
if not ab2_data.empty:
    ab2_animals = aggregate_animal_data(ab2_data, 'AB2')

    create_correlation_plot(ab2_animals, 'avgAB', 'rot1-3avg',
                            'AB2: avgAB vs rot1-3avg', 'avgAB Performance', 'rot1-3avg')

    create_correlation_plot(ab2_animals, 'avgAB', 'rot10-12avg',
                            'AB2: avgAB vs rot10-12avg', 'avgAB Performance', 'rot10-12avg')
else:
    print("No AB2 data found")

# 5-6. CD1 Analysis
cd1_data = df[df['MyDay'] == 'CD1'].copy()
if not cd1_data.empty:
    cd1_animals = aggregate_animal_data(cd1_data, 'CD1')

    create_correlation_plot(cd1_animals, 'avgCD', 'rot1-3avg',
                            'CD1: avgCD vs rot1-3avg', 'avgCD Performance', 'rot1-3avg')

    create_correlation_plot(cd1_animals, 'avgCD', 'rot10-12avg',
                            'CD1: avgCD vs rot10-12avg', 'avgCD Performance', 'rot10-12avg')
else:
    print("No CD1 data found")

# 7-8. CD2 Analysis
cd2_data = df[df['MyDay'] == 'CD2'].copy()
if not cd2_data.empty:
    cd2_animals = aggregate_animal_data(cd2_data, 'CD2')

    create_correlation_plot(cd2_animals, 'avgCD', 'rot1-3avg',
                            'CD2: avgCD vs rot1-3avg', 'avgCD Performance', 'rot1-3avg')

    create_correlation_plot(cd2_animals, 'avgCD', 'rot10-12avg',
                            'CD2: avgCD vs rot10-12avg', 'avgCD Performance', 'rot10-12avg')
else:
    print("No CD2 data found")

# 9-10. DC5/DC6 Analysis - Average DC5 and DC6 for each animal
print("\n" + "=" * 40)
print("DC5/DC6 AVERAGING ANALYSIS")
print("=" * 40)

dc5_data = df[df['MyDay'] == 'DC5'].copy()
dc6_data = df[df['MyDay'] == 'DC6'].copy()

if not dc5_data.empty and not dc6_data.empty:
    # Aggregate DC5 and DC6 data by animal
    dc5_animals = aggregate_animal_data(dc5_data, 'DC5')
    dc6_animals = aggregate_animal_data(dc6_data, 'DC6')

    # Find animals that appear in both DC5 and DC6
    dc5_animal_set = set(dc5_animals['Animal'].unique())
    dc6_animal_set = set(dc6_animals['Animal'].unique())
    common_animals = dc5_animal_set.intersection(dc6_animal_set)

    print(f"\nAnimals with both DC5 and DC6 data: {sorted(common_animals)}")

    if common_animals:
        # Create averaged dataset
        averaged_data = []

        for animal in common_animals:
            dc5_row = dc5_animals[dc5_animals['Animal'] == animal].iloc[0]
            dc6_row = dc6_animals[dc6_animals['Animal'] == animal].iloc[0]

            # Average the avgDC values
            if not (pd.isna(dc5_row['avgDC']) or pd.isna(dc6_row['avgDC'])):
                avg_dc_value = (dc5_row['avgDC'] + dc6_row['avgDC']) / 2

                # Use rotarod data (average from both days if available)
                rot1_3 = dc5_row['rot1-3avg'] if not pd.isna(dc5_row['rot1-3avg']) else dc6_row['rot1-3avg']
                rot10_12 = dc5_row['rot10-12avg'] if not pd.isna(dc5_row['rot10-12avg']) else dc6_row['rot10-12avg']

                averaged_data.append({
                    'Animal': animal,
                    'avg_DC_combined': avg_dc_value,
                    'rot1-3avg': rot1_3,
                    'rot10-12avg': rot10_12,
                    'Genotype': dc5_row['Genotype']
                })

                print(
                    f"Animal {animal} ({dc5_row['Genotype']}): DC5={dc5_row['avgDC']:.3f}, DC6={dc6_row['avgDC']:.3f}, Average={avg_dc_value:.3f}")

        if averaged_data:
            avg_df = pd.DataFrame(averaged_data)

            create_correlation_plot(avg_df, 'avg_DC_combined', 'rot1-3avg',
                                    'DC Average (DC5&DC6) vs rot1-3avg', 'Average DC Performance', 'rot1-3avg')

            create_correlation_plot(avg_df, 'avg_DC_combined', 'rot10-12avg',
                                    'DC Average (DC5&DC6) vs rot10-12avg', 'Average DC Performance', 'rot10-12avg')
        else:
            print("No valid averaged DC data could be created")
    else:
        print("No animals found with both DC5 and DC6 data")
else:
    print("Missing DC5 or DC6 data in dataset")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)