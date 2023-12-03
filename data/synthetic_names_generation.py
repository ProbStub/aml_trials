import pandas as pd
from faker import Faker
import random
import re

# Create a Faker instance
fake = Faker()


# Function to generate a single name variation pattern
def generate_name_variation(name, variation_type):
    if variation_type == 'Abbreviation Usage':
        if '.' not in name:
            # Insert a dot to simulate abbreviation
            return f"{name.split()[0][0]}. {' '.join(name.split()[1:])}"

    elif variation_type == 'Initials vs. Full Names':
        if '.' not in name:
            # Convert the first name to initials
            return f"{name.split()[0][0]}. {' '.join(name.split()[1:])}"

    elif variation_type == 'Hyphenated or Combined Surnames':
        if '-' not in name:
            # Insert a hyphen to simulate a hyphenated surname
            return f"{name.split()[0]}-{name.split()[1]}"

    # Add more cases for other name variation patterns

    # If no variation is applied, return the original name
    return name


# Function to generate the client_list dataset with name variation patterns
def generate_client_list(size=10, variation_types=None):
    if variation_types is None:
        variation_types = ['Abbreviation Usage', 'Initials vs. Full Names', 'Hyphenated or Combined Surnames']
        # Add other variation types as needed

    data = {
        'ent_num': [],
        'SDN_Name': [],
        'SDN_Type': [],
        # Add other columns as needed
    }

    for _ in range(size):
        # Generate fake data for the dataset
        ent_num = fake.unique.random_number()
        sdn_name = fake.name()
        sdn_type = fake.random_element(elements=('Individual', 'Company'))

        # Apply each name variation pattern to the 'SDN_Name' column
        for variation_type in variation_types:
            sdn_name_variation = generate_name_variation(sdn_name, variation_type)

            # Add the generated data to the dataset
            data['ent_num'].append(ent_num)
            data['SDN_Name'].append(sdn_name_variation)
            data['SDN_Type'].append(sdn_type)
            # Add other columns as needed

    # Create a DataFrame from the generated data
    df = pd.DataFrame(data)
    return df


# Generate the client_list dataset with multiple name variation patterns
client_list_df = generate_client_list(size=5, variation_types=['Abbreviation Usage', 'Initials vs. Full Names',
                                                               'Hyphenated or Combined Surnames'])

# Display the generated dataset
print(client_list_df)

