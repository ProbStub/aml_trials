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
####
## Retrieve the OFAC Advanced SDNs from XML
####

import requests
import xml.etree.ElementTree as ET
import pandas as pd

def parse_ofac_xml(xml_url):
    # Download the XML file
    response = requests.get(xml_url)
    xml_data = response.content

    # Parse the XML data
    root = ET.fromstring(xml_data)

    # Define the namespace
    namespace = {'ofac': 'http://www.un.org/sanctions/1.0'}

    # Initialize lists to store data
    data = {
        'party_type': [],
        'name': [],
        'registration_country': [],
        'official_identification_number': [],
        'official_identification_type': [],
        'official_identification_country': [],
        'first_address': [],
        'first_city': [],
        'first_zip': [],
        'first_state': [],
        'first_country': [],
        'second_address': [],
        'second_city': [],
        'second_zip': [],
        'second_state': [],
        'second_country': [],
        'third_address': [],
        'third_city': [],
        'third_zip': [],
        'third_state': [],
        'third_country': [],
        'as_of_date': [],
        'first_alias': [],
        'first_alias_type': [],
        'second_alias': [],
        'second_alias_type': [],
        'first_name': [],
        'last_name': [],
        'title': [],
        'date_of_birth': [],
        'place_of_birth': [],
        'primary_citizenship': [],
        'secondary_citizenship': [],
        'tertiary_citizenship': []
    }

    # Iterate through the entries in the XML file
    for entry in root.findall('.//DistinctParty', namespaces=namespace):
        data['party_type'].append(entry.find('ofac:sdnType', namespace).text)
        data['name'].append(f"{entry.find('ofac:firstName', namespace).text} {entry.find('ofac:lastName', namespace).text}")
        data['registration_country'].append(entry.find('ofac:registrationCountry', namespace).text)
        data['official_identification_number'].append(entry.find('ofac:idNumber', namespace).text)
        data['official_identification_type'].append(entry.find('ofac:idType', namespace).text)
        data['official_identification_country'].append(entry.find('ofac:idCountry', namespace).text)
        data['first_address'].append(entry.find('ofac:address1', namespace).text)
        data['first_city'].append(entry.find('ofac:city', namespace).text)
        data['first_zip'].append(entry.find('ofac:zip', namespace).text)
        data['first_state'].append(entry.find('ofac:stateOrProvince', namespace).text)
        data['first_country'].append(entry.find('ofac:country', namespace).text)
        data['second_address'].append(entry.find('ofac:address2', namespace).text)
        data['second_city'].append(entry.find('ofac:address2City', namespace).text)
        data['second_zip'].append(entry.find('ofac:address2Zip', namespace).text)
        data['second_state'].append(entry.find('ofac:address2StateOrProvince', namespace).text)
        data['second_country'].append(entry.find('ofac:address2Country', namespace).text)
        data['third_address'].append(entry.find('ofac:address3', namespace).text)
        data['third_city'].append(entry.find('ofac:address3City', namespace).text)
        data['third_zip'].append(entry.find('ofac:address3Zip', namespace).text)
        data['third_state'].append(entry.find('ofac:address3StateOrProvince', namespace).text)
        data['third_country'].append(entry.find('ofac:address3Country', namespace).text)
        data['as_of_date'].append(entry.find('ofac:programList/ofac:program/ofac:date', namespace).text)
        data['first_alias'].append(entry.find('ofac:alias1', namespace).text)
        data['first_alias_type'].append(entry.find('ofac:aliasType1', namespace).text)
        data['second_alias'].append(entry.find('ofac:alias2', namespace).text)
        data['second_alias_type'].append(entry.find('ofac:aliasType2', namespace).text)
        data['first_name'].append(entry.find('ofac:firstName', namespace).text)
        data['last_name'].append(entry.find('ofac:lastName', namespace).text)
        data['title'].append(entry.find('ofac:title', namespace).text)
        data['date_of_birth'].append(entry.find('ofac:dateOfBirth', namespace).text)
        data['place_of_birth'].append(entry.find('ofac:placeOfBirth', namespace).text)
        data['primary_citizenship'].append(entry.find('ofac:citizenshipList/ofac:citizenship[1]', namespace).text)
        data['secondary_citizenship'].append(entry.find('ofac:citizenshipList/ofac:citizenship[2]', namespace).text)
        data['tertiary_citizenship'].append(entry.find('ofac:citizenshipList/ofac:citizenship[3]', namespace).text)

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)

    return df

# Specify the URL to download the OFAC SDN List XML file
ofac_xml_url = './data/input/sdn_advanced.xml'

# Call the function to parse the XML and get a DataFrame
#ofac_df = parse_ofac_xml(ofac_xml_url)

# Display the DataFrame
#print(ofac_df)

####
## Retrieve the OFAC Advanced SDNs from CSV
####
import pandas as pd
import numpy as np
import io
import requests
import re



def parse_ofac_csv(sdn_content_b, add_content_b, alt_content_b):

    # Load data into Pandas DataFrames
    sdn_content = sdn_content_b.decode('utf-8')
    sdn_df = pd.read_csv(io.StringIO(sdn_content), header=None,
                             names=["ent_num", "SDN_Name",
                             "SDN_Type", "Program",
                             "Title", "Call_Sign",
                             "Vess_type", "Tonnage", "GRT",
                             "Vess_flag", "Vess_owner",
                             "Remarks"])
    add_content = add_content_b.decode('utf-8')
    add_df = pd.read_csv(io.StringIO(add_content), header=None,
                         names=["ent_num", "Add_num", "Address",
                                "City/State/Province/Postal Code",
                                "Country", "Add_remarks"])
    alt_content = alt_content_b.decode('utf-8')
    alt_df = pd.read_csv(io.StringIO(alt_content), header=None,
                         names=["ent_num", "alt_num", "alt_type",
                                "alt_name", "alt_remarks"])

    # Combine DataFrames based on ent_num
    df = pd.merge(sdn_df, add_df, on="ent_num", how="outer")
    df = pd.merge(df, alt_df, on="ent_num", how="outer")

    return df

def extract_info(remarks):
    if pd.isna(remarks):  # Check if NaN
        return pd.Series({})

    info_dict = {}
    keywords = ['DOB', 'POB', 'nationality', 'gender', 'National ID NO.', 'Tax ID No.', 'Passport',
                'Registration Number']

    for keyword in keywords:
        start_index = remarks.find(keyword)
        if start_index != -1:
            start_index += len(keyword) + 1  # Skip the space after the keyword
            end_index = remarks.find(';', start_index)
            if end_index == -1:
                end_index = None
            info_dict[keyword] = remarks[start_index:end_index].strip()

    return pd.Series(info_dict)

def extract_country(col_value):
    if pd.isna(col_value):  # Check if NaN
        return np.nan

    matches = re.findall(r'\((.*?)\)', col_value)
    if matches:
        return matches[0].strip()
    else:
        return np.nan

response = requests.get('https://www.treasury.gov/ofac/downloads/sdn.csv')
if response.status_code == 200:
    # Get the content of the response
    sdn_content_b = response.content
response = requests.get('https://www.treasury.gov/ofac/downloads/add.csv')
if response.status_code == 200:
    # Get the content of the response
    add_content_b = response.content
response = requests.get('https://www.treasury.gov/ofac/downloads/alt.csv')
if response.status_code == 200:
    # Get the content of the response
    alt_content_b = response.content

# Sample data for the first file
combined_df = parse_ofac_csv(sdn_content_b, add_content_b, alt_content_b)

df_info = combined_df['Remarks'].apply(extract_info)
combined_df = pd.concat([combined_df, df_info], axis=1)
combined_df['Passport_Country'] = combined_df['Passport'].apply(extract_country)

combined_df.to_csv('./data/input/sdn_combined.csv')
# Print the combined DataFrame
print(combined_df)

####
## Generate fake party records
####
import random
from collections import OrderedDict
import pandas as pd
from faker import Faker
from faker.config import AVAILABLE_LOCALES
from deep_translator import GoogleTranslator
from icu import Transliterator

all_locales = [local for local in AVAILABLE_LOCALES]

# Ensure selecting locals that have at least address, person and a company provider
selected_locales = []
for locale in all_locales:
    fake = Faker(locale)

    filtered = [
        provider
        for provider in fake.get_providers()
            if f"person.{locale}" in str(provider.__class__)
               or f"address.{locale}" in str(provider.__class__)
               or f"company.{locale}" in str(provider.__class__)
    ]
    if len(filtered) == 3:
        selected_locales.append(filtered)
locales = [str(provider[0].__class__).split('.')[3] for provider in selected_locales]

fake = Faker(locales)
transliterator = Transliterator.createInstance('Any-Latin')

# TODO: Test distribution, correct transliteration, conditions (e.g, 2nd citizen != 1st)
# TODO: Parametrize distribution of aliases, birth attributes, addresses/nationalities completness
def generate_individual():
    current_locale = random.choice(locales)
    party_type = 'individual'
    first_name_elements = random.choices([1, 2, 3, 4, 5], [0.88, 0.06, 0.03, 0.02, 0.01]).pop()
    gender = random.choices(['male', 'female', 'nonbinary'], [0.45, 0.45, 0.1]).pop()
    if party_type == 'individual':
        if gender == 'male':
            first_name = ' '.join([transliterator.transliterate(fake[current_locale].first_name_male())
                                   for _ in range(first_name_elements)])
            last_name = transliterator.transliterate(fake[current_locale].last_name_male())
            name = f"{first_name}, {last_name}"
            first_alias = f"{'. '.join([name[0] for name in first_name.split(' ')])}, " \
                          f"{last_name}" if random.random() < 0.33 else None
            first_alias_type = random.choice(['strong', 'weak']) if first_alias else None
            second_alias = transliterator.transliterate(fake[current_locale].first_name_male())\
                if first_alias and random.random() < 0.5 else None
            second_alias_type = random.choice(['strong', 'weak']) if second_alias else None
            last_name = last_name
        if gender == 'female':
            first_name = ' '.join([transliterator.transliterate(fake[current_locale].first_name_female())
                                   for _ in range(first_name_elements)])
            last_name = transliterator.transliterate(fake[current_locale].first_name_female())
            name = f"{first_name}, {last_name}"
            first_alias = f"{'. '.join([name[0] for name in first_name.split(' ')])}, " \
                          f"{last_name}" if random.random() < 0.33 else None
            first_alias_type = random.choice(['strong', 'weak']) if first_alias else None
            second_alias = transliterator.transliterate(fake[current_locale].first_name_female()) \
                if first_alias and random.random() < 0.5 else None
            second_alias_type = random.choice(['strong', 'weak']) if second_alias else None
            last_name = last_name
        if gender == 'nonbinary':
            first_name = ' '.join([transliterator.transliterate(fake[current_locale].first_name_nonbinary())
                                   for _ in range(first_name_elements)])
            last_name = transliterator.transliterate(fake[current_locale].first_name_nonbinary())
            name = f"{first_name}, {last_name}"
            first_alias = f"{'. '.join([name[0] for name in first_name.split(' ')])}, " \
                          f"{last_name}" if random.random() < 0.33 else None
            first_alias_type = random.choice(['strong', 'weak']) if first_alias else None
            second_alias = transliterator.transliterate(fake[current_locale].first_name_nonbinary()) \
                if first_alias and random.random() < 0.5 else None
            second_alias_type = random.choice(['strong', 'weak']) if second_alias else None
            last_name = last_name
        title = transliterator.transliterate(fake[current_locale].job()) \
            if  random.random() < 0.1 else None
        date_of_birth = fake[current_locale].date_of_birth() \
            if  random.random() < 0.5 else None
        place_of_birth = transliterator.transliterate(fake[current_locale].city()) \
            if date_of_birth else None
        if random.random() < 0.2:
            official_identification_type = random.choices(['passport', 'ssn', 'other']).pop()

            if official_identification_type == 'passport':
                try:
                    official_identification_number = fake[current_locale].passport_number()
                except:
                    official_identification_number = None
                    print('Warning: No passport id generated')
            if official_identification_type == 'ssn':
                try:
                    official_identification_number = fake[current_locale].ssn()
                except:
                    official_identification_number = None
                    print('Warning: No ssn generated')
            if official_identification_type == 'other':
                try:
                    official_identification_number = random.randint(10000000000000, 99999999999999)
                except:
                    official_identification_number = None
                    print('Warning: No other id generated')
            official_identification_country = fake[current_locale].current_country() \
                if official_identification_number else None
        else:
            official_identification_number = None
            official_identification_type = None
            official_identification_country = None

        primary_citizenship = official_identification_country \
            if date_of_birth and random.random() < 0.8 else None
        registration_country = fake[current_locale].current_country() \
            if random.random() < 0.3 else None
        # country() returns a random country
        secondary_citizenship = fake[current_locale].country() \
            if primary_citizenship and random.random() < 0.2 else None
        tertiary_citizenship = fake[current_locale].country() \
            if secondary_citizenship and random.random() < 0.5 else None
    else:
        first_name = None
        name = None
        first_alias = None
        first_alias_type = None
        second_alias = None
        second_alias_type = None
        last_name = None
        title =  None
        date_of_birth =  None
        place_of_birth =  None
        official_identification_number = None
        official_identification_type = None
        registration_country = None
        official_identification_country = None
        primary_citizenship = None
        secondary_citizenship = None
        tertiary_citizenship = None

    first_address = transliterator.transliterate(fake[current_locale].street_address()) \
        if random.random() < 0.9 else None
    first_city = transliterator.transliterate(fake[current_locale].city()) \
        if first_address else None
    first_zip = fake[current_locale].postcode() if first_address else None
    try:
        first_state = transliterator.transliterate(fake[current_locale].state())\
            if first_address and random.random() < 0.5 else None
    except:
        first_state = None
        print('Warning: no state details created')
    first_country = fake[current_locale].current_country() if first_address else None
    second_address = transliterator.transliterate(fake[current_locale].street_address())\
        if random.random() < 0.1 else None
    second_city = transliterator.transliterate(fake[current_locale].city())\
        if second_address else None
    second_zip = transliterator.transliterate(fake[current_locale].postcode())\
        if second_address else None
    try:
        second_state = transliterator.transliterate(fake[current_locale].state())\
            if second_address and random.random() < 0.3 else None
    except:
        second_state = None
        print('Warning: no state details created')
    second_country = transliterator.transliterate(fake[current_locale].current_country())\
        if second_address else None
    third_address = transliterator.transliterate(fake[current_locale].street_address())\
        if random.random() < 0.01 else None
    third_city = transliterator.transliterate(fake[current_locale].city())\
        if third_address else None
    third_zip = fake[current_locale].postcode() if third_address else None
    try:
        third_state = transliterator.transliterate(fake[current_locale].state())\
            if third_address and random.random() < 0.3 else None
    except:
        third_state = None
        print('Warning: no state details created')
    third_country = transliterator.transliterate(fake[current_locale].current_country())\
        if third_address else None

    as_of_date = pd.to_datetime('now').date()

    return {
        'party_type': party_type,
        'name': name,
        'first_alias': first_alias,
        'first_alias_type': first_alias_type,
        'second_alias': second_alias,
        'second_alias_type': second_alias_type,
        'first_name': first_name,
        'last_name': last_name,
        'title': title,
        'date_of_birth': date_of_birth,
        'place_of_birth': place_of_birth,
        'official_identification_number': official_identification_number,
        'official_identification_type': official_identification_type,
        'official_identification_country': official_identification_country,
        'primary_citizenship': primary_citizenship,
        'secondary_citizenship': secondary_citizenship,
        'tertiary_citizenship': tertiary_citizenship,
        'registration_country': registration_country,
        'first_address': first_address,
        'first_city': first_city,
        'first_zip': first_zip,
        'first_state': first_state,
        'first_country': first_country,
        'second_address': second_address,
        'second_city': second_city,
        'second_zip': second_zip,
        'second_state': second_state,
        'second_country': second_country,
        'third_address': third_address,
        'third_city': third_city,
        'third_zip': third_zip,
        'third_state': third_state,
        'third_country': third_country,
        'as_of_date': as_of_date,
    }

def generate_entity():
    current_locale = random.choice(locales)
    party_type = 'entity'
    entity_name = transliterator.transliterate(fake[current_locale].company())

    if random.random() < 0.2:
        official_identification_type = random.choices(['vat', 'other']).pop()

        if official_identification_type == 'vat':
            try:
                official_identification_number = fake[current_locale].vat_id()
            except:
                official_identification_number = None
                print('Warning: No VAT id generated')
        if official_identification_type == 'other':
            try:
                official_identification_number = f"{'C'}-{(random.randint(10000000000000, 99999999999999))}"
            except:
                official_identification_number = None
                print('Warning: No other id generated')
        official_identification_country = fake[current_locale].current_country() \
            if official_identification_number else None
    else:
        official_identification_number = None
        official_identification_type = None
        official_identification_country = None

    first_address = transliterator.transliterate(fake[current_locale].street_address()) \
        if random.random() < 0.9 else None
    first_city = transliterator.transliterate(fake[current_locale].city()) \
        if first_address else None
    first_zip = fake[current_locale].postcode() if first_address else None
    try:
        first_state = transliterator.transliterate(fake[current_locale].state()) \
            if first_address and random.random() < 0.5 else None
    except:
        first_state = None
        print('Warning: no state details created')
    first_country = fake[current_locale].current_country() if first_address else None
    second_address = transliterator.transliterate(fake[current_locale].street_address()) \
        if random.random() < 0.1 else None
    second_city = transliterator.transliterate(fake[current_locale].city()) \
        if second_address else None
    second_zip = transliterator.transliterate(fake[current_locale].postcode()) \
        if second_address else None
    try:
        second_state = transliterator.transliterate(fake[current_locale].state()) \
            if second_address and random.random() < 0.3 else None
    except:
        second_state = None
        print('Warning: no state details created')
    second_country = transliterator.transliterate(fake[current_locale].current_country()) \
        if second_address else None
    third_address = transliterator.transliterate(fake[current_locale].street_address()) \
        if random.random() < 0.01 else None
    third_city = transliterator.transliterate(fake[current_locale].city()) \
        if third_address else None
    third_zip = fake[current_locale].postcode() if third_address else None
    try:
        third_state = transliterator.transliterate(fake[current_locale].state()) \
            if third_address and random.random() < 0.3 else None
    except:
        third_state = None
        print('Warning: no state details created')
    third_country = transliterator.transliterate(fake[current_locale].current_country())\
    if third_address else None

    registration_country = fake[current_locale].current_country() if random.random() < 0.3 else None

    as_of_date = pd.to_datetime('now').date()

    return {
        'party_type': party_type,
        'name': entity_name,
        'registration_country': registration_country,
        'official_identification_number': official_identification_number,
        'official_identification_type': official_identification_type,
        'official_identification_country': official_identification_country,
        'first_address': first_address,
        'first_city': first_city,
        'first_zip': first_zip,
        'first_state': first_state,
        'first_country': first_country,
        'second_address': second_address,
        'second_city': second_city,
        'second_zip': second_zip,
        'second_state': second_state,
        'second_country': second_country,
        'third_address': third_address,
        'third_city': third_city,
        'third_zip': third_zip,
        'third_state': third_state,
        'third_country': third_country,
        'as_of_date': as_of_date,
    }

def generate_vessel():
    # TODO: Overwrite address for vessels with None
    current_locale = random.choice(locales)
    party_type = 'vessel'
    vessel_name = transliterator.transliterate(fake[current_locale].word().split()[0])
    first_alias = transliterator.transliterate(fake[current_locale].word().split()[0])\
        if random.random() < 0.33 else None
    first_alias_type = random.choice(['strong', 'weak']) if first_alias else None
    second_alias = transliterator.transliterate(fake[current_locale].word().split()[0])\
        if first_alias and random.random() < 0.5 else None
    second_alias_type = random.choice(['strong', 'weak']) if second_alias else None
    registration_country = fake[current_locale].current_country()
    as_of_date = pd.to_datetime('now').date()

    return {
        'party_type': party_type,
        'name': vessel_name,
        'registration_country': registration_country,
        'first_alias': first_alias,
        'first_alias_type': first_alias_type,
        'second_alias': second_alias,
        'second_alias_type': second_alias_type,
        'as_of_date': as_of_date,
    }

def generate_fake_data(num_records):
    fake_data = []
    for _ in range(num_records):
        party_type = random.choices(['individual', 'entity', 'vessel'], weights=[0.6, 0.3, 0.1])[0]
        if party_type == 'individual':
            fake_data.append(generate_individual()) # TODO: possible index-out-of-bound condition
        elif party_type == 'entity':
            fake_data.append(generate_entity())
        elif party_type == 'vessel':
            fake_data.append(generate_vessel())
    return fake_data

# Generate valid fake data for background noise on 100 records
fake_global_party_list = generate_fake_data(5000)

# TODO: Add in actually sanctioned parties

# TODO: Apply identity permutation patters to both to noise and actual records

# Convert the list of dictionaries to a DataFrame
synthetic_df = pd.DataFrame(fake_global_party_list)

# Print the DataFrame
print(synthetic_df)
