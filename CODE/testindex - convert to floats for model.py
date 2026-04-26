# import csv

# def find_value_index(search_value):
#     # Open the CSV file
#     with open('[REDACTED_BY_SCRIPT]', 'r', encoding='utf-8') as file:
#         # Read the first line
#         line = file.readline()
#         # Split the line by commas
#         values = line.split(',')
        
#         # Try to find the value in the list
#         try:
#             index = values.index(search_value)
#             print(f"Found '{search_value}' at index {index}")
#         except ValueError:
#             print(f"'{search_value}' not found in the file")

# # Get input from user
# search_term = input("[REDACTED_BY_SCRIPT]")
# find_value_index(search_term)

# import csv

# def find_value_index(search_value):
#     # Open the CSV file
#     with open('[REDACTED_BY_SCRIPT]', 'r', encoding='utf-8') as file:
#         # Read the first line
#         line = file.readline()
#         # Split the line by commas
#         values = line.split(',')
        
#         # Try to find the value in the list
#         try:
#             index = values.index(search_value)
#             print(f"Found '{search_value}' at index {index}")
#         except ValueError:
#             print(f"'{search_value}' not found in the file")

# # Get input from user
# search_term = "247 m2"
# find_value_index(search_term)


import csv

# List of indices to extract


def extract_values_at_indices(file_path,indices):
    all_rows_values = []
    
    # Open and read the CSV file
    with open(file_path, 'r', encoding='utf-8') as file:
        # Read all lines
        lines = file.readlines()
        
        # Process each line
        for line in lines:
            # Split the line by commas
            values = line.split(',')
            
            # Extract values at specified indices
            row_values = []
            for index in indices:
                try:
                    # Remove any quotes and whitespace
                    value = values[index].strip().strip('"')
                    row_values.append(value)
                except IndexError:
                    # Handle cases where index is out of range
                    row_values.append("N/A")
            
            all_rows_values.append(row_values)
            
            # Print the extracted values for this row
    
    return all_rows_values


#Chimnie
indices = [0,1,2,3,4,5,6,7,8,9,10,11,17,18,19,20,21,22,23,24,25,26,27,32,33,34,35,36,37]
file_path = '[REDACTED_BY_SCRIPT]'
extracted_values1 = extract_values_at_indices(file_path,indices)

#Homipi
indices = [10,11,12,13,14,15,16,17,18,19,21,23,24,41,74,85,96]
file_path = r'[REDACTED_BY_SCRIPT]'
extracted_values2 = extract_values_at_indices(file_path,indices)

#Streetscan2
indices = [368,369,370,371,372,373,374,375,376,377,378,379,380]
file_path = r'[REDACTED_BY_SCRIPT]'
extracted_values3 = extract_values_at_indices(file_path,indices)

#BnL
indices = [0,2,3,4,10,14,21,55]
file_path = r'[REDACTED_BY_SCRIPT]'
extracted_values4 = extract_values_at_indices(file_path,indices)

extracted_values = []
for i in range(len(extracted_values1)):
    extracted_valuesI = extracted_values1[i] + extracted_values2[i] + extracted_values3[i] + extracted_values4[i]
    extracted_values.append(extracted_valuesI)

for i in range(len(extracted_values)-1):
    i+=1
    #cnverts caluation to real numbers
    extracted_values[i][0] = extracted_values[i][0].replace('Â£', '').replace('K', '000')
    extracted_values[i].append(extracted_values[i][0][(extracted_values[i][0].index('-'))+1:])
    extracted_values[i][0] = extracted_values[i][0][:extracted_values[i][0].index('-')]

    if 'HIGH' in extracted_values[i][11]:
        extracted_values[i][11] = extracted_values[i][11].replace('HIGH', '2')
    else:pass
    if 'HIGH' in extracted_values[i][22]:
        extracted_values[i][22] = extracted_values[i][22].replace('HIGH', '2')
    else:pass

    try:extracted_values[i][18] = extracted_values[i][18].replace(' sqft', '')
    except:extracted_values[i][18] = ""

    if 'Detached' in extracted_values[i][23]:
        extracted_values[i][23] = extracted_values[i][23].replace('Detached', '2')
    elif 'Semi-detached' in extracted_values[i][23]:
        extracted_values[i][23] = extracted_values[i][23].replace('Semi-detached', '1')
    elif 'Terraced' in extracted_values[i][23]:
        extracted_values[i][23] = extracted_values[i][23].replace('Terraced', '0')
    else:extracted_values[i][23] = ""

    if 'Freehold' in extracted_values[i][24]:
        extracted_values[i][24] = extracted_values[i][24].replace('Freehold', '1')
    elif 'Leasehold' in extracted_values[i][24]:
        extracted_values[i][24] = extracted_values[i][24].replace('Leasehold', '0')
    else:extracted_values[i][24] = ""

    #epc rating
    extracted_values[i].append(extracted_values[i][25][-2:-1])
    extracted_values[i][25] = extracted_values[i][25][0]
    extracted_values[i][25] = extracted_values[i][25] + " "
    if 'D' in extracted_values[i][25]:
        extracted_values[i][25] = extracted_values[i][25].replace('D', '3')
    elif 'A' in extracted_values[i][25]:
        extracted_values[i][25] = extracted_values[i][25].replace('A', '0')
    elif 'B' in extracted_values[i][25]:
        extracted_values[i][25] = extracted_values[i][25].replace('B', '1')
    elif 'C' in extracted_values[i][25]:
        extracted_values[i][25] = extracted_values[i][25].replace('C', '2')
    elif 'E' in extracted_values[i][25]:
        extracted_values[i][25] = extracted_values[i][25].replace('E', '4')
    elif 'F' in extracted_values[i][25]:
        extracted_values[i][25] = extracted_values[i][25].replace('F', '5')
    else:extracted_values[i][25] = " "
    extracted_values[i][25] = extracted_values[i][25].replace(" ","")

    extracted_values[i][-1] = extracted_values[i][-1] + " "
    if 'D' in extracted_values[i][-1]:
        extracted_values[i][-1] = extracted_values[i][-1].replace('D', '3')
    elif 'A' in extracted_values[i][-1]:
        extracted_values[i][-1] = extracted_values[i][-1].replace('A', '0')
    elif 'B' in extracted_values[i][-1]:
        extracted_values[i][-1] = extracted_values[i][-1].replace('B', '1')
    elif 'C' in extracted_values[i][-1]:
        extracted_values[i][-1] = extracted_values[i][-1].replace('C', '2')
    elif 'E' in extracted_values[i][-1]:
        extracted_values[i][-1] = extracted_values[i][-1].replace('E', '4')
    elif 'F' in extracted_values[i][-1]:
        extracted_values[i][-1] = extracted_values[i][-1].replace('F', '5')
    else:extracted_values[i][-1] = " "
    extracted_values[i][-1] = extracted_values[i][-1].replace(" ","")

    if "Detached" in extracted_values[i][29]:
        extracted_values[i][29] = extracted_values[i][29].replace('Detached', '2')
    elif "Semi-detached" in extracted_values[i][29]:
        extracted_values[i][29] = extracted_values[i][29].replace('Semi-detached', '1')
    elif "Terraced" in extracted_values[i][29]:
        extracted_values[i][29] = extracted_values[i][29].replace('Terraced', '0')
    else:extracted_values[i][29] = ""

    extracted_values[i][27] = extracted_values[i][27].replace(' sqft', '')
    extracted_values[i][28] = extracted_values[i][28].replace(' sqft', '')

    extracted_values[i][36] = extracted_values[i][36][-2:].replace(' ','')

    if "Freehold" in extracted_values[i][37]:
        extracted_values[i][37] = extracted_values[i][37].replace('Freehold','1')
    elif "Leased" in extracted_values[i][37]:
        extracted_values[i][37] = extracted_values[i][37].replace('Leased','0')
    else:extracted_values[i][37] = ""

    if "Receps" in extracted_values[i][38]:
        extracted_values[i][38] = extracted_values[i][38].replace(' Receps', '')
    else:extracted_values[i][38] = ""

    #epc rating
    if 'D' in extracted_values[i][39]:
        extracted_values[i][39] = extracted_values[i][39].replace('D', '3')
    elif 'A' in extracted_values[i][39]:
        extracted_values[i][39] = extracted_values[i][39].replace('A', '0')
    elif 'B' in extracted_values[i][39]:
        extracted_values[i][39] = extracted_values[i][39].replace('B', '1')
    elif 'C' in extracted_values[i][39]:
        extracted_values[i][39] = extracted_values[i][39].replace('C', '2')
    elif 'E' in extracted_values[i][39]:
        extracted_values[i][39] = extracted_values[i][39].replace('E', '4')
    elif 'F' in extracted_values[i][39]:
        extracted_values[i][39] = extracted_values[i][39].replace('F', '5')
    else:extracted_values[i][39] = " "
    extracted_values[i][39] = extracted_values[i][39].replace(" ","")

    #date house was built
    try: 
        extracted_values[i][40] = extracted_values[i][40].replace('(', ' ')
        extracted_values[i][40] = extracted_values[i][40][-6:-2]   
    except:extracted_values[i][40] = "2025"

    extracted_values[i][42] = extracted_values[i][42].replace('Â£', ' ')
    extracted_values[i][42] = extracted_values[i][42].replace('.', '')
    extracted_values[i][44] = extracted_values[i][44].replace('Â£', ' ')
    extracted_values[i][44] = extracted_values[i][44].replace('.', '')

    try:extracted_values[i][43] = extracted_values[i][43].replace('High', '1')
    except:extracted_values[i][43] = "0"

    extracted_values[i][46] = extracted_values[i][46][0]
    extracted_values[i][47] = extracted_values[i][47][3]
    extracted_values[i][49] = extracted_values[i][49][3]
    extracted_values[i][50] = extracted_values[i][50][3]
    extracted_values[i][51] = extracted_values[i][51][3]
    extracted_values[i][52] = extracted_values[i][52][3]
    extracted_values[i][53] = extracted_values[i][53][3]
    extracted_values[i][54] = extracted_values[i][54][3]
    extracted_values[i][55] = extracted_values[i][55][3]
    extracted_values[i][56] = extracted_values[i][56][3]
    extracted_values[i][57] = extracted_values[i][57][3]
    extracted_values[i][58] = extracted_values[i][58][3]

    extracted_values[i][59] = extracted_values[i][59].replace(' ft2','')
    try:extracted_values[i][59] = extracted_values[i][59].replace('.','')
    except:extracted_values[i][59] = extracted_values[i][59]
    
    if 'High' in extracted_values[i][60]:
        extracted_values[i][60] = extracted_values[i][60].replace('High', "2")
    elif 'Medium' in extracted_values[i][60]:
        extracted_values[i][60] = extracted_values[i][60].replace('Medium', "1")
    else:extracted_values[i][60] = ""
    
    #no. houses plot data
    try:
        plD1 = extracted_values[i][61][extracted_values[i][61].index("on ") + 3:extracted_values[i][61].index(" out")]
        plD1.replace(' ', "")
        plD1 = int(plD1)
        plD2 = extracted_values[i][61][extracted_values[i][61].index("of ") + 3:extracted_values[i][61].index(" other")]
        plD2.replace(' ', "")
        plD2 = int(plD2)
        extracted_values[i][61] = (plD1/plD2)
    except:extracted_values[i][61] = ""

    extracted_values[i][62] = extracted_values[i][62].replace(' ', "")
    if 'D' in extracted_values[i][62]:
        extracted_values[i][62] = extracted_values[i][62].replace('D', '3')
    elif 'A' in extracted_values[i][62]:
        extracted_values[i][62] = extracted_values[i][62].replace('A', '0')
    elif 'B' in extracted_values[i][62]:
        extracted_values[i][62] = extracted_values[i][62].replace('B', '1')
    elif 'C' in extracted_values[i][62]:
        extracted_values[i][62] = extracted_values[i][62].replace('C', '2')
    elif 'D' in extracted_values[i][62]:
        extracted_values[i][62] = extracted_values[i][62].replace('D', '3')
    elif 'E' in extracted_values[i][62]:
        extracted_values[i][62] = extracted_values[i][62].replace('E', '4')
    elif 'F' in extracted_values[i][62]:
        extracted_values[i][62] = extracted_values[i][62].replace('F', '5')
    else:extracted_values[i][62] = " "
    extracted_values[i][62] = extracted_values[i][62].replace(" ","")

    extracted_values[i][63] = extracted_values[i][63].replace(' m2', "")

    extracted_values[i][64] = extracted_values[i][64] + " "
    if 'D' in extracted_values[i][64]:
        extracted_values[i][64] = extracted_values[i][64].replace('D', '3')
    elif 'A' in extracted_values[i][64]:
        extracted_values[i][64] = extracted_values[i][64].replace('A', '0')
    elif 'B' in extracted_values[i][64]:
        extracted_values[i][64] = extracted_values[i][64].replace('B', '1')
    elif 'C' in extracted_values[i][64]:
        extracted_values[i][64] = extracted_values[i][64].replace('C', '2')
    elif 'D' in extracted_values[i][64]:
        extracted_values[i][64] = extracted_values[i][64].replace('D', '3')
    elif 'E' in extracted_values[i][64]:
        extracted_values[i][64] = extracted_values[i][64].replace('E', '4')
    elif 'F' in extracted_values[i][64]:
        extracted_values[i][64] = extracted_values[i][64].replace('F', '5')
    else:extracted_values[i][64] = " "
    extracted_values[i][64] = extracted_values[i][64].replace(" ","")

    extracted_values[i][65] = extracted_values[i][65].replace('Â£', ' ').replace('.','')
    extracted_values[i][66] = extracted_values[i][66][0]
    
    try:extracted_values[i][18] = int(extracted_values[i][18])
    except:extracted_values[i][18] = ""
    extracted_values[i][18] = str(extracted_values[i][18])

    # indices_to_remove = {62, 48, 41, 28, 26}  # Using a set for faster lookup
    # extracted_values[i] = [value for j, value in enumerate(extracted_values[i]) if j not in indices_to_remove]

    










def lists_to_csv(list_of_lists, output_file_path):
    # Open the file in write mode with newline='' to handle line endings properly
    with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        # Create a CSV writer object
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)  # QUOTE_ALL ensures all fields are quoted
        
        # Write each list as a row in the CSV file
        for row in list_of_lists:
            writer.writerow(row)

# Example usage:
output_path = r'[REDACTED_BY_SCRIPT]'
lists_to_csv(extracted_values, output_path)

