import json
import csv

fileName= "BnL"

def replace_commas_in_values(data):
    """[REDACTED_BY_SCRIPT]"""
    if isinstance(data, dict):
        return {k: replace_commas_in_values(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_commas_in_values(item) for item in data]
    elif isinstance(data, str):
        return data.replace(',', '.')
    return data

def process_json_file(input_path, output_path):
    # Read input JSON file
    with open(input_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    # Process data
    modified_data = replace_commas_in_values(original_data)
    
    # Write output JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(modified_data, f, indent=4, ensure_ascii=False)


# Example usage
input_json = r'[REDACTED_BY_SCRIPT]' + fileName + '.json'
output_json = r'[REDACTED_BY_SCRIPT]' + fileName + '.json'
process_json_file(input_json, output_json)








def json_to_csv(json_file_path, csv_file_path):
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
    
    if isinstance(data, dict):
        data = [data]
    
    fieldnames = set()
    for item in data:
        fieldnames.update(item.keys())
    fieldnames = sorted(fieldnames)
    
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(
            csv_file, 
            fieldnames=fieldnames,
            quoting=csv.QUOTE_ALL,  # Force quoting for all fields
            quotechar='"',
            escapechar='\\'
        )
        writer.writeheader()
        
        for item in data:
            row = {}
            for field in fieldnames:
                value = item.get(field)
                # Explicit type conversion with special handling
                if value is None:
                    converted = ''
                elif isinstance(value, (list, dict)):
                    converted = json.dumps(value)
                else:
                    converted = str(value)
                row[field] = converted
            writer.writerow(row)

# Example usage
if __name__ == "__main__":
    #json_to_csv(r"[REDACTED_BY_SCRIPT]", r"[REDACTED_BY_SCRIPT]")
    input1=r"[REDACTED_BY_SCRIPT]" + fileName + ".json"
    output1=r"[REDACTED_BY_SCRIPT]" + fileName + ".csv"
    json_to_csv(input1, output1)


def remove_newlines(input_file, output_file):
    with open(input_file, mode='r', newline='', encoding='utf-8') as infile, \
         open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.reader(infile)
        # Configure writer to quote all fields
        writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)
        
        for row in reader:
            # Remove newline characters and tabs from each cell
            cleaned_row = [cell.replace('\n', ' ').replace('  ', '') for cell in row]
            writer.writerow(cleaned_row)

# Example usage
input_csv = r"[REDACTED_BY_SCRIPT]" + fileName + ".csv"
output_csv = r"[REDACTED_BY_SCRIPT]" + fileName + "2.csv"
remove_newlines(input_csv, output_csv)