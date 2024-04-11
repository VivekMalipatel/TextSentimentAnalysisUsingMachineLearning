import csv

file_path = 'go_emotions_pre_processed.csv'
output_file_path_direct = 'Converted_Dataset.csv'

with open(file_path, 'r', newline='') as input_csvfile, open(output_file_path_direct, 'w', newline='') as output_csvfile:
    reader = csv.DictReader(input_csvfile)
    writer = csv.writer(output_csvfile)
    # Writing the header for the new file
    writer.writerow(['text', 'labels'])
    for row in reader:
        text = row['text']
        emotions = [int(row[emotion]) for emotion in reader.fieldnames[2:]]  # Skip 'id' and 'text'
        writer.writerow([text, emotions])
