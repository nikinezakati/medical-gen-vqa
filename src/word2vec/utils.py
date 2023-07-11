
import csv
def read_lines(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        lines = [item.replace("\n", "") for item in lines]
        return lines

def write_lines(file, lines, delimiter):
    with open(file, 'w', encoding="utf-8") as f:
        f.write(f"{delimiter}".join(lines))
        print(f"Written {len(lines)} lines to \"{file}\"")

def read_tsv(file):
    with open(file, 'r', encoding="utf-8") as f:
        data = f.read()
        data = data.split("\t")
        return data

def save_csv(csv_columns, csv_data, url):
    with open(url, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in csv_data:
            writer.writerow(data)