import csv
class gen_utils:
    def load_data(self, file_name):
        with open(f'{file_name}', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                x, y = row
                self.coords.append([float(x)/3, float(y)/3, 0])
        file.close()
    