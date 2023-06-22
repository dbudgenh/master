import pandas as pd

def main():
    mapping = get_mapping(csv_file='C:/Users/david/Desktop/Python/master/data/birds.csv')
    print(mapping)

def get_mapping(csv_file):
    bird_frame = pd.read_csv(csv_file,delimiter=',')
    unique_ids = bird_frame.drop_duplicates(subset='class id')
    mapping = dict(zip(unique_ids['class id'].astype(int),unique_ids['labels']))
    return mapping


if __name__ == '__main__':
    main()