import pandas as pd
import os
#.\venv\Scripts\activate

# Create a sample DataFrame with column names
data = {'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
    }

df = pd.DataFrame(data)

# # Adding new row to df for V2
new_row_loc = {'Name': 'GF1', 'Age': 20, 'City': 'City1'}
df.loc[len(df.index)] = new_row_loc

# # Adding new row to df for V3
new_row_loc2 = {'Name': 'GF2', 'Age': 30, 'City': 'City2'}
df.loc[len(df.index)] = new_row_loc2

# Ensure the "data" directory exists at the root level
data_dir = 'data'
os.makedirs(data_dir, exist_ok=True)

# Define the file path
file_path = os.path.join(data_dir, 'sample_data.csv')

# Save the DataFrame to a CSV file, including column names
df.to_csv(file_path, index=False)

print(f"CSV file saved to {file_path}")



'''# mycode.py
import pandas as pd
import os

def create_initial_data():
    if not os.path.exists('data'):
        os.makedirs('data')
    
    df = pd.DataFrame({
        'Name': ['Alice', 'Bob'],
        'Age': [25, 30],
        'City': ['New York', 'London']
    })
    df.to_csv('data/sample_data.csv', index=False)
    print("Initial data created!")

def append_new_data():
    df = pd.read_csv('data/sample_data.csv')
    new_row = {'Name': 'Rina', 'Age': 35, 'City': 'Paris'}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv('data/sample_data.csv', index=False)
    print("New data appended!")

def append_more_data():
    df = pd.read_csv('data/sample_data.csv')
    new_row = {'Name': 'Pina', 'Age': 28, 'City': 'Tokyo'}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv('data/sample_data.csv', index=False)
    print("More data appended!")

# -----------------------
if __name__ == "__main__":
    # Run in order
    create_initial_data()   # only creates if needed
    append_new_data()       # adds Rina
    append_more_data()      # adds pina

    
    '''