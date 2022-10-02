import numpy as np, json, pickle

__locations = None
__data_columns = None
__model = None

def load_saved_artifacts():
    print("Loading Artifacts...")
    global __data_columns
    global __locations
    global __model

    # __data_columns = json.load(open('artifacts/columns.json','r'))
    # __locations = __data_columns['data_columns'][3:]
    # __model = pickle.load(open('artifacts/bhp.pkl','rb'))

    with open("./artifacts/columns.json", 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    with open("./artifacts/bhp.pickle", 'rb') as f:
        __model = pickle.load(f)
    
    print("Artifacts Loaded.")

def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0], 2)

def get_location_names():
    return __locations

if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar', 1000, 3, 3))
    print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
    print(get_estimated_price('Kalhalli', 1000, 2, 2))
    print(get_estimated_price('Ejipurar', 1000, 2, 2))