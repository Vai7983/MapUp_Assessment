import pandas as pd


def generate_car_matrix(datasets/dataset-1.csv)->pd.DataFrame:
    """
    Creates a DataFrame  for id combinations.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values, 
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    # Read the dataset into a DataFrame
    df = pd.read_csv(datasets/dataset-1.csv)

    # Pivot the DataFrame to create the desired matrix
    car_matrix = df.pivot(index='id_1', columns='id_2', values='car').fillna(0)

    # Set diagonal values to 0
    car_matrix.values[[range(car_matrix.shape[0])]*2] = 0

    return car_matrixs


def get_type_count(datasets/dataset-1.csv)->dict:
    """
    Categorizes 'car' values into types and returns a dictionary of counts.

    Args:
        df (pandas.DataFrame)

    Returns:
        dict: A dictionary with car types as keys and their counts as values.
    """
     # Read the dataset into a DataFrame
    df = pd.read_csv(datasets/dataset-1.csv)

    # Define the conditions for categorizing car_type
    conditions = [
        (df['car'] <= 15),
        (df['car'] > 15) & (df['car'] <= 25),
        (df['car'] > 25)
    ]

    # Define the corresponding values for car_type
    car_type_values = ['low', 'medium', 'high']

    # Add a new column 'car_type' based on the conditions
    df['car_type'] = pd.cut(df['car'], bins=[-float('inf'), 15, 25, float('inf')], labels=car_type_values, include_lowest=True)

    # Count the occurrences of each car_type
    type_count = df['car_type'].value_counts().to_dict()

    # Sort the dictionary alphabetically based on keys
    type_count = dict(sorted(type_count.items()))

    return type_count

def get_bus_indexes(datasets/dataset-1.csv)->list:
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """
    # Read the dataset into a DataFrame
    df = pd.read_csv(datasets/dataset-1.csv)

    # Calculate the mean value of the 'bus' column
    bus_mean = df['bus'].mean()

    # Identify indices where 'bus' values are greater than twice the mean
    bus_indexes = df[df['bus'] > 2 * bus_mean].index.tolist()

    # Sort the indices in ascending order
    bus_indexes.sort()

    return bus_indexes


def filter_routes(datasets/dataset-1.csv)->list:
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """
    # Group by 'route' and calculate the average of 'truck' for each route
    route_avg_truck = df.groupby('route')['truck'].mean()

    # Filter routes where the average 'truck' value is greater than 7
    filtered_routes = route_avg_truck[route_avg_truck > 7].index.tolist()

    # Sort the list of routes
    filtered_routes.sort()

    return filtered_routes


def multiply_matrix(matrix)->pd.DataFrame:
    """
    Multiplies matrix values with custom conditions.

    Args:
        matrix (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Modified matrix with values multiplied based on custom conditions.
    """
    # Create a deep copy of the input DataFrame to avoid modifying the original
    modified_matrix = matrix.copy()

    # Apply the specified logic to modify values
    modified_matrix[modified_matrix > 20] *= 0.75
    modified_matrix[modified_matrix <= 20] *= 1.25

    # Round the values to 1 decimal place
    modified_matrix = modified_matrix.round(1)

    return modified_matrix


def time_check(df)->pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(datasets/dataset-2.csv)

    # Combine date and time columns to create a datetime column
    df['timestamp'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    
    # Extract day of the week and hour of the day
    df['day_of_week'] = df['timestamp'].dt.day_name()
    df['hour_of_day'] = df['timestamp'].dt.hour

    # Check if each (id, id_2) pair has correct timestamps
    correct_timestamps = (
        (df.groupby(['id', 'id_2', 'day_of_week'])['hour_of_day'].nunique() == 24) &
        (df.groupby(['id', 'id_2'])['day_of_week'].nunique() == 7)
    )

    return correct_timestamps
