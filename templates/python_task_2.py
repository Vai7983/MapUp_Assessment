import pandas as pd


def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
     # Read the CSV file into a DataFrame
    df = pd.read_csv(datasets/dataset-3.csv)

    # Create a pivot table with id_1 and id_2 as index and columns, and distance as values
    distance_matrix = df.pivot_table(index='id_1', columns='id_2', values='distance', fill_value=0)

    # Make the matrix symmetric
    distance_matrix = distance_matrix.add(distance_matrix.T, fill_value=0)

    # Set diagonal values to 0
    distance_matrix.values[[range(distance_matrix.shape[0])]*2] = 0

    # Iterate to update cumulative distances
    for col in distance_matrix.columns:
        for row in distance_matrix.index:
            if distance_matrix.loc[row, col] == 0 and row != col:
                # Find intermediate points to calculate cumulative distance
                intermediates = distance_matrix.loc[row].index.intersection(distance_matrix[col].index)
                intermediates = intermediates[intermediates != row]
                intermediates = intermediates[intermediates != col]

                # Update cumulative distance
                cumulative_distance = sum(
                    distance_matrix.loc[row, intermediate] + distance_matrix.loc[intermediate, col]
                    for intermediate in intermediates
                )
                distance_matrix.loc[row, col] = cumulative_distance

    return distance_matrix

def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Extract upper triangle of the distance matrix (excluding diagonal)
    upper_triangle = distance_matrix.where(np.triu(np.ones(distance_matrix.shape), k=1).astype(bool))

    # Reset the index and stack the DataFrame to get a Series
    stacked_series = upper_triangle.stack()

    # Convert the Series to a DataFrame and reset the index
    distance_df = stacked_series.reset_index()

    # Rename the columns
    distance_df.columns = ['id_start', 'id_end', 'distance']

    return distance_df


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Filter the DataFrame based on the reference_value
    reference_df = df[df['id_start'] == reference_value]

    # Calculate the average distance for the reference_value
    reference_avg_distance = reference_df['distance'].mean()

    # Calculate the lower and upper thresholds within 10%
    lower_threshold = reference_avg_distance - 0.1 * reference_avg_distance
    upper_threshold = reference_avg_distance + 0.1 * reference_avg_distance

    # Filter the DataFrame based on the thresholds
    within_threshold_df = df[(df['distance'] >= lower_threshold) & (df['distance'] <= upper_threshold)]

    # Get the unique values from the id_start column and sort them
    result_ids = sorted(within_threshold_df['id_start'].unique())

    return result_ids

def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Define rate coefficients for each vehicle type
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    # Iterate over vehicle types and calculate toll rates
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        # Calculate toll rates by multiplying distance with the rate coefficient
        df[vehicle_type] = df['distance'] * rate_coefficient

    return df


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Define time ranges and discount factors
    time_ranges = [
        (time(0, 0, 0), time(10, 0, 0), 0.8),
        (time(10, 0, 0), time(18, 0, 0), 1.2),
        (time(18, 0, 0), time(23, 59, 59), 0.8)
    ]
    
    weekend_discount_factor = 0.7

    # Create a new DataFrame to store the time-based toll rates
    time_based_toll_rates_df = pd.DataFrame()

    # Iterate over each time range
    for start_time, end_time, discount_factor in time_ranges:
        # Apply time range filters
        mask = (df['start_time'] >= start_time) & (df['end_time'] <= end_time)
        
        # Apply discount factor to vehicle columns based on the time range
        for vehicle_type in ['moto', 'car', 'rv', 'bus', 'truck']:
            df.loc[mask, vehicle_type] *= discount_factor

        # Append the result to the new DataFrame
        time_based_toll_rates_df = time_based_toll_rates_df.append(df[mask])

    # Apply constant discount factor for weekends
    weekend_mask = (df['start_day'].isin(['Saturday', 'Sunday']))
    for vehicle_type in ['moto', 'car', 'rv', 'bus', 'truck']:
        df.loc[weekend_mask, vehicle_type] *= weekend_discount_factor

    # Append the weekend results to the new DataFrame
    time_based_toll_rates_df = time_based_toll_rates_df.append(df[weekend_mask])

    return time_based_toll_rates_df
